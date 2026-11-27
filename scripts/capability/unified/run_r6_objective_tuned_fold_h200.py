#!/usr/bin/env python3
"""H200 completion driver for Result 6 full-cohort ESMFold2 structure evaluation.

Never mutates frozen historical GPFS trees (read-only scan of
``20261110074842_20277c6a196f/ot_structure_evidence/objects``). Writes all new
per-sequence fold objects and merged TSVs into a fresh ``--work-dir``:

  1. ``--mode prepare``: scans existing read-only GPFS ``objects/*/result.json``
     (``status == "ok"`` and matching pinned ``evaluation_signature``) across 32
     threads, joins against the 47,778-sequence main cohort and the 252
     unfolded ``long_cap`` sequences, and emits ``existing_ok.jsonl`` plus
     longest-first ``pending_to_fold.jsonl``.
  2. ``--mode fold --shard-index K --num-shards M --device cuda:G``: folds
     round-robin shard ``K`` of ``pending_to_fold.jsonl`` using offline
     ``ESMFold2-hf`` under the exact frozen measurement contract
     (``bfloat16``, no autocast, ``allow_tf32=False``, 32 diffusion samples,
     ``seed=20260905``, ``seed_rule="sequence_digest"``, highest-``pTM``
     selection, exact OOM replica splitting).
  3. ``--mode export``: asserts 100.0% completion (``47,778/47,778`` main
     cohort + ``2,000/2,000`` ``long_cap`` cohort) and emits
     ``checkpoint_structures_68of68.tsv`` and ``long_cap_structures_2000of2000.tsv``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation import structure_evidence as se  # noqa: E402

EXPECTED_SIGNATURE = "9dc43daab2888ea6db0746f9e6567c34e52b379048066d0218a9d920951a8660"
NUM_SHARDS_68 = 68
FOLD_SEED = 20260905
FOLD_SEED_RULE = "sequence_digest"
NUM_DIFFUSION_SAMPLES = 32

PINNED_MODEL_FILES_SHA256 = {
    "config.json": "a821096ea81dd82bed9a78b5e3ab3b270c0d473c4566f09f94c063f256d52240",
    "model-00001-of-00006.safetensors": "a24f050e2a2f9746b43b435c53719afc575a374fa302731b6911356a3d8b5cac",
    "model-00002-of-00006.safetensors": "1af1615f6f6d1b3c0a24a63d9052a0ecad5b3831d1513c1a5fea73870c1bec73",
    "model-00003-of-00006.safetensors": "357b55ec56192fd01a1cfc39a5260caf2242503ac7c554f614e43c7a469f0ced",
    "model-00004-of-00006.safetensors": "f065aca1a13308bce41b78f728c28a0a29e3a9db6af531a74a258dbb06f8721b",
    "model-00005-of-00006.safetensors": "e0c507dce5b2ddb22f77305607b4332506137850c5378e98e9729345264668b7",
    "model-00006-of-00006.safetensors": "90015eb13bbeab54237b18d6748dbd47f4bfd108fdf45704d66e5c3a93f8df7c",
}

_SAMPLE_OUTPUT_FIELDS = (
    "sample_atom_coords", "plddt_logits", "plddt", "plddt_per_atom", "plddt_ca",
    "complex_plddt", "complex_iplddt", "pae_logits", "pae", "pde_logits", "pde",
    "resolved_logits", "ptm", "iptm", "pair_chains_iptm",
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _scan_one_existing(obj_dir: Path) -> dict[str, Any] | None:
    res_path = obj_dir / "result.json"
    if not res_path.is_file():
        return None
    try:
        rec = json.loads(res_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if rec.get("status") != "ok":
        return None
    if rec.get("evaluation_signature") != EXPECTED_SIGNATURE:
        return None
    sha = str(rec.get("sequence_sha256") or "")
    if sha != obj_dir.name:
        return None
    return {
        "sequence_sha256": sha,
        "length": int(rec["length"]),
        "mean_ca_plddt": float(rec["mean_ca_plddt"]),
        "ptm": float(rec["ptm"]),
        "diffusion_sample_index": int(rec["diffusion_sample_index"]),
        "diffusion_sample_splits": rec.get("diffusion_sample_splits", [NUM_DIFFUSION_SAMPLES]),
        "evaluation_signature": EXPECTED_SIGNATURE,
        "status": "ok",
        "source_batch": "existing_ot_20261110",
    }


def _mode_prepare(args: argparse.Namespace) -> None:
    t0 = time.monotonic()
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    # 1. Load 47,778 main cohort sequences
    main_rows = _read_jsonl(args.main_cohort)
    main_by_sha: dict[str, str] = {}
    for row in main_rows:
        seq = str(row["sequence"])
        sha = se.sequence_digest(seq)
        main_by_sha[sha] = seq
    print(f"[prepare] main cohort unique sequences: {len(main_by_sha)}", flush=True)

    # 2. Load 252 pending long_cap sequences (optional)
    lc_by_sha: dict[str, str] = {}
    if args.long_cap_pending is not None and args.long_cap_pending.is_file():
        for row in _read_jsonl(args.long_cap_pending):
            seq = str(row["sequence"])
            sha = se.sequence_digest(seq)
            lc_by_sha[sha] = seq
        print(f"[prepare] long_cap pending unique sequences: {len(lc_by_sha)}", flush=True)

    # 3. Scan existing read-only GPFS objects directory in parallel
    obj_dirs = [p for p in args.existing_objects.iterdir() if p.is_dir()]
    print(f"[prepare] scanning {len(obj_dirs)} existing GPFS object dirs...", flush=True)
    existing_ok: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=32) as pool:
        for rec in pool.map(_scan_one_existing, obj_dirs):
            if rec is not None:
                existing_ok[rec["sequence_sha256"]] = rec

    main_existing = {sha: existing_ok[sha] for sha in main_by_sha if sha in existing_ok}
    existing_path = work / "existing_ok.jsonl"
    existing_path.write_text(
        "".join(json.dumps(main_existing[sha], sort_keys=True) + "\n" for sha in sorted(main_existing)),
        encoding="utf-8",
    )

    # 4. Build union of missing main + missing long_cap sequences (sorted longest-first)
    pending_map: dict[str, dict[str, Any]] = {}
    for sha, seq in main_by_sha.items():
        if sha not in existing_ok:
            pending_map[sha] = {
                "sequence_sha256": sha,
                "sequence": seq,
                "length": len(seq),
                "cohorts": ["main"],
            }
    for sha, seq in lc_by_sha.items():
        if sha in existing_ok:
            continue
        if sha in pending_map:
            pending_map[sha]["cohorts"].append("long_cap")
        else:
            pending_map[sha] = {
                "sequence_sha256": sha,
                "sequence": seq,
                "length": len(seq),
                "cohorts": ["long_cap"],
            }

    pending_sorted = sorted(
        pending_map.values(),
        key=lambda item: (-int(item["length"]), str(item["sequence_sha256"])),
    )
    pending_path = work / "pending_to_fold.jsonl"
    pending_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in pending_sorted),
        encoding="utf-8",
    )
    summary = {
        "main_total": len(main_by_sha),
        "main_already_folded_ok": len(main_existing),
        "main_missing": len(main_by_sha) - len(main_existing),
        "long_cap_requested_pending": len(lc_by_sha),
        "long_cap_already_in_ot": sum(1 for sha in lc_by_sha if sha in existing_ok),
        "union_pending_to_fold": len(pending_sorted),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    _atomic_write_json(work / "prepare_summary.json", summary)
    print(f"[prepare] DONE: {json.dumps(summary)}", flush=True)


def _concat_fold_outputs(parts: list[Any]) -> Any:
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Output

    first = parts[0]
    distogram = first.distogram_logits
    merged: dict[str, Any] = {
        "distogram_logits": None if distogram is None else distogram.detach().cpu(),
    }
    for name in _SAMPLE_OUTPUT_FIELDS:
        tensors = [getattr(part, name) for part in parts]
        if any(item is None for item in tensors):
            merged[name] = None
        else:
            merged[name] = torch.cat([item.detach().cpu() for item in tensors], dim=0)
    return EsmFold2Output(**merged)


def _fold_chunk(model: Any, sequence: str, n_samples: int, stream_seed: int, prepare_protein_features: Any, device: Any) -> Any:
    import torch

    torch.cuda.set_device(device)
    torch.manual_seed(stream_seed)
    torch.cuda.manual_seed_all(stream_seed)
    with torch.inference_mode():
        features = prepare_protein_features(sequence, device=device)
        return model.fold(**features, num_diffusion_samples=n_samples)


def _fold_replica_axis(model: Any, sequence: str, n_samples: int, seed: int, prepare_protein_features: Any, device: Any) -> tuple[Any, list[int]]:
    import torch

    # L >= 700 always requires [16, 16] on 143 GiB H200 (matching frozen long_cap_termination_20261110);
    # starting at chunks=2 uses the exact same seed+0 / seed+1 stream while skipping the wasted OOM pass.
    chunks = 2 if len(sequence) >= 700 else 1
    while True:
        splits = se.diffusion_sample_splits(n_samples, chunks)
        try:
            parts = [
                _fold_chunk(model, sequence, count, seed + idx, prepare_protein_features, device)
                for idx, count in enumerate(splits)
            ]
            return _concat_fold_outputs(parts), splits
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            if chunks >= n_samples:
                raise
            chunks = min(n_samples, chunks * 2)


def _mode_fold(args: argparse.Namespace) -> None:
    import torch
    from transformers import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import prepare_protein_features

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("ESMFold2 requires an assigned CUDA GPU")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(FOLD_SEED)
    torch.cuda.manual_seed_all(FOLD_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Verify model checkpoint matches exact pinned SHA256s on shard 0 (or always if fast).
    if args.verify_model_hashes:
        actual = se.checkpoint_file_digests(args.model)
        for fname, expected_sha in PINNED_MODEL_FILES_SHA256.items():
            if actual.get(fname) != expected_sha:
                raise RuntimeError(f"Model file digest mismatch for {fname}: {actual.get(fname)} != {expected_sha}")

    work = args.work_dir
    out_objs = work / "new_objects"
    out_objs.mkdir(parents=True, exist_ok=True)

    pending_all = _read_jsonl(work / "pending_to_fold.jsonl")
    assigned = [row for idx, row in enumerate(pending_all) if idx % args.num_shards == args.shard_index]

    todo: list[dict[str, Any]] = []
    for row in assigned:
        sha = str(row["sequence_sha256"])
        obj_file = out_objs / f"{sha}.json"
        if obj_file.is_file():
            try:
                prev = json.loads(obj_file.read_text(encoding="utf-8"))
                if prev.get("status") == "ok" and prev.get("evaluation_signature") == EXPECTED_SIGNATURE:
                    continue
            except Exception:
                pass
        todo.append(row)

    print(
        f"[fold shard {args.shard_index}/{args.num_shards} {args.device}] "
        f"assigned={len(assigned)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        return

    t_load = time.monotonic()
    model = EsmFold2Model.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16)
    model = model.eval().to(device)
    print(
        f"[fold shard {args.shard_index}/{args.num_shards}] model loaded in {time.monotonic() - t_load:.1f}s",
        flush=True,
    )

    t_start = time.monotonic()
    for ordinal, row in enumerate(todo, 1):
        sha = str(row["sequence_sha256"])
        seq = str(row["sequence"])
        t_seq = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        output = None
        try:
            seed = se.fold_seed(seq, base_seed=FOLD_SEED, rule=FOLD_SEED_RULE)
            output, splits = _fold_replica_axis(
                model, seq, NUM_DIFFUSION_SAMPLES, seed, prepare_protein_features, device
            )
            summary = se.summarize_esmfold2(output, sequence=seq)
            result = {
                "schema_version": se.SCHEMA_VERSION,
                "evaluation_signature": EXPECTED_SIGNATURE,
                "sequence_sha256": sha,
                "length": len(seq),
                "mean_ca_plddt": float(summary["mean_ca_plddt"]),
                "ptm": float(summary["ptm"]),
                "diffusion_sample_index": int(summary["diffusion_sample_index"]),
                "diffusion_sample_splits": splits,
                "status": "ok",
                "predictor": se.PREDICTOR_ID,
                "fold_seed": FOLD_SEED,
                "fold_seed_rule": FOLD_SEED_RULE,
                "source_batch": "completed_r6_20261124",
                "elapsed_seconds": round(time.monotonic() - t_seq, 4),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _atomic_write_json(out_objs / f"{sha}.json", result)
        finally:
            del output
            if ordinal % 16 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        log_every = 5 if len(seq) >= 700 else 20
        if ordinal % log_every == 0 or ordinal == len(todo):
            rate = ordinal / max(time.monotonic() - t_start, 1e-6)
            eta = (len(todo) - ordinal) / max(rate, 1e-6)
            print(
                f"[fold shard {args.shard_index}/{args.num_shards}] "
                f"{ordinal}/{len(todo)} L={len(seq)} pLDDT={result['mean_ca_plddt']:.2f} "
                f"pTM={result['ptm']:.4f} rate={rate:.2f}/s eta={eta/60:.1f}m",
                flush=True,
            )


def _mode_export(args: argparse.Namespace) -> None:
    work = args.work_dir
    out_objs = work / "new_objects"

    # 1. Load existing_ok.jsonl + all new_objects/*.json
    records: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(work / "existing_ok.jsonl"):
        records[str(rec["sequence_sha256"])] = rec
    new_count = 0
    for path in sorted(out_objs.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") == "ok" and rec.get("evaluation_signature") == EXPECTED_SIGNATURE:
            records[str(rec["sequence_sha256"])] = rec
            new_count += 1
    print(f"[export] loaded existing+new ok records: total={len(records)} (new={new_count})", flush=True)

    # 2. Export checkpoint_structures_68of68.tsv (47,778 rows)
    main_rows = _read_jsonl(args.main_cohort)
    arm_map_raw = json.loads(args.digest_arm_map.read_text(encoding="utf-8"))
    arm_by_sha = {k: sorted(v) for k, v in arm_map_raw.items()}

    missing_main = []
    out_main_rows = []
    for row in main_rows:
        sha = se.sequence_digest(str(row["sequence"]))
        rec = records.get(sha)
        if rec is None:
            missing_main.append(sha)
            continue
        arms = arm_by_sha.get(sha, [])
        if len(arms) != 1:
            raise RuntimeError(f"Digest {sha} maps to {arms} (expected exactly 1 arm)")
        shard = int(sha, 16) % NUM_SHARDS_68
        out_main_rows.append(
            (
                arms[0],
                sha,
                int(rec["length"]),
                rec["mean_ca_plddt"],
                rec["ptm"],
                shard,
                FOLD_SEED,
                FOLD_SEED_RULE,
                int(rec["diffusion_sample_index"]),
                EXPECTED_SIGNATURE,
            )
        )
    if missing_main:
        raise RuntimeError(f"Main cohort incomplete: {len(missing_main)} missing, e.g. {missing_main[:5]}")

    cols_main = [
        "arm", "sequence_sha256", "length_residues", "mean_ca_plddt_0_100", "ptm_0_1",
        "shard_index_of_68", "fold_seed", "fold_seed_rule", "diffusion_sample_index",
        "evaluation_signature",
    ]
    hdr_main = [
        "# per-structure fold measurements, campaign objective_tuned_20261109",
        "# fold snapshot 20261110074842_20277c6a196f + 20261124 full completion ; predictor ESMFold2 at 32 diffusion samples",
        "# complete 68/68 shards: all 47,778 unique sequences (0 sampling, 0 missing); status == ok only",
        "#",
        "\t".join(cols_main),
    ]
    main_tsv = work / "checkpoint_structures_68of68.tsv"
    with main_tsv.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(hdr_main) + "\n")
        for r in sorted(out_main_rows, key=lambda x: (x[0], x[1])):
            fh.write("\t".join(str(v) for v in r) + "\n")
    main_sha = hashlib.sha256(main_tsv.read_bytes()).hexdigest()
    print(f"[export] wrote {main_tsv} ({len(out_main_rows)} rows, sha256={main_sha})", flush=True)

    # 3. Export long_cap_structures_2000of2000.tsv (2,000 rows)
    if args.long_cap_all is not None and args.long_cap_existing_tsv is not None:
        lc_existing: dict[tuple[str, int], dict[str, str]] = {}
        with args.long_cap_existing_tsv.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                lc_existing[(row["arm"], int(row["attempt_index"]))] = row

        lc_all = _read_jsonl(args.long_cap_all)
        lc_missing = []
        lc_out_rows = []
        for row in lc_all:
            arm = str(row["arm"])
            arm_tag = "base" if arm == "protgpt3-10b" else "tuned"
            attempt_idx = int(row["attempt_index"])
            sha = str(row["sequence_sha256"])
            length = int(row["length"])
            stop_status = str(row["stop_status"])
            population = str(row["population"])
            prev = lc_existing.get((arm, attempt_idx))
            if prev is not None and prev.get("status") == "ok":
                lc_out_rows.append(
                    {
                        "arm_tag": arm_tag,
                        "arm": arm,
                        "attempt_index": attempt_idx,
                        "sequence_sha256": sha,
                        "length": length,
                        "stop_status": stop_status,
                        "population": population,
                        "status": "ok",
                        "plddt": prev["plddt"],
                        "ptm": prev["ptm"],
                        "diffusion_sample_index": prev["diffusion_sample_index"],
                        "evaluation_signature": prev["evaluation_signature"],
                        "source_batch": "existing_long_cap_20261110",
                    }
                )
            else:
                rec = records.get(sha)
                if rec is None:
                    lc_missing.append((arm, attempt_idx, sha))
                    continue
                lc_out_rows.append(
                    {
                        "arm_tag": arm_tag,
                        "arm": arm,
                        "attempt_index": attempt_idx,
                        "sequence_sha256": sha,
                        "length": int(rec["length"]),
                        "stop_status": stop_status,
                        "population": population,
                        "status": "ok",
                        "plddt": rec["mean_ca_plddt"],
                        "ptm": rec["ptm"],
                        "diffusion_sample_index": int(rec["diffusion_sample_index"]),
                        "evaluation_signature": EXPECTED_SIGNATURE,
                        "source_batch": "completed_r6_20261124",
                    }
                )
        if lc_missing:
            raise RuntimeError(f"long_cap incomplete: {len(lc_missing)} missing, e.g. {lc_missing[:5]}")

        lc_cols = [
            "arm_tag", "arm", "attempt_index", "sequence_sha256", "length",
            "stop_status", "population", "status", "plddt", "ptm",
            "diffusion_sample_index", "evaluation_signature", "source_batch",
        ]
        lc_tsv = work / "long_cap_structures_2000of2000.tsv"
        with lc_tsv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=lc_cols, delimiter="\t")
            writer.writeheader()
            for r in sorted(lc_out_rows, key=lambda x: (x["arm_tag"], int(x["attempt_index"]))):
                writer.writerow(r)
        lc_sha = hashlib.sha256(lc_tsv.read_bytes()).hexdigest()
        print(f"[export] wrote {lc_tsv} ({len(lc_out_rows)} rows, sha256={lc_sha})", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "fold", "export"), required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--existing-objects", type=Path, default=None)
    p.add_argument("--main-cohort", type=Path, default=None)
    p.add_argument("--digest-arm-map", type=Path, default=None)
    p.add_argument("--long-cap-pending", type=Path, default=None)
    p.add_argument("--long-cap-all", type=Path, default=None)
    p.add_argument("--long-cap-existing-tsv", type=Path, default=None)
    p.add_argument("--model", type=Path, default=Path("/gpfs/jiaotongdamoxing/zhk_zip/models/ESMFold2-hf"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--verify-model-hashes", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "prepare":
        _mode_prepare(args)
    elif args.mode == "fold":
        _mode_fold(args)
    elif args.mode == "export":
        _mode_export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
