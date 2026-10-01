#!/usr/bin/env python3
"""Fold a frozen generation/control JSONL shard with offline ESMFold2 assets."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation import structure_evidence as se  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402

_SAMPLE_OUTPUT_FIELDS = (
    "sample_atom_coords", "plddt_logits", "plddt", "plddt_per_atom", "plddt_ca",
    "complex_plddt", "complex_iplddt", "pae_logits", "pae", "pde_logits", "pde",
    "resolved_logits", "ptm", "iptm", "pair_chains_iptm",
)


def _concat_fold_outputs(parts):
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Output

    first = parts[0]
    distogram = first.distogram_logits
    merged = {
        "distogram_logits": None if distogram is None else distogram.detach().cpu(),
    }
    for name in _SAMPLE_OUTPUT_FIELDS:
        tensors = [getattr(part, name) for part in parts]
        if any(item is None for item in tensors):
            merged[name] = None
        else:
            merged[name] = torch.cat([item.detach().cpu() for item in tensors], dim=0)
    return EsmFold2Output(**merged)


def _fold_chunk(model, sequence, n_samples, stream_seed, fold_kwargs, prepare_protein_features, device):
    import torch

    torch.cuda.set_device(device)
    torch.manual_seed(stream_seed)
    torch.cuda.manual_seed_all(stream_seed)
    kwargs = dict(fold_kwargs)
    kwargs["num_diffusion_samples"] = n_samples
    with torch.inference_mode():
        features = prepare_protein_features(sequence, device=device)
        return model.fold(**features, **kwargs)


def _fold_replica_axis(model, sequence, n_samples, seed, fold_kwargs, prepare_protein_features, device):
    """Fold the replica axis on the assigned card, in more chunks after OOM.

    Chunk ``i`` draws from stream seed ``seed + i``; the sample count and the
    highest-pTM selection rule do not change with the split. Chunks stay on one
    device and run one at a time, because seeding is process-global: folding two
    chunks concurrently would leave the realized stream undefined and break the
    declared per-chunk seed.
    """

    import torch

    chunks = 1
    while True:
        splits = se.diffusion_sample_splits(n_samples, chunks)
        try:
            parts = [
                _fold_chunk(model, sequence, count, seed + index,
                            fold_kwargs, prepare_protein_features, device)
                for index, count in enumerate(splits)
            ]
            return _concat_fold_outputs(parts), splits
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            if chunks >= n_samples:
                raise
            chunks = min(n_samples, chunks * 2)


def _has_distribution(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


FOLDING_RUNTIME_ERROR = (
    "ESMFold2 needs an isolated folding runtime that imports transformers.EsmFold2Model "
    "(contributed 2026-08-19). The validated ct environment is Transformers 4.57.3 and "
    "cannot load this predictor. Do not fall back to facebook/esmfold_v1."
)


def _folding_imports():
    try:
        from transformers import EsmFold2Model
        from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features
    except ImportError as error:
        raise RuntimeError(FOLDING_RUNTIME_ERROR) from error
    return EsmFold2Model, output_to_pdb, prepare_protein_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Staged biohub/ESMFold2-hf checkpoint; required only to fold")
    parser.add_argument("--device", default="cuda:0", help="Logical device within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=None, help="Override config.chunk_size; default keeps the released 64")
    parser.add_argument("--num-loops", type=int, default=None, help="None keeps the staged checkpoint num_loops")
    parser.add_argument("--num-sampling-steps", type=int, default=None, help="None keeps config.structure_head.inference_num_steps")
    parser.add_argument("--num-diffusion-samples", type=int, default=None, help="None keeps config.structure_head.num_diffusion_samples")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--index-only", action="store_true", help="Combine all result objects into index.jsonl without GPU/model loading")
    parser.add_argument("--verify", action="store_true", help="Replay an existing tree against its own receipts; writes nothing, needs no GPU")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.verify and args.index_only:
        parser.error("--verify writes nothing; run it after --index-only, not with it")
    if args.model is None and not (args.verify or args.index_only):
        parser.error("--model is required to fold")
    if not 1 <= args.min_length <= args.max_length <= 1024:
        parser.error("invalid supported lengths")
    if args.num_loops is not None and args.num_loops < 1:
        parser.error("num-loops must be positive when set")
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error("chunk-size must be positive when set")
    if args.num_sampling_steps is not None and args.num_sampling_steps < 1:
        parser.error("num-sampling-steps must be positive when set")
    if args.num_diffusion_samples is not None and args.num_diffusion_samples < 1:
        parser.error("num-diffusion-samples must be positive when set")

    rows = se.load_cohort(args.cohort)
    if args.verify:
        summary = se.verify_evidence(
            rows, args.out, shard_index=args.shard_index, num_shards=args.num_shards,
        )
        print(json.dumps(summary), flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    # A resumed attempt must not leave its previous completion marker visible
    # while new work is running or a changed configuration is being rejected.
    (args.out / "structure_evidence.json").unlink(missing_ok=True)
    if args.index_only:
        signature = se.tree_signature(args.out)
        cohorts = {json.loads(path.read_text())["cohort_sha256"] for path in args.out.glob("worker-*.json")}
        if cohorts != {sha256_file(args.cohort)}:
            raise ValueError("worker receipts disagree with the cohort digest")
        summary = se.write_index(rows, args.out, signature, shard_index=0, num_shards=1, filename="index.jsonl")
        print(json.dumps(summary), flush=True)
        if summary["status_counts"].get("pending", 0):
            raise SystemExit("incomplete evaluation: pending rows remain")
        write_json(args.out / "structure_evidence.json", summary)
        return

    EsmFold2Model, output_to_pdb, prepare_protein_features = _folding_imports()
    import torch
    import transformers

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("ESMFold2 runtime requires an assigned CUDA GPU")
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 8:
        raise RuntimeError("GPU requires Ampere or newer")
    config = se.require_esmfold2_checkpoint(args.model)
    if args.chunk_size is not None:
        config = dict(config)
        config["chunk_size"] = args.chunk_size
    model_files = se.checkpoint_file_digests(args.model)
    structure_head = config.get("structure_head") or {}
    contract = {
        "schema_version": se.SCHEMA_VERSION,
        "model_id": se.PREDICTOR_ID,
        "predictor_kind": se.PREDICTOR_KIND,
        "msa": None,
        "model_files_sha256": model_files,
        "code_sha256": {"module": sha256_file(Path(se.__file__)), "runner": sha256_file(Path(__file__))},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy")
            if _has_distribution(name)
        },
        "python": platform.python_version(),
        "min_length": args.min_length,
        "max_length": args.max_length,
        "chunk_size": args.chunk_size if args.chunk_size is not None else config.get("chunk_size"),
        "num_loops": args.num_loops if args.num_loops is not None else config.get("num_loops"),
        "published_default_num_loops": config.get("num_loops"),
        "num_sampling_steps": args.num_sampling_steps if args.num_sampling_steps is not None else structure_head.get("inference_num_steps"),
        "published_default_num_sampling_steps": structure_head.get("inference_num_steps"),
        "num_diffusion_samples": args.num_diffusion_samples if args.num_diffusion_samples is not None else structure_head.get("num_diffusion_samples"),
        "published_default_num_diffusion_samples": structure_head.get("num_diffusion_samples"),
        "precision": "bfloat16; no autocast; TF32 disabled; single sequence; no MSA",
        "batch_size": 1,
        "ordering": "longest sequence first, SHA256 tie-break; exact sequence hash sharding",
        "seed": args.seed,
        "never_truncate": True,
        "sample_selection": "highest pTM, matching infer_protein_as_pdb",
        "plddt_conversion": "ESMFold2 plddt_ca raw [0,1] multiplied by 100 for metrics; PDB B factors remain the library 0-1 scale",
        "raw_outputs": "selected-sample CA pLDDT, full PAE, pTM, diffusion sample index; latent embeddings/logits omitted",
        "interpretation": se.INTERPRETATION,
        "transformers": transformers.__version__,
    }
    signature = se.contract_signature(contract)
    assigned = {se.sequence_digest(row["sequence"]): row["sequence"] for row in rows if se.shard_for(row["sequence"], args.num_shards) == args.shard_index}
    manifest_path = args.out / f"worker-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    manifest = {
        "evaluation_signature": signature, "contract": contract,
        "cohort_sha256": sha256_file(args.cohort),
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "unique_sequences_assigned": len(assigned),
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory,
        "gpu_free_memory_before_bytes": torch.cuda.mem_get_info(device)[0],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["evaluation_signature"] != signature or previous["cohort_sha256"] != manifest["cohort_sha256"]:
            raise ValueError("refusing to resume changed evaluation/cohort into same output")
    write_json(manifest_path, manifest)
    pending = []
    for sha, sequence in sorted(assigned.items(), key=lambda item: (-len(item[1]), item[0])):
        directory = args.out / "objects" / sha
        existing = se.load_result(directory, signature, sha)
        if existing is not None and not (args.retry_failed and existing["status"] == "failed"):
            continue
        reason = se.eligibility(sequence, min_length=args.min_length, max_length=args.max_length)
        if reason:
            write_json(directory / "result.json", {
                "schema_version": se.SCHEMA_VERSION, "evaluation_signature": signature,
                "sequence_sha256": sha, "length": len(sequence),
                "status": "not_evaluable", "reason": reason, "evaluator_truncated": False,
            })
        else:
            pending.append((sha, sequence))
    se.write_index(rows, args.out, signature, shard_index=args.shard_index, num_shards=args.num_shards)
    print(json.dumps({"shard": args.shard_index, "pending_unique_sequences": len(pending), "gpu": properties.name}), flush=True)
    if pending:
        n_samples = (
            args.num_diffusion_samples
            if args.num_diffusion_samples is not None
            else structure_head.get("num_diffusion_samples")
        )
        if not n_samples:
            raise RuntimeError("structure_head.num_diffusion_samples is missing")
        fold_kwargs = {}
        if args.num_loops is not None:
            fold_kwargs["num_loops"] = args.num_loops
        if args.num_sampling_steps is not None:
            fold_kwargs["num_sampling_steps"] = args.num_sampling_steps
        # Sequence-sharded queue cells share a host and each cell owns one card,
        # so this process folds on --device only.
        model_started = time.monotonic()
        model = EsmFold2Model.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
        )
        if args.chunk_size is not None:
            model.config.chunk_size = args.chunk_size
        model = model.eval().to(device)
        print(json.dumps({
            "model_loaded_seconds": time.monotonic() - model_started,
            "device": str(device),
            "diffusion_samples": n_samples,
        }), flush=True)
        for ordinal, (sha, sequence) in enumerate(pending, 1):
            directory = args.out / "objects" / sha
            started = time.monotonic()
            result = {
                "schema_version": se.SCHEMA_VERSION, "evaluation_signature": signature,
                "sequence_sha256": sha, "length": len(sequence), "evaluator_truncated": False,
                "predictor": se.PREDICTOR_ID,
            }
            torch.cuda.reset_peak_memory_stats(device)
            output = None
            features = None
            try:
                output, splits = _fold_replica_axis(
                    model, sequence, n_samples, args.seed, fold_kwargs,
                    prepare_protein_features, device,
                )
                features = prepare_protein_features(sequence, device="cpu")
                summary = se.summarize_esmfold2(output, sequence=sequence)
                arrays = se.esmfold2_arrays(output, sequence=sequence, sample=summary["diffusion_sample_index"])
                pdb = output_to_pdb(output, features, sample_idx=summary["diffusion_sample_index"])
                result.update(summary)
                result["diffusion_sample_splits"] = splits
                result["files_sha256"] = se.save_prediction(directory, arrays, pdb)
                result["status"] = "ok"
            except torch.cuda.OutOfMemoryError as error:
                result.update(status="failed", reason="cuda_out_of_memory", error=str(error)[:1500])
            except ValueError as error:
                result.update(status="failed", reason="invalid_prediction", error=str(error)[:1500])
            except Exception as error:
                result.update(status="failed", reason="unexpected_runtime_error", error=f"{type(error).__name__}: {error}"[:1500])
                write_json(directory / "result.json", result)
                raise
            finally:
                del output
                del features
                gc.collect()
                torch.cuda.empty_cache()
            result["elapsed_seconds"] = time.monotonic() - started
            result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            write_json(directory / "result.json", result)
            progress = {key: result[key] for key in (
                "sequence_sha256", "length", "status", "reason", "elapsed_seconds",
                "peak_gpu_memory_bytes", "mean_ca_plddt", "ptm", "diffusion_sample_splits",
            ) if key in result}
            print(json.dumps({"ordinal": ordinal, "total": len(pending), **progress}), flush=True)
    summary = se.write_index(rows, args.out, signature, shard_index=args.shard_index, num_shards=args.num_shards)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = summary
    write_json(manifest_path, manifest)
    write_json(args.out / "structure_evidence.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
