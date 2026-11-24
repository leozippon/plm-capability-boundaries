#!/usr/bin/env python3
"""GPU driver for Experiment 4.1: ProGen3 unconditional pool generation + ESMFold2 folding.

Two clean subcommands so Phase A (ProGen3 generation under ``ct-20260905``) and
Phase B (ESMFold2 folding under ``esmfold2`` runtime with ``transformers>=5.16``)
run across up to 6 H200 GPUs in parallel:

- ``generate-progen3``: loads a ProGen3 checkpoint through the validated strict
  loader (``load_general_plm`` / ``load_progen3``), verifies cached-vs-full
  logit parity, and generates ``N=800`` unconditional sequences
  (``temperature=1.0``, ``top_p=0.95``, ``max_new_tokens=400``) via
  ``ProGen3Generator`` with KV-cache.
- ``fold-esmfold2``: folds the first ``N=400`` eligible AA20 sequences
  (``20 <= length <= 400``) from an input JSONL using ``biohub/ESMFold2-hf`` in
  ``bfloat16`` under the exact ``evaluate_r5_structures.py`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.generation import structure_evidence as se

AA20_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]*")
AA20_SET = frozenset(AA20)
SEED_RULE = "sequence_digest"
DEFAULT_FOLD_BASE_SEED = 20261120


def _seq_sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _arm_slug(arm: str) -> str:
    return arm.replace("-", "_")


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


@torch.no_grad()
def _check_kv_cache_parity(pg: Any) -> float:
    batch = pg.batch(["MALWMRLLPLLALLALWGP"])
    inputs = {k: batch[k][:, :10] for k in ("input_ids", "position_ids", "sequence_ids")}
    full = pg.model(**inputs, use_cache=False, return_dict=True).logits[:, -1, :].float()
    prefix = {k: v[:, :-1] for k, v in inputs.items()}
    cached = pg.model(**prefix, use_cache=True, return_dict=True)
    final = {k: v[:, -1:] for k, v in inputs.items()}
    inc = (
        pg.model(**final, past_key_values=cached.past_key_values, use_cache=True, return_dict=True)
        .logits[:, -1, :]
        .float()
    )
    diff = float((full - inc).abs().max().item())
    if not torch.isfinite(inc).all() or diff > 0.125:
        raise RuntimeError(f"ProGen3 KV-cache parity check failed: max_abs_diff={diff:.6g}")
    return diff


def cmd_generate_progen3(args: argparse.Namespace) -> None:
    from src.capability.general_plm import load_general_plm
    from src.capability.generation.progen3_generation import install_generation_compatibility

    t0 = time.monotonic()
    arm = str(args.arm)
    slug = _arm_slug(arm)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    adapter = load_general_plm(
        arm,
        device=args.device,
        dtype="bfloat16",
        model_root=args.model_root,
    )
    from progen3.generator import ProGen3Generator  # noqa: PLC0415

    pg = adapter.handle
    install_generation_compatibility(pg.model)
    cache_diff = _check_kv_cache_parity(pg)
    print(f"[{arm}] loaded on {args.device}; kv_cache_max_diff={cache_diff:.5f}", flush=True)

    max_new = int(args.max_new_tokens)
    batch_size = max(1, int(args.batch_size))
    generator = ProGen3Generator(
        pg.model,
        max_batch_tokens=batch_size * (2 + max_new + 2),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
    # Pure nucleus top_p sampling matching generate_causal_free (disable top_k cutoff).
    generator.default_gen_config.top_k = 0
    generator.default_gen_config.repetition_penalty = 1.0
    generator.default_gen_config.bos_token_id = pg.config.bos_token_id
    generator.default_gen_config.output_logits = False

    n_draws = int(args.n_draws)
    base_seed = int(args.seed)
    rows: list[dict[str, Any]] = []

    for start in range(0, n_draws, batch_size):
        end = min(start + batch_size, n_draws)
        b_idx = start // batch_size
        b_seed = base_seed + b_idx
        torch.manual_seed(b_seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(b_seed)
        samples = list(generator.generate("1", end - start, 0, max_new))
        if len(samples) != end - start:
            raise RuntimeError(f"[{arm}] expected {end - start} samples, got {len(samples)}")
        for off, item in enumerate(samples):
            draw_idx = start + off
            raw = str(item.generation or "")
            compiled = item.sequence
            match = AA20_RE.match(raw)
            prefix = match.group(0) if match is not None else ""
            native_complete = compiled is not None
            seq = str(compiled) if native_complete else prefix
            seq = seq[:max_new]
            if native_complete:
                stop_reason = "native_terminal_eos"
            elif raw == prefix:
                stop_reason = "budget_censored_residue_continuation"
            else:
                stop_reason = "native_format_not_compilable"
            rows.append(
                {
                    "seq_id": f"{slug}_{draw_idx:04d}",
                    "arm": arm,
                    "draw_index": draw_idx,
                    "batch_seed": b_seed,
                    "sequence": seq,
                    "sequence_sha256": _seq_sha(seq) if seq else "",
                    "length": len(seq),
                    "terminated_eos": bool(native_complete),
                    "official_compilation_valid": bool(native_complete),
                    "source_stop_reason": stop_reason,
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "max_new_tokens": max_new,
                }
            )
        if end % 160 == 0 or end == n_draws:
            print(
                f"[{arm}] generated {end}/{n_draws} ({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    _atomic_write_jsonl(args.out_jsonl, rows)
    eligible = [r for r in rows if 20 <= r["length"] <= max_new and set(r["sequence"]) <= AA20_SET]
    summary = {
        "arm": arm,
        "out_jsonl": str(args.out_jsonl),
        "n_draws": len(rows),
        "n_eligible_20_400": len(eligible),
        "n_terminated_eos": sum(1 for r in rows if r["terminated_eos"]),
        "mean_length": float(np.mean([r["length"] for r in rows])) if rows else 0.0,
        "kv_cache_max_diff": cache_diff,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    print(json.dumps(summary, indent=2), flush=True)


def cmd_fold_esmfold2(args: argparse.Namespace) -> None:
    from transformers import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import prepare_protein_features

    t0 = time.monotonic()
    in_path: Path = args.input_jsonl
    if not in_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    arm = str(args.arm)
    min_len = int(args.min_length)
    max_len = int(args.max_length)
    max_seqs = int(args.max_sequences)

    selected: list[dict[str, Any]] = []
    skipped = 0
    for line_no, line in enumerate(in_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        seq = str(raw.get("sequence") or "").strip().upper()
        if se.eligibility(seq, min_length=min_len, max_length=max_len) is not None:
            skipped += 1
            continue
        seq_id = str(
            raw.get("seq_id")
            or raw.get("record_id")
            or raw.get("id")
            or f"{_arm_slug(arm)}_{line_no - 1:04d}"
        )
        selected.append(
            {
                "seq_id": seq_id,
                "arm": str(raw.get("arm") or arm),
                "sequence": seq,
                "sequence_sha256": _seq_sha(seq),
                "length": len(seq),
            }
        )
        if max_seqs > 0 and len(selected) >= max_seqs:
            break

    if not selected:
        raise RuntimeError(f"[{arm}] no eligible sequences in {in_path} (skipped={skipped})")

    unique_seqs: dict[str, str] = {}
    for row in selected:
        unique_seqs[row["sequence_sha256"]] = row["sequence"]

    esmfold2_dir = Path(args.esmfold2_dir).resolve()
    se.require_esmfold2_checkpoint(esmfold2_dir)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    t_load = time.monotonic()
    model = (
        EsmFold2Model.from_pretrained(esmfold2_dir, local_files_only=True, dtype=torch.bfloat16)
        .eval()
        .to(dev)
    )
    load_sec = time.monotonic() - t_load
    print(
        f"[{arm}] ESMFold2 loaded in {load_sec:.1f}s; folding {len(unique_seqs)} unique "
        f"(for {len(selected)} records, skipped={skipped}) on {args.device}",
        flush=True,
    )

    fold_cache: dict[str, dict[str, Any]] = {}
    items = sorted(unique_seqs.items(), key=lambda kv: (len(kv[1]), kv[0]))
    base_seed = int(args.base_seed)
    diff_samples = int(args.diffusion_samples)

    for idx, (sha, seq) in enumerate(items, 1):
        seed = se.fold_seed(seq, base_seed=base_seed, rule=SEED_RULE)
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        ts = time.monotonic()
        with torch.inference_mode():
            features = prepare_protein_features(seq, device=dev)
            output = model.fold(**features, num_diffusion_samples=diff_samples)
        summ = se.summarize_esmfold2(output, sequence=seq)
        plddt = float(summ["mean_ca_plddt"])
        fold_cache[sha] = {
            "plddt": plddt,
            "mean_ca_plddt": plddt,
            "fraction_ca_plddt_ge70": float(summ["fraction_ca_plddt_ge70"]),
            "ptm": float(summ["ptm"]),
            "mean_pae_angstrom": float(summ["mean_pae_angstrom"]),
            "diffusion_sample_index": int(summ["diffusion_sample_index"]),
            "fold_seconds": round(time.monotonic() - ts, 4),
        }
        del output, features
        if idx % 50 == 0 or idx == len(items):
            print(
                f"[{arm}] folded {idx}/{len(items)} ({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    out_rows: list[dict[str, Any]] = []
    for row in selected:
        merged = {**row, **fold_cache[row["sequence_sha256"]]}
        out_rows.append(merged)

    _atomic_write_jsonl(args.out_jsonl, out_rows)
    plddts = [r["plddt"] for r in out_rows]
    ptms = [r["ptm"] for r in out_rows]
    summary = {
        "arm": arm,
        "out_jsonl": str(args.out_jsonl),
        "n_folded": len(out_rows),
        "n_unique": len(unique_seqs),
        "mean_plddt": float(np.mean(plddts)),
        "fraction_plddt_ge70": float(np.mean([x >= 70.0 for x in plddts])),
        "mean_ptm": float(np.mean(ptms)),
        "mean_length": float(np.mean([r["length"] for r in out_rows])),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate-progen3", help="Generate unconditional ProGen3 pool")
    p_gen.add_argument("--arm", required=True)
    p_gen.add_argument("--out-jsonl", type=Path, required=True)
    p_gen.add_argument("--device", default="cuda:0")
    p_gen.add_argument("--model-root", type=Path, required=True)
    p_gen.add_argument("--n-draws", type=int, default=800)
    p_gen.add_argument("--batch-size", type=int, default=32)
    p_gen.add_argument("--max-new-tokens", type=int, default=400)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-p", type=float, default=0.95)
    p_gen.add_argument("--seed", type=int, default=20261124)

    p_fold = sub.add_parser("fold-esmfold2", help="Fold JSONL pool with biohub/ESMFold2-hf")
    p_fold.add_argument("--arm", required=True)
    p_fold.add_argument("--input-jsonl", type=Path, required=True)
    p_fold.add_argument("--out-jsonl", type=Path, required=True)
    p_fold.add_argument("--esmfold2-dir", type=Path, required=True)
    p_fold.add_argument("--device", default="cuda:0")
    p_fold.add_argument("--min-length", type=int, default=20)
    p_fold.add_argument("--max-length", type=int, default=400)
    p_fold.add_argument("--max-sequences", type=int, default=400)
    p_fold.add_argument("--diffusion-samples", type=int, default=1)
    p_fold.add_argument("--base-seed", type=int, default=DEFAULT_FOLD_BASE_SEED)

    args = parser.parse_args()
    if args.cmd == "generate-progen3":
        cmd_generate_progen3(args)
    elif args.cmd == "fold-esmfold2":
        cmd_fold_esmfold2(args)
    else:
        raise ValueError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    main()
