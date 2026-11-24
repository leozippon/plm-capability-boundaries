#!/usr/bin/env python3
"""Score native self-likelihoods across all 30 generative checkpoints (33 pools) for Experiment 4.3.

Supports all four native scoring routes in this repository:
1. ``readout_stage46``: 18 PREREGISTERED_33_ARMS causal/joint checkpoints + ZymCTRL
   via ``src.capability.readouts.readout_extraction`` + ``scripts.capability.stages.context_homologue._target_nll``
   (singleton scoring for ProGen3 and every arm where needed, verified against E17 ``protgpt2`` to <1e-5).
2. ``general_plm_causal``: 7 GP causal/hybrid checkpoints (``progen3-{219m,339m,762m,1b}``,
   ``protgpt3-{112m,10b}``, ``dayhoff-170m`` with ``output_router_logits=False``).
3. ``general_plm_masked``: ``dplm-{150m,650m,3b}`` exact leave-one-out masked PLL
   (batched across masked positions per sequence, ``mask_batch_size=256``).
4. ``esm3_masked``: ``esm3-1.4b`` sequence-only exact leave-one-out masked PLL
   (``load_esm3_offline(..., eager_load_aux=False)``, ``mask_batch_size=256``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGE46_ARMS: frozenset[str] = frozenset(
    {
        "progen2-small",
        "progen2-base",
        "progen2-medium",
        "progen2-large",
        "progen2-xlarge",
        "progen3-112m",
        "progen3-3b",
        "protgpt2",
        "protgpt3-1.3b",
        "proteinglm-7b-clm",
        "rita-xl",
        "galactica-125m",
        "galactica-1.3b",
        "galactica-6.7b",
        "galactica-30b",
        "instructprotein",
        "prollama-stage-1",
        "prollama",
        "zymctrl",
    }
)

GP_CAUSAL_ARMS: frozenset[str] = frozenset(
    {
        "progen3-219m",
        "progen3-339m",
        "progen3-762m",
        "progen3-1b",
        "protgpt3-112m",
        "protgpt3-10b",
        "dayhoff-170m",
    }
)

GP_MASKED_ARMS: frozenset[str] = frozenset(
    {
        "dplm-150m",
        "dplm-650m",
        "dplm-3b",
    }
)

ALL_KNOWN_ARMS: tuple[str, ...] = tuple(
    sorted(STAGE46_ARMS | GP_CAUSAL_ARMS | GP_MASKED_ARMS | {"esm3-1.4b"}, key=len, reverse=True)
)


def _seq_sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _load_stage46_module() -> Any:
    path = ROOT / "scripts/capability/stages/context_homologue.py"
    spec = importlib.util.spec_from_file_location("exp4_3_stage46", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stage46 module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_arm_from_pool(pool_id: str, rows: list[dict[str, Any]]) -> str:
    for row in rows:
        cand = str(row.get("arm_id") or row.get("arm") or row.get("model") or "").strip().lower().replace("_", "-")
        if cand in ALL_KNOWN_ARMS:
            return cand
    norm = pool_id.lower().replace("_", "-")
    for arm in ALL_KNOWN_ARMS:
        if norm == arm or norm.startswith(arm + "-"):
            return arm
    raise ValueError(f"Cannot resolve canonical arm from pool_id={pool_id!r}")


def _route_for_arm(arm: str) -> tuple[str, str]:
    if arm == "esm3-1.4b":
        return "esm3_masked", "float32"
    if arm in GP_MASKED_ARMS:
        return "general_plm_masked", "float32"
    if arm in GP_CAUSAL_ARMS:
        dtype = "bfloat16" if arm.startswith("progen3-") else "float32"
        return "general_plm_causal", dtype
    if arm in STAGE46_ARMS:
        if arm.startswith("progen3-"):
            return "readout_stage46", "bfloat16"
        if arm == "protgpt2":
            return "readout_stage46", "bfloat16"
        return "readout_stage46", "float32"
    raise ValueError(f"Unsupported arm for Experiment 4.3 scoring: {arm!r}")


def _score_protgpt2_batch(
    arm_obj: Any, sequences: list[str]
) -> list[tuple[float, int]]:
    """Exact E17 score_generated_likelihood.py batched protgpt2 scoring (bfloat16)."""
    from src.capability.core import scoring
    from src.capability.core.arms import Cohort, rendering_marker_ids, tokenize_batch

    cohort = Cohort(
        name="exp4_3_protgpt2",
        kind="protein",
        records=sequences,
        min_symbols=min(len(s) for s in sequences),
        max_symbols=max(len(s) for s in sequences),
    )
    rendered = cohort.input_strings(arm_obj)
    ids_t, mask_t = tokenize_batch(arm_obj, rendered, max_len=1024)
    target_mask = scoring.sequence_target_mask(
        ids_t,
        mask_t,
        rule=scoring.target_rule(arm_obj.spec.input_format, ec_conditioning="native"),
        start_token_id=None,
        end_token_id=None,
        marker_token_ids=rendering_marker_ids(arm_obj),
    )
    with torch.inference_mode():
        logits = (
            arm_obj.model(
                input_ids=ids_t.to(arm_obj.device),
                attention_mask=mask_t.to(arm_obj.device),
            )
            .logits.float()
            .cpu()
        )
    entries = scoring.per_sequence_scores(logits, logits, ids_t, target_mask)
    return [(float(e["clean_nll_sum"]), int(e["token_count"])) for e in entries]


def _score_single_stage46(
    arm_obj: Any,
    stage46: Any,
    sequence: str,
    *,
    class_key: str | None = None,
) -> tuple[float, int]:
    from src.capability.readouts.readout_extraction import (
        bind_readout_ec,
        forward_readout_rows,
        pack_sequence,
    )

    if arm_obj.name == "protgpt2":
        return _score_protgpt2_batch(arm_obj, [sequence])[0]

    if arm_obj.name == "zymctrl":
        if not class_key:
            raise ValueError("zymctrl requires class_key (4-field EC) on every row")
        bind_readout_ec(arm_obj, str(class_key))
    ids, (start, end), _ = pack_sequence(arm_obj, sequence)
    logits, scored_ids = forward_readout_rows(arm_obj, [ids], stage46)
    res = stage46._target_nll(logits[:1], scored_ids[:1], start, end)
    nll_sum = float(res["nll_sum"])
    scored_tokens = int(res["scored_tokens"])
    if not math.isfinite(nll_sum) or scored_tokens < 1:
        raise FloatingPointError(f"{arm_obj.name}: invalid NLL {nll_sum} / tokens {scored_tokens}")
    return nll_sum, scored_tokens


@torch.no_grad()
def _score_single_gp_causal(adapter: Any, sequence: str) -> tuple[float, int]:
    from src.capability.general_plm.adapters import ProGen3CausalAdapter

    if isinstance(adapter, ProGen3CausalAdapter):
        pv = adapter.exact_pll(sequence)
        return float(pv.nll_sum), int(len(pv.terms))

    ids, (start, end) = adapter._pack_item(sequence)
    tensor = torch.as_tensor([list(ids)], dtype=torch.long, device=adapter.device)
    kwargs: dict[str, Any] = {"input_ids": tensor, "use_cache": False}
    sig = inspect.signature(adapter.model.forward)
    if "logits_to_keep" in sig.parameters:
        kwargs["logits_to_keep"] = 0
    elif "num_logits_to_keep" in sig.parameters:
        kwargs["num_logits_to_keep"] = 0
    if "output_router_logits" in sig.parameters:
        kwargs["output_router_logits"] = False
    with adapter._precision_ctx():
        out = adapter.model(**kwargs)
        logits = out.logits.float()
        shift_logits = logits[0, start - 1 : end - 1]
        targets = tensor[0, start:end]
        terms = F.cross_entropy(shift_logits, targets, reduction="none")
    if not torch.isfinite(terms).all():
        raise FloatingPointError(f"{adapter.arm}: non-finite causal NLL term")
    return float(terms.sum()), int(terms.numel())


@torch.no_grad()
def _score_single_gp_masked(
    adapter: Any, sequence: str, *, mask_batch_size: int
) -> tuple[float, int]:
    ids, offset = adapter._encode_sequence(sequence)
    length = len(sequence)
    dev = adapter.device
    base = torch.as_tensor([ids], dtype=torch.long, device=dev)
    mask_id = int(adapter.mask_id)
    bs = max(1, int(mask_batch_size))
    terms = torch.empty(length, dtype=torch.float32, device=dev)

    with adapter._precision_ctx():
        for start in range(0, length, bs):
            stop = min(start + bs, length)
            p_idx = torch.arange(start, stop, device=dev, dtype=torch.long)
            batch = base.repeat(stop - start, 1)
            row_ar = torch.arange(stop - start, device=dev, dtype=torch.long)
            tok_col = p_idx + int(offset)
            batch[row_ar, tok_col] = mask_id
            logits = adapter._forward_logits(batch)
            logprobs = F.log_softmax(logits[row_ar, tok_col], dim=-1)
            targets = base[0, tok_col]
            terms[p_idx] = -logprobs[row_ar, targets]
            del logits, logprobs

    if not torch.isfinite(terms).all():
        raise FloatingPointError(f"{adapter.arm}: non-finite masked PLL term")
    return float(terms.sum()), length


@torch.no_grad()
def _score_single_esm3_masked(
    handle: Any, sequence: str, *, mask_batch_size: int
) -> tuple[float, int]:
    from src.capability.esm3_ext.esm3_adapter import forward_tracks, tokenize_sequence_ids

    length = len(sequence)
    dev = handle.device
    base = tokenize_sequence_ids(handle, sequence).unsqueeze(0)
    mask_id = int(handle.mask_id)
    bs = max(1, int(mask_batch_size))
    terms = torch.empty(length, dtype=torch.float32, device=dev)

    for start in range(0, length, bs):
        stop = min(start + bs, length)
        p_idx = torch.arange(start, stop, device=dev, dtype=torch.long)
        batch = base.repeat(stop - start, 1)
        row_ar = torch.arange(stop - start, device=dev, dtype=torch.long)
        tok_col = p_idx + 1
        batch[row_ar, tok_col] = mask_id
        logits = forward_tracks(handle, sequence_tokens=batch).sequence_logits.float()
        logprobs = F.log_softmax(logits[row_ar, tok_col], dim=-1)
        targets = base[0, tok_col]
        terms[p_idx] = -logprobs[row_ar, targets]
        del logits, logprobs

    if not torch.isfinite(terms).all():
        raise FloatingPointError("esm3-1.4b: non-finite masked PLL term")
    return float(terms.sum()), length


def _verify_protgpt2_e17(
    *,
    parity_jsonl: Path,
    device: str,
) -> dict[str, Any]:
    from src.capability.readouts.readout_extraction import load_readout_arm

    ref_rows = [
        json.loads(line)
        for line in parity_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage46 = _load_stage46_module()
    arm_obj = load_readout_arm("protgpt2", stage46, device=device, dtype="bfloat16")
    seqs = [str(ref["sequence"]).strip().upper() for ref in ref_rows]
    scored_batch = _score_protgpt2_batch(arm_obj, seqs)
    diffs: list[float] = []
    for ref, (nll_sum, scored_tokens) in zip(ref_rows, scored_batch, strict=True):
        rid = str(ref.get("seq_id") or ref.get("id"))
        mean_tok = nll_sum / scored_tokens
        exp_tok = float(ref.get("expected_mean_nll_per_token_nats", ref.get("mean_nll_per_token_nats")))
        exp_cnt = int(ref.get("expected_scored_tokens", ref.get("scored_tokens")))
        diff = abs(mean_tok - exp_tok)
        diffs.append(diff)
        if scored_tokens != exp_cnt:
            raise RuntimeError(
                f"protgpt2 E17 token count mismatch on {rid}: {scored_tokens} vs {exp_cnt}"
            )
    max_diff = max(diffs, default=0.0)
    if max_diff >= 1e-5:
        raise RuntimeError(
            f"protgpt2 E17 parity gate FAILED: max_abs_diff={max_diff:.6g} >= 1e-5"
        )
    res = {
        "gate": "protgpt2_e17_parity",
        "n_rows": len(diffs),
        "max_abs_diff_nats_per_token": max_diff,
        "passed": True,
    }
    print(json.dumps(res, indent=2), flush=True)
    return res


def _score_arm_pools(
    arm: str,
    pool_items: list[tuple[str, list[dict[str, Any]], Path]],
    *,
    device: str,
    model_root: Path | None,
    esm3_model_dir: Path | None,
    mask_batch_size: int,
) -> list[dict[str, Any]]:
    route, dtype = _route_for_arm(arm)
    t_load = time.monotonic()

    stage46 = None
    arm_obj = None
    adapter_gp = None
    handle_esm3 = None

    if route == "readout_stage46":
        from src.capability.readouts.readout_extraction import load_readout_arm

        stage46 = _load_stage46_module()
        arm_obj = load_readout_arm(arm, stage46, device=device, dtype=dtype)
        paradigm = "causal_next_token"
    elif route == "general_plm_causal":
        from src.capability.general_plm import load_general_plm

        adapter_gp = load_general_plm(
            arm, device=device, dtype=dtype, model_root=model_root
        )
        paradigm = "causal_next_token"
    elif route == "general_plm_masked":
        from src.capability.general_plm import load_general_plm

        adapter_gp = load_general_plm(
            arm, device=device, dtype=dtype, model_root=model_root
        )
        paradigm = "masked_marginal_pseudolikelihood"
    elif route == "esm3_masked":
        from src.capability.esm3_ext.esm3_adapter import load_esm3_offline

        handle_esm3 = load_esm3_offline(
            model_dir=esm3_model_dir,
            device=device,
            dtype=dtype,
            eager_load_aux=False,
        )
        paradigm = "masked_marginal_pseudolikelihood"
    else:
        raise ValueError(f"Unknown route {route!r}")

    print(
        f"[{arm}] loaded via {route} ({dtype}) on {device} in {time.monotonic() - t_load:.1f}s "
        f"for {len(pool_items)} pool(s)",
        flush=True,
    )

    cache: dict[tuple[str, str | None], tuple[float, int]] = {}
    summaries: list[dict[str, Any]] = []

    for pool_id, rows, out_path in pool_items:
        t0 = time.monotonic()
        scored_rows: list[dict[str, Any]] = []
        if arm == "protgpt2" and route == "readout_stage46":
            seqs_all = [str(row["sequence"]).strip().upper() for row in rows]
            for start_i in range(0, len(seqs_all), 8):
                chunk_seqs = seqs_all[start_i : start_i + 8]
                for seq, res_pair in zip(
                    chunk_seqs, _score_protgpt2_batch(arm_obj, chunk_seqs), strict=True
                ):
                    cache[(seq, None)] = res_pair
        for idx, row in enumerate(rows, 1):
            seq = str(row["sequence"]).strip().upper()
            seq_id = str(row.get("seq_id") or row.get("id") or f"{pool_id}_{idx - 1:04d}")
            class_key = row.get("class_key")
            key = (seq, str(class_key) if class_key is not None else None)
            if key not in cache:
                if route == "readout_stage46":
                    cache[key] = _score_single_stage46(
                        arm_obj, stage46, seq, class_key=key[1]
                    )
                elif route == "general_plm_causal":
                    cache[key] = _score_single_gp_causal(adapter_gp, seq)
                elif route == "general_plm_masked":
                    cache[key] = _score_single_gp_masked(
                        adapter_gp, seq, mask_batch_size=mask_batch_size
                    )
                else:
                    cache[key] = _score_single_esm3_masked(
                        handle_esm3, seq, mask_batch_size=mask_batch_size
                    )
            nll_sum, scored_tokens = cache[key]
            mean_tok = nll_sum / scored_tokens
            mean_res = nll_sum / len(seq)
            scored_rows.append(
                {
                    "seq_id": seq_id,
                    "id": seq_id,
                    "pool_id": pool_id,
                    "arm": arm,
                    "arm_id": arm,
                    "class_key": class_key,
                    "sequence": seq,
                    "sequence_sha256": str(row.get("sequence_sha256") or _seq_sha(seq)),
                    "length": len(seq),
                    "paradigm": paradigm,
                    "route": route,
                    "dtype": dtype,
                    "scored_tokens": int(scored_tokens),
                    "nll_sum_nats": float(nll_sum),
                    "mean_nll_per_scored_token": float(mean_tok),
                    "mean_nll_per_token_nats": float(mean_tok),
                    "mean_nll_per_residue": float(mean_res),
                    "mean_nll_per_residue_nats": float(mean_res),
                    "mean_pll_per_residue": float(-mean_res),
                    "high_likelihood": float(-mean_tok),
                }
            )
            if idx % 100 == 0 or idx == len(rows):
                print(
                    f"[{pool_id}] scored {idx}/{len(rows)} "
                    f"(unique={len(cache)}, {time.monotonic() - t0:.1f}s)",
                    flush=True,
                )
        _atomic_write_jsonl(out_path, scored_rows)
        vals = [r["mean_nll_per_scored_token"] for r in scored_rows]
        summaries.append(
            {
                "pool_id": pool_id,
                "arm": arm,
                "route": route,
                "dtype": dtype,
                "n_scored": len(scored_rows),
                "mean_nll_per_scored_token_mean": sum(vals) / len(vals) if vals else 0.0,
                "elapsed_seconds": round(time.monotonic() - t0, 2),
            }
        )

    del arm_obj, adapter_gp, handle_esm3
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", nargs="*", default=())
    parser.add_argument("--inputs-dir", type=Path, default=None)
    parser.add_argument("--scores-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--esm3-model-dir", type=Path, default=None)
    parser.add_argument("--mask-batch-size", type=int, default=256)
    parser.add_argument("--verify-protgpt2-e17", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_protgpt2_e17 is not None:
        _verify_protgpt2_e17(
            parity_jsonl=args.verify_protgpt2_e17,
            device=args.device,
        )
        if not args.pools:
            return

    if not args.pools or args.inputs_dir is None or args.scores_dir is None:
        parser.error("Provide --pools, --inputs-dir, and --scores-dir")

    by_arm: dict[str, list[tuple[str, list[dict[str, Any]], Path]]] = {}
    for pool_id in args.pools:
        in_path = args.inputs_dir / f"{pool_id}.jsonl"
        if not in_path.is_file():
            raise FileNotFoundError(f"Pool input JSONL not found: {in_path}")
        out_path = args.scores_dir / f"{pool_id}.jsonl"
        rows = [
            json.loads(line)
            for line in in_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"Empty pool input JSONL: {in_path}")
        arm = _resolve_arm_from_pool(pool_id, rows)
        by_arm.setdefault(arm, []).append((pool_id, rows, out_path))

    all_summaries: list[dict[str, Any]] = []
    for arm, items in by_arm.items():
        all_summaries.extend(
            _score_arm_pools(
                arm,
                items,
                device=args.device,
                model_root=args.model_root,
                esm3_model_dir=args.esm3_model_dir,
                mask_batch_size=args.mask_batch_size,
            )
        )
    print(json.dumps({"summaries": all_summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
