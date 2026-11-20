#!/usr/bin/env python3
"""R4 multi-sequence & causal context-access runner on the frozen 800-unit
EXP-R2-228 cohort (``results/R5/context_homologue_20260826/cohort.json``).

Evaluates ``msa-transformer-100m``, ``protgpt3-msa``, ``dayhoff-3b-msa``,
``protgpt3-112m``, ``protgpt3-1.3b``, ``protgpt3-10b``, ``progen3-*``, and
``dayhoff-*`` across all 5 declared conditions (``position_only``,
``mono_shuffled``, ``no_context``, ``unrelated``, ``homologue``) and all 4
identity bands (``id_70_90``, ``id_50_70``, ``id_30_50``, ``id_lt_30``),
reporting exact paired AUROC and fractional NLL reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context import context_homologue as ch_mod
from src.capability.general_plm import arms_for_module, load_general_plm
from src.capability.general_plm.eval_utils import (
    atomic_write_json,
    cluster_bootstrap_mean,
    sha256_file,
)

SCHEMA = "general_plm_r4_multiseq_context_v1"
CONDITIONS = (
    ch_mod.POSITION_ONLY,
    ch_mod.MONO_SHUFFLED,
    ch_mod.NO_CONTEXT,
    ch_mod.UNRELATED,
    ch_mod.HOMOLOGUE,
)


def _deterministic_shuffle(seq: str, seed_key: str) -> str:
    digest = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    chars = list(seq)
    perm = rng.permutation(len(chars))
    return "".join(chars[int(i)] for i in perm)


def _pick_unrelated_partners(
    unit: dict[str, Any],
    records: list[str],
    related_map: dict[str, list[int]] | list[list[int]],
    n_needed: int,
    *,
    filler_idx: int,
) -> list[str]:
    """Deterministic matched-length unrelated records excluding near-duplicates/homologs."""
    target_idx = int(unit["target"])
    if isinstance(related_map, dict):
        forbidden = set(related_map.get(str(target_idx), ())) | set(related_map.get(target_idx, ()))
    else:
        forbidden = set(related_map[target_idx]) if target_idx < len(related_map) else set()
    forbidden.add(target_idx)
    forbidden.add(filler_idx)
    forbidden.update(int(p) for p in unit["partners"])

    # Match each homolog partner's length from a deterministic offset in the pool.
    digest = int(hashlib.sha256(unit["key"].encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    n_rec = len(records)
    picked: list[str] = []
    used: set[int] = set()
    partner_lens = [len(records[int(p)]) for p in unit["partners"][:n_needed]]

    for want_len in partner_lens:
        best_idx = None
        best_gap = 10**9
        # Sample 128 deterministic candidates to find closest length match without scanning all 20k.
        cands = rng.integers(0, n_rec, size=128)
        for cand in cands:
            ci = int(cand)
            if ci in forbidden or ci in used:
                continue
            gap = abs(len(records[ci]) - want_len)
            if gap < best_gap:
                best_gap = gap
                best_idx = ci
                if gap == 0:
                    break
        if best_idx is None:
            for ci in range(n_rec):
                if ci not in forbidden and ci not in used:
                    best_idx = ci
                    break
        if best_idx is not None:
            used.add(best_idx)
            picked.append(records[best_idx])
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=arms_for_module("r4"))
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--budget-tokens", type=int, default=1024, help="1024 matched budget or native 4096/8192")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stratum", default="retained", choices=("retained", "all"))
    parser.add_argument("--unaligned-msa-control", action="store_true", help="MSA Transformer unaligned-stack control")
    parser.add_argument("--max-units", type=int, default=0, help="0 = all units in selected stratum")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=DATA_ROOT / "results/R5/context_homologue_20260826/cohort.json",
    )
    parser.add_argument("--expect-file", default="")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    prot = cohort["protein"]
    records: list[str] = prot["records"]
    filler_idx = int(prot["filler"]) if isinstance(prot["filler"], int) else int(prot["filler"].get("index", 0))
    filler_seq = records[filler_idx] if isinstance(prot["filler"], int) else str(prot["filler"].get("sequence", records[filler_idx]))
    related_map = prot.get("related", {})

    units = prot["units"]
    if args.stratum != "all":
        units = [u for u in units if u.get("stratum") == args.stratum]
    if args.max_units > 0:
        units = units[: args.max_units]
    if not units:
        raise SystemExit("No R4 units selected")

    adapter = load_general_plm(
        args.arm,
        device=args.device,
        dtype=args.dtype,
        model_root=args.model_root,
    )

    t0 = time.monotonic()
    unit_rows: list[dict[str, Any]] = []
    for u_idx, unit in enumerate(units, 1):
        target_seq = records[int(unit["target"])]
        homolog_seqs = [records[int(p)] for p in unit["partners"]]
        unrelated_seqs = _pick_unrelated_partners(
            unit, records, related_map, len(homolog_seqs), filler_idx=filler_idx
        )
        shuffled_seqs = [
            _deterministic_shuffle(seq, f"{unit['key']}|mono|{i}")
            for i, seq in enumerate(homolog_seqs)
        ]
        # Length-matched fixed filler repeated to match each homolog's length.
        filler_seqs = [
            (filler_seq * ((len(seq) // len(filler_seq)) + 1))[: len(seq)]
            for seq in homolog_seqs
        ]

        cond_inputs = {
            ch_mod.POSITION_ONLY: filler_seqs,
            ch_mod.MONO_SHUFFLED: shuffled_seqs,
            ch_mod.NO_CONTEXT: [],
            ch_mod.UNRELATED: unrelated_seqs,
            ch_mod.HOMOLOGUE: homolog_seqs,
        }

        cond_scores: dict[str, Any] = {}
        for cond in CONDITIONS:
            cond_scores[cond] = adapter.score_r4_context(
                target_seq,
                cond_inputs[cond],
                budget_tokens=args.budget_tokens,
                batch_size=args.batch_size,
                aligned_msa=not args.unaligned_msa_control,
            )

        nll_hom = float(cond_scores[ch_mod.HOMOLOGUE]["nll_sum"])
        nll_unr = float(cond_scores[ch_mod.UNRELATED]["nll_sum"])
        nll_pos = float(cond_scores[ch_mod.POSITION_ONLY]["nll_sum"])
        nll_shuf = float(cond_scores[ch_mod.MONO_SHUFFLED]["nll_sum"])

        unit_rows.append(
            {
                "key": unit["key"],
                "band": unit["band"],
                "stratum": unit["stratum"],
                "group": int(unit["group"]),
                "target_length": len(target_seq),
                "conditions": cond_scores,
                "hom_beats_unrelated": 1.0 if nll_hom < nll_unr else (0.5 if nll_hom == nll_unr else 0.0),
                "hom_beats_mono_shuffled": 1.0 if nll_hom < nll_shuf else (0.5 if nll_hom == nll_shuf else 0.0),
                "fractional_reduction_vs_unrelated": (nll_unr - nll_hom) / nll_pos if nll_pos > 0 else None,
                "fractional_reduction_vs_shuffled": (nll_shuf - nll_hom) / nll_pos if nll_pos > 0 else None,
            }
        )
        if u_idx % 50 == 0 or u_idx == len(units):
            print(f"[R4:{adapter.arm}] {u_idx}/{len(units)} units", flush=True)

    # Aggregate dimensionless endpoints by identity band (cluster bootstrap over `group`).
    bands_summary: dict[str, Any] = {}
    for band in ("id_70_90", "id_50_70", "id_30_50", "id_lt_30"):
        sub = [r for r in unit_rows if r["band"] == band]
        if not sub:
            continue
        groups = [r["group"] for r in sub]
        bands_summary[band] = {
            "n_units": len(sub),
            "auroc_hom_vs_unrelated": cluster_bootstrap_mean(
                [r["hom_beats_unrelated"] for r in sub], groups, seed=20261125
            ),
            "auroc_hom_vs_mono_shuffled": cluster_bootstrap_mean(
                [r["hom_beats_mono_shuffled"] for r in sub], groups, seed=20261126
            ),
            "fractional_reduction_vs_unrelated": cluster_bootstrap_mean(
                [r["fractional_reduction_vs_unrelated"] for r in sub if r["fractional_reduction_vs_unrelated"] is not None],
                [r["group"] for r in sub if r["fractional_reduction_vs_unrelated"] is not None],
                seed=20261127,
            ),
            "mean_k_homologue": float(np.mean([r["conditions"][ch_mod.HOMOLOGUE]["k_items"] for r in sub])),
        }

    elapsed = time.monotonic() - t0
    suffix = f"_b{args.budget_tokens}" + ("_unaligned" if args.unaligned_msa_control else "")
    out_name = args.expect_file or f"r4_multiseq_{args.arm}{suffix}.json"
    atomic_write_json(
        args.out / out_name,
        {
            "schema": SCHEMA,
            "roster": "general_plm_20261120_exploratory",
            "arm": args.arm,
            "cohort_sha256": sha256_file(args.cohort),
            "budget_tokens": args.budget_tokens,
            "stratum": args.stratum,
            "aligned_msa": not args.unaligned_msa_control,
            "n_units": len(unit_rows),
            "elapsed_seconds": elapsed,
            "provenance": adapter.provenance(),
            "bands": bands_summary,
            "units": unit_rows,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[R4:{args.arm}] wrote {args.out / out_name}", flush=True)


if __name__ == "__main__":
    main()
