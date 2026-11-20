#!/usr/bin/env python3
"""ESM3-1.4B R5 Generative Expression Stage.

Implements all three R5 generative experiments for ESM3-1.4B:
  (a) R5a Modification-Extent Ladder:
      k = 1, 2, 5, 10, 20, 40 (CONDITION_BIDIRECTIONAL_INFILL) plus declared
      ``full_length_matched`` (100%-masked cosine unmasking at each parent's exact
      length L, CONDITION_DIFFUSION_LENGTH_SUPPLIED_FULL) on the 16 ladder backbones.
  (b) R5b Length-Distribution-Matched De Novo Sequence-Only Generation:
      N = 800 sequences (lengths matched to empirical natural Swiss-Prot / ProGen2
      distribution in 16..400 aa) via 100%-masked cosine unmasking, plus a 128-seq
      stratified ``fold_cohort.jsonl`` ready for ESMFold2 + Pfam + DIAMOND.
  (c) R5c Multimodal Conditioned Generation on the 16 Backbones:
      1. cond_3d_backbone       : inverse folding from backbone 3D coordinates + VQ-VAE
      2. cond_ss8_only          : SS8 secondary-structure track only
      3. cond_function_keywords : InterPro/Pfam/EC function-keyword track only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.esm3_ext.esm3_adapter import (  # noqa: E402
    encode_function_annotations,
    encode_ss8_tokens,
    encode_structure_tokens,
    exact_pll,
    generate_sequence_cosine,
    infill_window_cosine,
    load_backbone_from_structure_file,
    load_esm3_offline,
)

WINDOW_EXTENTS: tuple[int, ...] = (1, 2, 5, 10, 20, 40)
FULL_LENGTH_MATCHED_RUNG = "full_length_matched"
CONDITION_BIDIRECTIONAL_INFILL = "bidirectional_infill"
CONDITION_DIFFUSION_LENGTH_SUPPLIED_FULL = "diffusion_length_supplied_full"

# Verified in-vocabulary ESM3 TF-IDF keywords mapped from EC class/subclass
EC_SUBCLASS_KEYWORDS: dict[str, str] = {
    "1": "dehydrogenase",
    "2.1": "methyltransferase",
    "2.4": "enzyme",
    "2.5": "enzyme",
    "2.7": "kinase",
    "2": "enzyme",
    "3.1": "enzyme",
    "3.4": "peptidase",
    "3.5": "enzyme",
    "3.6": "phosphatase",
    "3": "enzyme",
    "4": "lyase",
    "5": "isomerase",
    "6": "ligase",
    "7": "translocase",
}


def _ec_to_keyword(ec: str) -> str:
    parts = [p for p in ec.strip().split(".") if p]
    if len(parts) >= 2 and f"{parts[0]}.{parts[1]}" in EC_SUBCLASS_KEYWORDS:
        return EC_SUBCLASS_KEYWORDS[f"{parts[0]}.{parts[1]}"]
    if parts and parts[0] in EC_SUBCLASS_KEYWORDS:
        return EC_SUBCLASS_KEYWORDS[parts[0]]
    return "enzyme"


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seq_identity(a: str, b: str) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    return float(sum(1 for x, y in zip(a, b) if x == y) / len(a))


def _write_json_with_sha256(path: Path, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def load_backbones(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_pfam_annotations(pfam_tsv: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Load ``uniprot -> [(pfam_id, start, end), ...]`` if TSV exists."""

    mapping: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    if not pfam_tsv.is_file():
        return mapping
    with pfam_tsv.open("r", encoding="utf-8") as fh:
        _ = fh.readline()
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    mapping[parts[0]].append((parts[3], int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    return mapping


def _find_backbone_structure(
    backbone: dict[str, Any],
    structures_dir: Path | None,
    alphafold_dir: Path,
) -> tuple[Path | None, str]:
    acc = str(backbone.get("accession", ""))
    bid = str(backbone.get("backbone_id", ""))
    sha = str(backbone.get("parent_sequence_sha256", ""))
    if structures_dir is not None:
        for stem in (bid, acc, sha, f"AF-{acc}-F1-model_v6"):
            if not stem:
                continue
            for ext in (".pdb", ".pdb.gz", ".cif", ".cif.gz"):
                cand = structures_dir / f"{stem}{ext}"
                if cand.is_file():
                    return cand, "supplied_structure_dir"
    for ext in (".pdb.gz", ".cif.gz", ".pdb", ".cif"):
        cand = alphafold_dir / f"AF-{acc}-F1-model_v6{ext}"
        if cand.is_file():
            return cand, "alphafold_db"
    return None, "missing"


def _predict_backbone_with_esm3(handle: Any, sequence: str) -> dict[str, Any]:
    """Fallback structure/SS8 prediction via ESM3's own structure & SS8 heads (flagged provenance)."""

    from esm.sdk.api import ESMProtein, GenerationConfig

    protein = ESMProtein(sequence=sequence)
    folded = handle.model.generate(
        protein,
        GenerationConfig(
            track="structure",
            num_steps=max(8, min(32, len(sequence) // 4)),
            temperature=0.0,
        ),
    )
    coords = folded.coordinates.detach().cpu().numpy().astype(np.float32)
    ss8_prot = handle.model.generate(
        ESMProtein(sequence=sequence, coordinates=folded.coordinates),
        GenerationConfig(track="secondary_structure", num_steps=8, temperature=0.0),
    )
    ss8 = ss8_prot.secondary_structure or ("_" * len(sequence))
    return {
        "sequence": sequence,
        "coords_atom37": coords,
        "ss8": ss8,
        "structure_source": "esm3_self_predicted_fallback",
    }


def sample_empirical_lengths(
    length_cohort: Path,
    *,
    n_samples: int,
    min_len: int = 16,
    max_len: int = 400,
    seed: int = 20261120,
) -> list[int]:
    """Sample empirical protein lengths matching the frozen natural Swiss-Prot cohort."""

    rng = np.random.default_rng(seed)
    pool: list[int] = []
    if length_cohort.is_file():
        data = json.loads(length_cohort.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "protein" in data and "records" in data["protein"]:
            pool = [
                len(s)
                for s in data["protein"]["records"]
                if isinstance(s, str) and min_len <= len(s) <= max_len
            ]
        elif isinstance(data, list):
            pool = [int(x) for x in data if min_len <= int(x) <= max_len]
    if not pool:
        # Clean deterministic fallback distribution spanning 64..320 aa
        pool = list(range(64, 321))
    replace = len(pool) < n_samples
    chosen = rng.choice(np.asarray(pool, dtype=np.int64), size=n_samples, replace=replace)
    return [int(x) for x in chosen.tolist()]


def run_r5a_ladder(
    handle: Any,
    backbones: Sequence[dict[str, Any]],
    *,
    n_draws: int,
    full_num_steps: int,
    temperature: float,
    top_p: float,
    score_pll: bool,
    batch_size: int,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    """Run R5a modification-extent ladder (k=1..40 + full_length_matched) on the 16 backbones."""

    counts: dict[str, int] = defaultdict(int)
    nll_by_rung: dict[str, list[float]] = defaultdict(list)
    id_by_rung: dict[str, list[float]] = defaultdict(list)

    with out_path.open("w", encoding="utf-8") as fh:
        for b_idx, bb in enumerate(backbones):
            parent = str(bb["sequence"])
            L = len(parent)
            spans = bb.get("window_spans", {})

            # Window rungs k = 1, 2, 5, 10, 20, 40
            for k in WINDOW_EXTENTS:
                rung = f"k{k}"
                if rung in spans:
                    start, stop = int(spans[rung][0]), int(spans[rung][1])
                else:
                    start = max(1, (L - k) // 2)
                    stop = start + k
                if stop >= L:
                    continue
                for draw_idx in range(n_draws):
                    d_seed = seed + b_idx * 10000 + k * 100 + draw_idx
                    gen = infill_window_cosine(
                        handle,
                        parent,
                        start=start,
                        extent=k,
                        seed=d_seed,
                        num_steps=k,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    seq = gen["sequence"]
                    pll_info = (
                        exact_pll(handle, seq, batch_size=batch_size) if score_pll else None
                    )
                    rec = {
                        "arm": "esm3-sm-open-v1",
                        "backbone_id": bb["backbone_id"],
                        "accession": bb["accession"],
                        "rung": rung,
                        "extent": k,
                        "condition": CONDITION_BIDIRECTIONAL_INFILL,
                        "draw_index": draw_idx,
                        "seed": d_seed,
                        "window_span": [start, stop],
                        "length": L,
                        "sequence": seq,
                        "sequence_sha256": _sha256_str(seq),
                        "window": gen["window"],
                        "seq_identity_to_parent": _seq_identity(seq, parent),
                        "window_identity_to_parent": _seq_identity(
                            gen["window"], parent[start:stop]
                        ),
                        "pll_nll_sum_nats": pll_info["nll_sum_nats"] if pll_info else None,
                        "pll_mean_nll_per_residue_nats": (
                            pll_info["mean_nll_per_residue_nats"] if pll_info else None
                        ),
                    }
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    counts[rung] += 1
                    id_by_rung[rung].append(rec["seq_identity_to_parent"])
                    if pll_info:
                        nll_by_rung[rung].append(float(pll_info["mean_nll_per_residue_nats"]))

            # Declared 100%-masked length-matched rung
            for draw_idx in range(n_draws):
                d_seed = seed + b_idx * 10000 + 9900 + draw_idx
                gen = generate_sequence_cosine(
                    handle,
                    L,
                    seed=d_seed,
                    num_steps=full_num_steps,
                    temperature=temperature,
                    top_p=top_p,
                )
                seq = gen["sequence"]
                pll_info = exact_pll(handle, seq, batch_size=batch_size) if score_pll else None
                rec = {
                    "arm": "esm3-sm-open-v1",
                    "backbone_id": bb["backbone_id"],
                    "accession": bb["accession"],
                    "rung": FULL_LENGTH_MATCHED_RUNG,
                    "extent": L,
                    "condition": CONDITION_DIFFUSION_LENGTH_SUPPLIED_FULL,
                    "draw_index": draw_idx,
                    "seed": d_seed,
                    "window_span": [0, L],
                    "length": L,
                    "sequence": seq,
                    "sequence_sha256": _sha256_str(seq),
                    "seq_identity_to_parent": _seq_identity(seq, parent),
                    "pll_nll_sum_nats": pll_info["nll_sum_nats"] if pll_info else None,
                    "pll_mean_nll_per_residue_nats": (
                        pll_info["mean_nll_per_residue_nats"] if pll_info else None
                    ),
                }
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                counts[FULL_LENGTH_MATCHED_RUNG] += 1
                id_by_rung[FULL_LENGTH_MATCHED_RUNG].append(rec["seq_identity_to_parent"])
                if pll_info:
                    nll_by_rung[FULL_LENGTH_MATCHED_RUNG].append(
                        float(pll_info["mean_nll_per_residue_nats"])
                    )
            fh.flush()

    return {
        "n_backbones": len(backbones),
        "n_draws_per_rung": n_draws,
        "counts_by_rung": dict(counts),
        "mean_seq_identity_by_rung": {
            r: float(np.mean(vs)) for r, vs in sorted(id_by_rung.items())
        },
        "mean_pll_nll_by_rung": {
            r: float(np.mean(vs)) for r, vs in sorted(nll_by_rung.items())
        },
        "products_jsonl": out_path.name,
    }


def run_r5b_denovo(
    handle: Any,
    *,
    lengths: Sequence[int],
    n_fold_subset: int,
    num_steps: int,
    temperature: float,
    top_p: float,
    score_pll: bool,
    batch_size: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Run R5b length-distribution-matched 800-seq de novo sequence-only generation."""

    all_path = out_dir / "denovo_800.jsonl"
    fold_path = out_dir / "fold_cohort.jsonl"
    rows: list[dict[str, Any]] = []

    with all_path.open("w", encoding="utf-8") as fh:
        for idx, length in enumerate(lengths):
            d_seed = seed + 500000 + idx
            gen = generate_sequence_cosine(
                handle,
                int(length),
                seed=d_seed,
                num_steps=num_steps,
                temperature=temperature,
                top_p=top_p,
            )
            seq = gen["sequence"]
            pll_info = exact_pll(handle, seq, batch_size=batch_size) if score_pll else None
            rec = {
                "arm": "esm3-sm-open-v1",
                "sample_id": f"esm3_denovo_{idx:04d}",
                "condition": "diffusion_length_matched_denovo_seqonly",
                "seed": d_seed,
                "length": int(length),
                "sequence": seq,
                "sequence_sha256": _sha256_str(seq),
                "num_steps": gen["num_steps"],
                "temperature": float(temperature),
                "top_p": float(top_p),
                "pll_nll_sum_nats": pll_info["nll_sum_nats"] if pll_info else None,
                "pll_mean_nll_per_residue_nats": (
                    pll_info["mean_nll_per_residue_nats"] if pll_info else None
                ),
            }
            rows.append(rec)
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            if (idx + 1) % 20 == 0:
                fh.flush()

    # Deterministic length-stratified 128-sequence subset formatted for ESMFold2 folding
    ordered = sorted(rows, key=lambda r: (r["length"], r["sample_id"]))
    n_take = min(int(n_fold_subset), len(ordered))
    if n_take > 0:
        pick = sorted({int(round(i * (len(ordered) - 1) / max(1, n_take - 1))) for i in range(n_take)})
        subset = [ordered[i] for i in pick]
    else:
        subset = []
    with fold_path.open("w", encoding="utf-8") as fh:
        for rec in subset:
            fh.write(
                json.dumps(
                    {
                        "id": rec["sample_id"],
                        "arm": rec["arm"],
                        "sequence": rec["sequence"],
                        "sequence_sha256": rec["sequence_sha256"],
                        "length": rec["length"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    nlls = [r["pll_mean_nll_per_residue_nats"] for r in rows if r["pll_mean_nll_per_residue_nats"] is not None]
    return {
        "n_generated": len(rows),
        "n_fold_subset": len(subset),
        "length_min": int(min(r["length"] for r in rows)) if rows else 0,
        "length_median": float(np.median([r["length"] for r in rows])) if rows else 0.0,
        "length_max": int(max(r["length"] for r in rows)) if rows else 0,
        "mean_pll_nll_per_residue_nats": float(np.mean(nlls)) if nlls else None,
        "denovo_jsonl": all_path.name,
        "fold_cohort_jsonl": fold_path.name,
    }


def run_r5c_conditioned(
    handle: Any,
    backbones: Sequence[dict[str, Any]],
    *,
    structures_dir: Path | None,
    alphafold_dir: Path,
    pfam_map: dict[str, list[tuple[str, int, int]]],
    allow_esm3_self_fold: bool,
    n_draws: int,
    num_steps: int,
    temperature: float,
    top_p: float,
    score_pll: bool,
    batch_size: int,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    """Run R5c 3D-backbone, SS8-only, and function-keyword conditioned generation."""

    summary_id: dict[str, list[float]] = defaultdict(list)
    summary_nll: dict[str, list[float]] = defaultdict(list)
    source_counts: dict[str, int] = defaultdict(int)

    with out_path.open("w", encoding="utf-8") as fh:
        for b_idx, bb in enumerate(backbones):
            parent = str(bb["sequence"])
            L = len(parent)
            acc = str(bb["accession"])
            ec = str(bb.get("ec_label", ""))

            struct_path, struct_source = _find_backbone_structure(bb, structures_dir, alphafold_dir)
            coords_atom37: np.ndarray | None = None
            ss8_str: str | None = None
            if struct_path is not None:
                rec_s = load_backbone_from_structure_file(struct_path)
                if len(rec_s["coords_atom37"]) == L:
                    coords_atom37 = rec_s["coords_atom37"]
                    ss8_str = rec_s["ss8"]
            if coords_atom37 is None and allow_esm3_self_fold:
                pred = _predict_backbone_with_esm3(handle, parent)
                coords_atom37 = pred["coords_atom37"]
                ss8_str = pred["ss8"]
                struct_source = pred["structure_source"]
            source_counts[struct_source] += 1

            # Build conditioning tracks
            conditions_to_run: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
            if coords_atom37 is not None:
                coords_pad, struct_toks = encode_structure_tokens(
                    handle, coords_atom37, reference_sequence=parent
                )
                conditions_to_run.append(
                    (
                        "cond_3d_backbone",
                        {"structure_tokens": struct_toks, "structure_coords": coords_pad},
                        {"structure_source": struct_source},
                    )
                )
            if ss8_str is not None and set(ss8_str) != {"_"}:
                ss8_toks = encode_ss8_tokens(handle, ss8_str)
                conditions_to_run.append(
                    (
                        "cond_ss8_only",
                        {"ss8_tokens": ss8_toks},
                        {"structure_source": struct_source},
                    )
                )

            # Function-keyword conditioning (EC/InterPro mapped into ESM3 vocabulary)
            kw = _ec_to_keyword(ec)
            anns: list[tuple[str, int, int]] = [(kw, 1, L)]
            for pfam_id, p_start, p_end in pfam_map.get(acc, []):
                s_clamped = max(1, min(L, p_start))
                e_clamped = max(s_clamped, min(L, p_end))
                anns.append((pfam_id, s_clamped, e_clamped))
            func_toks, res_toks = encode_function_annotations(handle, L, anns, strict=False)
            conditions_to_run.append(
                (
                    "cond_function_keywords",
                    {"function_tokens": func_toks, "residue_annotation_tokens": res_toks},
                    {"function_annotations": [list(a) for a in anns]},
                )
            )

            for c_idx, (cname, track_kwargs, meta) in enumerate(conditions_to_run):
                for draw_idx in range(n_draws):
                    d_seed = seed + 900000 + b_idx * 1000 + c_idx * 100 + draw_idx
                    gen = generate_sequence_cosine(
                        handle,
                        L,
                        seed=d_seed,
                        num_steps=num_steps,
                        temperature=temperature,
                        top_p=top_p,
                        **track_kwargs,
                    )
                    seq = gen["sequence"]
                    pll_info = (
                        exact_pll(handle, seq, batch_size=batch_size) if score_pll else None
                    )
                    seq_id = _seq_identity(seq, parent)
                    rec = {
                        "arm": "esm3-sm-open-v1",
                        "backbone_id": bb["backbone_id"],
                        "accession": acc,
                        "ec_label": ec,
                        "condition": cname,
                        "draw_index": draw_idx,
                        "seed": d_seed,
                        "length": L,
                        "sequence": seq,
                        "sequence_sha256": _sha256_str(seq),
                        "seq_identity_to_parent": seq_id,
                        "seq_only_pll_mean_nll_nats": (
                            pll_info["mean_nll_per_residue_nats"] if pll_info else None
                        ),
                        **meta,
                    }
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    summary_id[cname].append(seq_id)
                    if pll_info:
                        summary_nll[cname].append(float(pll_info["mean_nll_per_residue_nats"]))
            fh.flush()

    return {
        "n_backbones": len(backbones),
        "structure_source_counts": dict(source_counts),
        "mean_seq_recovery_by_condition": {
            c: float(np.mean(vs)) for c, vs in sorted(summary_id.items())
        },
        "mean_seq_only_pll_nll_by_condition": {
            c: float(np.mean(vs)) for c, vs in sorted(summary_nll.items())
        },
        "products_jsonl": out_path.name,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--mode", choices=("all", "ladder", "denovo", "conditioned"), default="all")
    p.add_argument(
        "--backbones",
        type=Path,
        default=DATA_ROOT / "results/audit_20261011_h200/ladder_original/ladder_backbones.jsonl",
    )
    p.add_argument("--backbone-structures-dir", type=Path, default=None)
    p.add_argument("--alphafold-dir", type=Path, default=DATA_ROOT / "data/alphafold")
    p.add_argument("--pfam-tsv", type=Path, default=DATA_ROOT / "data/interpro/pfam_residue.tsv")
    p.add_argument(
        "--length-cohort",
        type=Path,
        default=DATA_ROOT / "results/R5/context_homologue_20260826/cohort.json",
    )
    p.add_argument("--allow-esm3-self-fold", action="store_true", default=True)
    p.add_argument("--n-draws", type=int, default=8)
    p.add_argument("--n-denovo", type=int, default=800)
    p.add_argument("--n-fold-subset", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--ladder-temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-score-pll", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Optional cap for smoke testing")
    p.add_argument("--seed", type=int, default=20261120)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    handle = load_esm3_offline(args.model, device=args.device, dtype=args.dtype)
    score_pll = not args.no_score_pll

    backbones = load_backbones(args.backbones) if args.backbones.is_file() else []
    backbones = [b for i, b in enumerate(backbones) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        backbones = backbones[: args.limit]

    results: dict[str, Any] = {}
    if args.mode in ("all", "ladder") and backbones:
        results["r5a_ladder"] = run_r5a_ladder(
            handle,
            backbones,
            n_draws=args.n_draws,
            full_num_steps=args.num_steps,
            temperature=args.ladder_temperature,
            top_p=args.top_p,
            score_pll=score_pll,
            batch_size=args.batch_size,
            seed=args.seed,
            out_path=args.out / "ladder_products.jsonl",
        )

    if args.mode in ("all", "denovo"):
        n_denovo = min(args.n_denovo, args.limit) if args.limit > 0 else args.n_denovo
        lengths = sample_empirical_lengths(
            args.length_cohort, n_samples=n_denovo, seed=args.seed
        )
        lengths = [length for i, length in enumerate(lengths) if i % args.num_shards == args.shard_index]
        results["r5b_denovo"] = run_r5b_denovo(
            handle,
            lengths=lengths,
            n_fold_subset=min(args.n_fold_subset, len(lengths)),
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            score_pll=score_pll,
            batch_size=args.batch_size,
            seed=args.seed,
            out_dir=args.out,
        )

    if args.mode in ("all", "conditioned") and backbones:
        pfam_map = load_pfam_annotations(args.pfam_tsv)
        results["r5c_conditioned"] = run_r5c_conditioned(
            handle,
            backbones,
            structures_dir=args.backbone_structures_dir,
            alphafold_dir=args.alphafold_dir,
            pfam_map=pfam_map,
            allow_esm3_self_fold=args.allow_esm3_self_fold,
            n_draws=args.n_draws,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            score_pll=score_pll,
            batch_size=args.batch_size,
            seed=args.seed,
            out_path=args.out / "conditioned_products.jsonl",
        )

    payload = {
        "schema": "esm3_r5_generation_v1",
        "model_facts": handle.facts,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_seconds": round(time.time() - t0, 2),
        "results": results,
    }
    _write_json_with_sha256(args.out / "esm3_r5_generation.json", payload)


if __name__ == "__main__":
    main()
