#!/usr/bin/env python3
"""Structural (ESMFold2-hf) and evolutionary (DIAMOND UniRef50) evaluation
pipeline for all R5 generated sequences (ESM3-1.4B, DPLM, ESM2, ProtGPT3,
ProGen3, Dayhoff, AMPLIFY, ESMC, SaProt, ProteinGLM-MLM).

Compatible with ``h200/run_external_baseline_h200.sh`` (accepts ``--device``
and ``--out`` first and writes atomic ``--expect r5_structure_eval.json``).
Must be invoked with ``runtimes/esmfold2/bin/python`` (``transformers>=5.16``
carrying ``EsmFold2Model``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.general_plm.eval_utils import (
    atomic_write_json,
    cluster_bootstrap_mean,
    safe_spearman,
)
from src.capability.generation import structure_evidence as se
from src.capability.ladder import fold_geometry as fg

SCHEMA = "general_plm_r5_structure_eval_v1"
AA20_SET = frozenset(AA20)
SEED_RULE = "sequence_digest"
DEFAULT_BASE_SEED = 20261120
IDENTITY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("lt_30", 0.0, 30.0),
    ("30_70", 30.0, 70.0),
    ("70_95", 70.0, 95.0),
    ("ge_95", 95.0, 100.1),
)


def _data_root() -> Path:
    env = os.environ.get("TRANSFER_DATA_ROOT", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    return ROOT


def _resolve_esmfold2_dir(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for var in ("ESMFOLD2_MODEL_DIR", "ESMFOLD2_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.append(Path(v))
    model_root = os.environ.get("MODEL_ROOT", "").strip()
    if model_root:
        candidates.append(Path(model_root) / "ESMFold2-hf")
    candidates.extend(
        [
            Path("/Data/lzp/models/ESMFold2-hf"),
            Path("/home/lzp/models/ESMFold2-hf"),
        ]
    )
    for cand in candidates:
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"ESMFold2-hf checkpoint not found. Checked {[str(c) for c in candidates]}"
    )


def _resolve_diamond(
    explicit_bin: Path | None, explicit_db: Path | None
) -> tuple[Path | None, Path | None]:
    bin_cands: list[Path] = []
    if explicit_bin is not None:
        bin_cands.append(explicit_bin)
    env_dir = os.environ.get("TRANSFER_DIAMOND_DIR", "").strip()
    if env_dir:
        bin_cands.append(Path(env_dir) / "diamond")
    bin_cands.append(ROOT / "external/tools/diamond/diamond")
    bin_cands.append(_data_root() / "external/tools/diamond/diamond")
    which = shutil.which("diamond")
    if which:
        bin_cands.append(Path(which))
    diamond_bin = next((p for p in bin_cands if p.is_file() and os.access(p, os.X_OK)), None)

    db_cands: list[Path] = []
    if explicit_db is not None:
        db_cands.append(explicit_db)
    env_db = os.environ.get("TRANSFER_DIAMOND_DB", "").strip()
    if env_db:
        db_cands.append(Path(env_db))
    db_cands.append(ROOT / "data/homology_db/uniref50_full.dmnd")
    db_cands.append(_data_root() / "data/homology_db/uniref50_full.dmnd")
    diamond_db = next((p for p in db_cands if p.is_file()), None)
    return diamond_bin, diamond_db


def _seq_sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _kabsch_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    rot, trans = fg._superpose(mobile, reference)
    aligned = fg._apply(rot, trans, mobile)
    diff = aligned - reference
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _identity_band(pident: float) -> str:
    for name, low, high in IDENTITY_BANDS:
        if low <= pident < high:
            return name
    return "ge_95" if pident >= 95.0 else "lt_30"


def normalize_r5_record(raw: dict[str, Any], source_file: str, line_no: int) -> dict[str, Any]:
    """Normalize ESM3-ext JSONLs and general-PLM ``r5_draws_*.jsonl`` into one common schema."""
    seq = str(raw.get("sequence") or "").strip().upper()
    rec_id = str(
        raw.get("record_id")
        or raw.get("id")
        or f"{Path(source_file).stem}__row{line_no:05d}"
    )
    arm = str(raw.get("arm") or rec_id.split("__", 1)[0] or "unknown")
    backbone_id = raw.get("backbone_id")
    rung = raw.get("rung")
    condition = raw.get("condition")
    if rung is not None:
        category = "ladder"
    elif isinstance(condition, str) and condition.startswith("cond_"):
        category = "conditioned"
    elif "conditioned" in Path(source_file).name:
        category = "conditioned"
    else:
        category = "denovo"

    # Model likelihood score (higher = more likely) for Spearman(model_score, pLDDT).
    model_score: float | None = None
    for nll_key in (
        "pll_nll_sum_nats",
        "nll_sum_nats",
        "pll_mean_nll_per_residue_nats",
        "seq_only_pll_mean_nll_nats",
    ):
        val = raw.get(nll_key)
        if val is not None and np.isfinite(float(val)):
            model_score = -float(val)
            break

    return {
        "record_id": rec_id,
        "source_file": Path(source_file).name,
        "arm": arm,
        "category": category,
        "backbone_id": str(backbone_id) if backbone_id is not None else None,
        "rung": str(rung) if rung is not None else None,
        "extent": int(raw["extent"]) if raw.get("extent") is not None else None,
        "condition": str(condition) if condition is not None else None,
        "draw_index": int(raw["draw_index"]) if raw.get("draw_index") is not None else None,
        "status": str(raw.get("status") or "filled"),
        "sequence": seq,
        "sequence_sha256": _seq_sha(seq) if seq else "",
        "length": len(seq),
        "model_score": model_score,
        "seq_identity_to_parent": (
            float(raw["seq_identity_to_parent"])
            if raw.get("seq_identity_to_parent") is not None
            else None
        ),
    }


def discover_input_jsonls(input_jsonls: list[Path], r5_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    for p in input_jsonls:
        cand = p if p.is_file() else (_data_root() / p)
        if not cand.is_file():
            raise FileNotFoundError(f"Input JSONL not found: {p}")
        paths.append(cand.resolve())
    if r5_dir is not None:
        base = r5_dir if r5_dir.is_dir() else (_data_root() / r5_dir)
        if not base.is_dir():
            raise FileNotFoundError(f"R5 directory not found: {r5_dir}")
        for name in ("ladder_products.jsonl", "fold_cohort.jsonl", "conditioned_products.jsonl"):
            cand = base / name
            if cand.is_file():
                paths.append(cand.resolve())
        for cand in sorted(base.glob("r5_draws_*.jsonl")):
            if cand.is_file():
                paths.append(cand.resolve())
    # Deduplicate preserving order
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    if not ordered:
        raise ValueError("No R5 JSONL files found; specify --input-jsonl and/or --r5-dir")
    return ordered


def _extract_ca_coords(
    output: Any, features: dict[str, Any], sequence: str, sample_idx: int
) -> np.ndarray:
    """Extract (L, 3) float64 CA trace for ``sample_idx`` via ``output_to_pdb``."""
    from transformers.models.esmfold2.protein_utils import output_to_pdb

    pdb_text = output_to_pdb(output, features, sample_idx=sample_idx)
    ca = fg.parse_ca_trace(pdb_text)
    if ca.shape != (len(sequence), 3):
        raise ValueError(
            f"ESMFold2 CA trace shape {ca.shape} does not match sequence length {len(sequence)}"
        )
    return ca


def fold_unique_sequences(
    sequences: dict[str, str],
    *,
    esmfold2_dir: Path,
    device: str,
    diffusion_samples: int,
    base_seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    """Fold every unique sequence once with ``biohub/ESMFold2-hf``."""
    from transformers import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import prepare_protein_features

    config = se.require_esmfold2_checkpoint(esmfold2_dir)
    dev = torch.device(device)
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

    results: dict[str, dict[str, Any]] = {}
    ca_traces: dict[str, np.ndarray] = {}
    t_fold = time.monotonic()
    items = sorted(sequences.items(), key=lambda kv: (len(kv[1]), kv[0]))

    for idx, (sha, seq) in enumerate(items, 1):
        seed = se.fold_seed(seq, base_seed=base_seed, rule=SEED_RULE)
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        t0 = time.monotonic()
        with torch.inference_mode():
            features = prepare_protein_features(seq, device=dev)
            output = model.fold(**features, num_diffusion_samples=int(diffusion_samples))
        summary = se.summarize_esmfold2(output, sequence=seq)
        sample_idx = int(summary["diffusion_sample_index"])
        ca = _extract_ca_coords(output, features, seq, sample_idx)
        ca_traces[sha] = ca
        results[sha] = {
            "length": len(seq),
            "mean_ca_plddt": float(summary["mean_ca_plddt"]),
            "fraction_ca_plddt_ge70": float(summary["fraction_ca_plddt_ge70"]),
            "ptm": float(summary["ptm"]),
            "mean_pae_angstrom": float(summary["mean_pae_angstrom"]),
            "diffusion_sample_index": sample_idx,
            "has_ca_trace": ca is not None,
            "fold_seconds": time.monotonic() - t0,
        }
        del output, features
        if idx % 50 == 0 or idx == len(items):
            print(f"[R5-eval:ESMFold2] folded {idx}/{len(items)} unique sequences", flush=True)

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    instrument = {
        "predictor": se.PREDICTOR_ID,
        "model_dir": str(esmfold2_dir),
        "model_type": config.get("model_type"),
        "diffusion_samples": int(diffusion_samples),
        "base_seed": int(base_seed),
        "dtype": "bfloat16",
        "load_seconds": load_sec,
        "fold_seconds_total": time.monotonic() - t_fold,
        "n_unique_folded": len(results),
    }
    return results, ca_traces, instrument


def run_diamond_uniref50(
    sequences: dict[str, str],
    *,
    diamond_bin: Path,
    diamond_db: Path,
    threads: int,
) -> dict[str, dict[str, Any]]:
    """Run DIAMOND blastp against UniRef50 and return best-hit identity per sequence SHA."""
    if not sequences:
        return {}
    with tempfile.TemporaryDirectory(prefix="r5_diamond_") as tmp_str:
        tmp = Path(tmp_str)
        query_faa = tmp / "queries.faa"
        out_tsv = tmp / "hits.tsv"
        with query_faa.open("w", encoding="utf-8") as fh:
            for sha, seq in sorted(sequences.items()):
                fh.write(f">{sha}\n{seq}\n")
        cmd = [
            str(diamond_bin),
            "blastp",
            "--query",
            str(query_faa),
            "--db",
            str(diamond_db),
            "--out",
            str(out_tsv),
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "qlen",
            "bitscore",
            "evalue",
            "--max-target-seqs",
            "1",
            "--sensitive",
            "--threads",
            str(max(1, int(threads))),
            "--quiet",
        ]
        subprocess.run(cmd, check=True)
        hits: dict[str, dict[str, Any]] = {}
        if out_tsv.is_file():
            for line in out_tsv.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                qseqid, sseqid, pident, aln_len, qlen, bitscore, evalue = line.split("\t")
                pident_f = float(pident)
                cov_id = pident_f * min(float(aln_len), float(qlen)) / max(1.0, float(qlen))
                prev = hits.get(qseqid)
                if prev is None or pident_f > prev["top_pident"]:
                    hits[qseqid] = {
                        "top_subject": sseqid,
                        "top_pident": pident_f,
                        "query_cov_identity": cov_id,
                        "bitscore": float(bitscore),
                        "evalue": float(evalue),
                    }
    return hits


def _stratify_by_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stratify records into <30%, 30-70%, 70-95%, >=95% DIAMOND identity bands."""
    total = len(rows)
    strata: dict[str, Any] = {}
    for band_name, _, _ in IDENTITY_BANDS:
        sub = [r for r in rows if r.get("identity_band") == band_name]
        plddts = [r["mean_ca_plddt"] for r in sub if r.get("mean_ca_plddt") is not None]
        ptms = [r["ptm"] for r in sub if r.get("ptm") is not None]
        lens = [r["length"] for r in sub]
        strata[band_name] = {
            "count": len(sub),
            "fraction": len(sub) / total if total else 0.0,
            "mean_plddt": float(np.mean(plddts)) if plddts else None,
            "fraction_plddt_ge70": float(np.mean([x >= 70.0 for x in plddts])) if plddts else None,
            "mean_ptm": float(np.mean(ptms)) if ptms else None,
            "mean_length": float(np.mean(lens)) if lens else None,
        }
    return strata


def summarize_evaluations(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate R5a ladder correlations, R5b de novo identity strata, and R5c conditioned metrics."""
    # 1. R5a modification ladder: grouped by (arm, condition, rung)
    ladder_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        if rec["category"] == "ladder" and rec.get("mean_ca_plddt") is not None:
            key = (rec["arm"], str(rec.get("condition") or "unknown"), str(rec.get("rung") or "unknown"))
            ladder_groups.setdefault(key, []).append(rec)

    ladder_summary: dict[str, Any] = {}
    for (arm, cond, rung), rows in sorted(ladder_groups.items()):
        by_bb: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            if r.get("backbone_id"):
                by_bb.setdefault(r["backbone_id"], []).append(r)

        rhos_plddt: list[float] = []
        rhos_plddt_bbs: list[str] = []
        rhos_tm: list[float] = []
        rhos_tm_bbs: list[str] = []
        for bb_id, bb_rows in sorted(by_bb.items()):
            sc = [r["model_score"] for r in bb_rows if r.get("model_score") is not None]
            pl = [r["mean_ca_plddt"] for r in bb_rows if r.get("model_score") is not None]
            rho = safe_spearman(sc, pl)
            if rho is not None:
                rhos_plddt.append(rho)
                rhos_plddt_bbs.append(bb_id)
            sc_tm = [
                r["model_score"]
                for r in bb_rows
                if r.get("model_score") is not None and r.get("tm_score_to_parent") is not None
            ]
            tms = [
                r["tm_score_to_parent"]
                for r in bb_rows
                if r.get("model_score") is not None and r.get("tm_score_to_parent") is not None
            ]
            rho_tm = safe_spearman(sc_tm, tms)
            if rho_tm is not None:
                rhos_tm.append(rho_tm)
                rhos_tm_bbs.append(bb_id)

        plddts = [r["mean_ca_plddt"] for r in rows]
        ptms = [r["ptm"] for r in rows]
        tms_all = [r["tm_score_to_parent"] for r in rows if r.get("tm_score_to_parent") is not None]
        rmsds_all = [r["scrmsd_angstrom"] for r in rows if r.get("scrmsd_angstrom") is not None]
        bbs_all = [str(r.get("backbone_id") or r["record_id"]) for r in rows]

        ladder_summary[f"{arm}|{cond}|{rung}"] = {
            "arm": arm,
            "condition": cond,
            "rung": rung,
            "n_records": len(rows),
            "n_backbones": len(by_bb),
            "mean_plddt": cluster_bootstrap_mean(plddts, bbs_all, seed=20261120),
            "mean_ptm": cluster_bootstrap_mean(ptms, bbs_all, seed=20261121),
            "mean_tm_score_to_parent": (
                cluster_bootstrap_mean(
                    tms_all,
                    [str(r.get("backbone_id") or r["record_id"]) for r in rows if r.get("tm_score_to_parent") is not None],
                    seed=20261122,
                )
                if tms_all
                else None
            ),
            "mean_scrmsd_angstrom": (
                cluster_bootstrap_mean(
                    rmsds_all,
                    [str(r.get("backbone_id") or r["record_id"]) for r in rows if r.get("scrmsd_angstrom") is not None],
                    seed=20261123,
                )
                if rmsds_all
                else None
            ),
            "within_backbone_spearman_score_plddt": cluster_bootstrap_mean(
                rhos_plddt, rhos_plddt_bbs, seed=20261124
            ),
            "within_backbone_spearman_score_tm": cluster_bootstrap_mean(
                rhos_tm, rhos_tm_bbs, seed=20261125
            ),
        }

    # 2. R5b de novo generation: grouped by arm
    denovo_by_arm: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if rec["category"] == "denovo" and rec.get("mean_ca_plddt") is not None:
            denovo_by_arm.setdefault(rec["arm"], []).append(rec)

    denovo_summary: dict[str, Any] = {}
    for arm, rows in sorted(denovo_by_arm.items()):
        plddts = [r["mean_ca_plddt"] for r in rows]
        ptms = [r["ptm"] for r in rows]
        lens = [r["length"] for r in rows]
        denovo_summary[arm] = {
            "n_folded": len(rows),
            "mean_plddt": float(np.mean(plddts)),
            "median_plddt": float(np.median(plddts)),
            "fraction_plddt_ge70": float(np.mean([x >= 70.0 for x in plddts])),
            "mean_ptm": float(np.mean(ptms)),
            "mean_length": float(np.mean(lens)),
            "uniref50_identity_strata": (
                _stratify_by_identity(rows) if any("identity_band" in r for r in rows) else None
            ),
        }

    # 3. R5c conditioned generation (e.g. cond_3d_backbone, cond_ss8, cond_function)
    cond_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        if rec["category"] == "conditioned" and rec.get("mean_ca_plddt") is not None:
            key = (rec["arm"], str(rec.get("condition") or "unknown"))
            cond_groups.setdefault(key, []).append(rec)

    cond_summary: dict[str, Any] = {}
    for (arm, cond), rows in sorted(cond_groups.items()):
        plddts = [r["mean_ca_plddt"] for r in rows]
        ptms = [r["ptm"] for r in rows]
        rmsds = [r["scrmsd_angstrom"] for r in rows if r.get("scrmsd_angstrom") is not None]
        tms = [r["tm_score_to_parent"] for r in rows if r.get("tm_score_to_parent") is not None]
        recov = [r["seq_identity_to_parent"] for r in rows if r.get("seq_identity_to_parent") is not None]
        cond_summary[f"{arm}|{cond}"] = {
            "arm": arm,
            "condition": cond,
            "n_folded": len(rows),
            "mean_plddt": float(np.mean(plddts)),
            "fraction_plddt_ge70": float(np.mean([x >= 70.0 for x in plddts])),
            "mean_ptm": float(np.mean(ptms)),
            "mean_scrmsd_angstrom": float(np.mean(rmsds)) if rmsds else None,
            "median_scrmsd_angstrom": float(np.median(rmsds)) if rmsds else None,
            "fraction_scrmsd_lt_2A": float(np.mean([x < 2.0 for x in rmsds])) if rmsds else None,
            "mean_tm_score_to_parent": float(np.mean(tms)) if tms else None,
            "fraction_tm_ge_0_5": float(np.mean([x >= 0.5 for x in tms])) if tms else None,
            "mean_seq_identity_to_parent": float(np.mean(recov)) if recov else None,
            "uniref50_identity_strata": (
                _stratify_by_identity(rows) if any("identity_band" in r for r in rows) else None
            ),
        }

    return {
        "r5a_ladder": ladder_summary,
        "r5b_denovo": denovo_summary,
        "r5c_conditioned": cond_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--r5-dir", type=Path, default=None)
    parser.add_argument("--esmfold2-dir", type=Path, default=None)
    parser.add_argument("--diamond-bin", type=Path, default=None)
    parser.add_argument("--diamond-db", type=Path, default=None)
    parser.add_argument(
        "--backbones",
        type=Path,
        default=None,
        help="Optional ladder_backbones.jsonl for parent CA-trace scRMSD and TM-score",
    )
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-sequences", type=int, default=0, help="0 = fold all unique eligible sequences")
    parser.add_argument("--diamond-threads", type=int, default=8)
    parser.add_argument("--skip-diamond", action="store_true")
    parser.add_argument("--expect-file", default="r5_structure_eval.json")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    input_paths = discover_input_jsonls(args.input_jsonl, args.r5_dir)
    esmfold2_dir = _resolve_esmfold2_dir(args.esmfold2_dir)

    records: list[dict[str, Any]] = []
    for path in input_paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            records.append(normalize_r5_record(json.loads(line), str(path), line_no))

    # Load parent backbone sequences if available (for exact scRMSD & TM-score).
    bb_path = args.backbones or (_data_root() / "results/audit_20261011_h200/ladder_original/ladder_backbones.jsonl")
    if not bb_path.is_file():
        alt = ROOT / "results/audit_20261011_h200/ladder_original/ladder_backbones.jsonl"
        bb_path = alt if alt.is_file() else bb_path
    parent_seqs: dict[str, str] = {}
    if bb_path.is_file():
        for line in bb_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                parent_seqs[str(row["backbone_id"])] = str(row["sequence"]).strip().upper()

    # Deduplicate eligible AA20 sequences.
    unique_seqs: dict[str, str] = {}
    ineligible_reasons: dict[str, int] = {}
    referenced_bbs: set[str] = set()
    for rec in records:
        seq = rec["sequence"]
        reason = se.eligibility(seq, min_length=args.min_length, max_length=args.max_length)
        if reason is not None:
            rec["eligibility_error"] = reason
            ineligible_reasons[reason] = ineligible_reasons.get(reason, 0) + 1
            continue
        sha = rec["sequence_sha256"]
        if args.max_sequences > 0 and sha not in unique_seqs and len(unique_seqs) >= args.max_sequences:
            continue
        unique_seqs[sha] = seq
        if rec.get("backbone_id") in parent_seqs:
            referenced_bbs.add(rec["backbone_id"])

    # Also fold referenced parent backbones (only when <=16 unique parents) so scRMSD/TM are exact.
    parent_shas: dict[str, str] = {}
    for bb_id in sorted(referenced_bbs):
        pseq = parent_seqs[bb_id]
        psha = _seq_sha(pseq)
        parent_shas[bb_id] = psha
        if psha not in unique_seqs and args.max_sequences == 0:
            unique_seqs[psha] = pseq
        elif psha not in unique_seqs and args.max_sequences > 0 and len(referenced_bbs) <= 2:
            unique_seqs[psha] = pseq

    fold_results, ca_traces, instrument = fold_unique_sequences(
        unique_seqs,
        esmfold2_dir=esmfold2_dir,
        device=args.device,
        diffusion_samples=args.diffusion_samples,
        base_seed=args.base_seed,
    )

    # Optional DIAMOND UniRef50 novelty oracle.
    diamond_bin, diamond_db = _resolve_diamond(args.diamond_bin, args.diamond_db)
    diamond_status = "skipped_by_flag"
    diamond_hits: dict[str, dict[str, Any]] = {}
    if not args.skip_diamond:
        if diamond_bin is not None and diamond_db is not None:
            t_dia = time.monotonic()
            diamond_hits = run_diamond_uniref50(
                unique_seqs,
                diamond_bin=diamond_bin,
                diamond_db=diamond_db,
                threads=args.diamond_threads,
            )
            diamond_status = f"completed_in_{time.monotonic() - t_dia:.1f}s"
        else:
            diamond_status = "skipped_missing_binary_or_db"

    # Attach fold metrics, parent scRMSD/TM-score, and DIAMOND strata to every record.
    for rec in records:
        sha = rec["sequence_sha256"]
        f_res = fold_results.get(sha)
        if f_res is None:
            continue
        rec.update(f_res)
        bb_id = rec.get("backbone_id")
        if bb_id and bb_id in parent_shas:
            p_ca = ca_traces.get(parent_shas[bb_id])
            m_ca = ca_traces.get(sha)
            if p_ca is not None and m_ca is not None and p_ca.shape == m_ca.shape:
                rec["scrmsd_angstrom"] = _kabsch_rmsd(m_ca, p_ca)
                rec["tm_score_to_parent"] = float(fg.tm_score(m_ca, p_ca)["tm_score"])
        if diamond_status.startswith("completed"):
            hit = diamond_hits.get(sha)
            top_pident = float(hit["top_pident"]) if hit is not None else 0.0
            rec["uniref50_top_pident"] = top_pident
            rec["uniref50_query_cov_identity"] = float(hit["query_cov_identity"]) if hit is not None else 0.0
            rec["uniref50_top_subject"] = hit["top_subject"] if hit is not None else None
            rec["identity_band"] = _identity_band(top_pident)

    # Persist per-record JSONL beside the summary JSON.
    folded_jsonl = args.out / "r5_folded_records.jsonl"
    with folded_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    summary = summarize_evaluations(records)
    payload = {
        "schema": SCHEMA,
        "roster": "general_plm_20261120_exploratory",
        "inputs": [str(p) for p in input_paths],
        "n_records_total": len(records),
        "n_records_folded": int(sum(1 for r in records if r.get("mean_ca_plddt") is not None)),
        "n_unique_sequences_folded": len(fold_results),
        "ineligible_counts": ineligible_reasons,
        "esmfold2_instrument": instrument,
        "diamond": {
            "status": diamond_status,
            "binary": str(diamond_bin) if diamond_bin else None,
            "db": str(diamond_db) if diamond_db else None,
            "n_unique_hits": len(diamond_hits),
        },
        "folded_records_jsonl": folded_jsonl.name,
        **summary,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.out / args.expect_file, payload)
    print(f"[R5-eval] wrote {args.out / args.expect_file}", flush=True)


if __name__ == "__main__":
    main()
