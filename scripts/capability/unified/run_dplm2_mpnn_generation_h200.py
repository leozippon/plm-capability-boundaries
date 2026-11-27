#!/usr/bin/env python3
"""Shardable H200 GPU runner for the 15-model DPLM-2 (x4) + ProteinMPNN (x11)
generation, folding, and native self-likelihood expansion across all 5 benchmarks:

  - ``exp4_1``:  Length-matched unconditional (DPLM-2 seq_only uniform/maxconf & co_generation)
                 and 3D-backbone inverse folding (DPLM-2 struct_conditioned & 11 ProteinMPNN).
  - ``r5a``:     Local span infilling on the 16 ``ladder_backbones.jsonl`` parents across
                 ``k in (1, 2, 5, 10, 20, 40)`` x 2 replicates.
  - ``exp4_2``:  Eight sealed domains (``1E0L..3DKM``): full 5-arm Potts-guided phase space
                 (``N, Pr, Pr_strong, C, Cx``) for DPLM-2 ``seq_only`` with ``< 1e-6`` bias
                 verification, ``N`` for DPLM-2 ``co_generation`` & ``struct_conditioned``,
                 and ``N`` (T=0.1) + ``N_T10`` (T=1.0) for all 11 ProteinMPNN checkpoints.
  - ``r4_cont``: Result 5 Downstream 2 continuation across the 22 canonical targets:
                 unaligned homolog-prefix continuation for DPLM-2 ``seq_only`` (budget=1024,
                 ``uniform_random``) + 50%-WT-prefix 3D-backbone suffix continuation for
                 DPLM-2 ``struct_conditioned`` and all 11 ProteinMPNN checkpoints.
  - ``fold``:    OOM-safe 32-sample ESMFold2 3D folding (strictly 1 worker per H200 GPU)
                 producing ``pLDDT``, ``pTM``, ``scTM`` / ``TM-score``, and ``[N,CA,C,O]``
                 refolded backbones for Exp 4.3 self-consistency likelihood scoring.
  - ``exp4_3``:  Native self-likelihood scoring (DPLM-2 ``exact_pll`` in ``seq_only`` and
                 ``struct_conditioned``; ProteinMPNN ``exact_sequence_ll_batch`` on BOTH
                 ESMFold2-refolded and input template backbones).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context.family_potts import load_family_model  # noqa: E402
from src.capability.generation import structure_evidence as se  # noqa: E402
from src.capability.ladder import fold_geometry as fg  # noqa: E402
from src.capability.unified import r4_continuation_yardstick as r4c  # noqa: E402
from src.capability.unified import r6_experiment_4_2 as exp42  # noqa: E402
from src.capability.unified import yardstick_77_80_generation as ygen  # noqa: E402
from src.capability.unified.roster import (  # noqa: E402
    DIMA_4_ARMS,
    DPLM2_4_ARMS,
    EXPANDED_19_NEW_ARMS,
    PROTEINMPNN_11_ARMS,
)
from src.capability.unified.wt_backbones_3d import load_backbone_npz  # noqa: E402

FOLD_SEED_RULE = "sequence_digest"
DEFAULT_MODEL_ROOT = Path.home() / "models" / "general_plm_20261120"
ALL_STAGES = ("exp4_1", "r5a", "exp4_2", "r4_cont", "fold", "exp4_3")


def _resolve_path(p: Path) -> Path:
    if p.is_absolute() or p.exists():
        return p.resolve() if p.exists() else p
    env_root = os.environ.get("TRANSFER_DATA_ROOT", "").strip()
    if env_root:
        cand = Path(env_root) / p
        if cand.exists():
            return cand.resolve()
    return REPO_ROOT / p


def _resolve_esmfold2_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for cand in (
        os.environ.get("ESMFOLD2_DIR"),
        os.environ.get("ESMFOLD2_MODEL_DIR"),
        str(Path.home() / "models" / "esmfold2-hf"),
        "/gpfs/jiaotongdamoxing/zhk_zip/models/esmfold2-hf",
    ):
        if cand and Path(cand).is_dir():
            return Path(cand)
    raise FileNotFoundError("Could not locate esmfold2-hf; pass --esmfold2-dir")


def _parse_model_filter(spec: str) -> tuple[set[str], set[str] | None]:
    """Return ``(allowed_arms, allowed_modes_or_None)`` from ``--models`` flag."""
    raw = spec.strip()
    if raw == "all":
        return set(EXPANDED_19_NEW_ARMS), None
    if raw == "dplm2":
        return set(DPLM2_4_ARMS), None
    if raw == "dplm2_seq":
        return set(DPLM2_4_ARMS), {"seq_only", "co_generation"}
    if raw == "dplm2_struct":
        return set(DPLM2_4_ARMS), {"struct_conditioned"}
    if raw == "proteinmpnn":
        return set(PROTEINMPNN_11_ARMS), None
    if raw == "dima":
        return set(DIMA_4_ARMS), None
    chosen = {x.strip() for x in raw.split(",") if x.strip()}
    unknown = chosen - set(EXPANDED_19_NEW_ARMS)
    if unknown:
        raise ValueError(f"Unknown model(s) in --models: {sorted(unknown)}")
    return chosen, None


def _select_shard(items: Sequence[Any], shard_index: int, num_shards: int) -> list[Any]:
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_index]


def _load_bb_coords(wt_bb_dir: Path, sequence: str) -> np.ndarray | None:
    try:
        rec = load_backbone_npz(wt_bb_dir, sequence)
        return np.asarray(rec["coords"], dtype=np.float32)
    except Exception:
        return None


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def _default_dplm2_dtype(arm_id: str, override: str | None) -> str:
    if override:
        return override
    return "bfloat16" if arm_id == "dplm2-3b" else "bfloat16"


class _AdapterPool:
    """Lazy single-active-model cache so GPU VRAM stays bounded while loading each weight once."""

    def __init__(
        self,
        *,
        model_root: Path,
        device: str,
        dtype_override: str | None = None,
        dima_root: Path | None = None,
    ) -> None:
        self.model_root = Path(model_root)
        self.dima_root = Path(dima_root) if dima_root is not None else None
        self.device = device
        self.dtype_override = dtype_override
        self._struct_tok: Any = None
        self._active_key: tuple[str, str] | None = None
        self._active_adapter: Any = None

    def get_struct_tokenizer(self) -> Any:
        if self._struct_tok is None:
            from src.capability.unified.dplm2_adapter import StructTokenizerEncoder

            self._struct_tok = StructTokenizerEncoder.from_pretrained(
                self.model_root, device=self.device
            )
        return self._struct_tok

    def get_dplm2(self, arm_id: str, mode: str) -> Any:
        from src.capability.unified.dplm2_adapter import load_dplm2_adapter

        eff_mode = "struct_conditioned" if mode == "struct_conditioned" else "seq_only"
        key = (arm_id, "dplm2")
        if self._active_key == key and self._active_adapter is not None:
            if eff_mode == "struct_conditioned" and self._active_adapter._struct_tokenizer is None:
                self._active_adapter._struct_tokenizer = self.get_struct_tokenizer()
            self._active_adapter.mode = eff_mode
            return self._active_adapter
        self.release_model()
        st_tok = self.get_struct_tokenizer() if eff_mode == "struct_conditioned" else None
        self._active_adapter = load_dplm2_adapter(
            arm_id,
            mode=eff_mode,
            model_root=self.model_root,
            device=self.device,
            dtype=_default_dplm2_dtype(arm_id, self.dtype_override),
            struct_tokenizer=st_tok,
        )
        self._active_key = key
        return self._active_adapter

    def get_proteinmpnn(self, arm_id: str) -> Any:
        from src.capability.unified.proteinmpnn_adapter import load_proteinmpnn_adapter

        key = (arm_id, "proteinmpnn")
        if self._active_key == key and self._active_adapter is not None:
            return self._active_adapter
        self.release_model()
        self._active_adapter = load_proteinmpnn_adapter(
            arm_id,
            model_root=self.model_root,
            device=self.device,
            dtype="float32",
        )
        self._active_key = key
        return self._active_adapter

    def get_dima(self, arm_id: str, *, load_encoder: bool = True) -> Any:
        from src.capability.unified.dima_adapter import load_dima_adapter

        key = (arm_id, f"dima_enc{int(load_encoder)}")
        if self._active_key == key and self._active_adapter is not None:
            return self._active_adapter
        # Reuse already-loaded encoder-bearing adapter when load_encoder=False is requested.
        if (
            not load_encoder
            and self._active_key == (arm_id, "dima_enc1")
            and self._active_adapter is not None
        ):
            return self._active_adapter
        self.release_model()
        eff_dtype = self.dtype_override or (
            "bfloat16" if arm_id in ("dima-esm2-3b", "dima-cheap-1024") else "float32"
        )
        self._active_adapter = load_dima_adapter(
            arm_id,
            dima_root=self.dima_root,
            model_root=self.model_root,
            device=self.device,
            dtype=eff_dtype,
            use_ema=True,
            load_encoder=load_encoder,
        )
        self._active_key = key
        return self._active_adapter

    def release_model(self) -> None:
        if self._active_adapter is not None:
            del self._active_adapter
            self._active_adapter = None
            self._active_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release_all(self) -> None:
        self.release_model()
        if self._struct_tok is not None:
            del self._struct_tok
            self._struct_tok = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Stage 1: Experiment 4.1 Generation
# --------------------------------------------------------------------------- #


def run_exp4_1_stage(args: argparse.Namespace, pool: _AdapterPool) -> None:
    allowed_arms, allowed_modes = _parse_model_filter(args.models)
    specs = [
        s
        for s in ygen.EXP4_1_NEW_SPECS
        if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes)
    ]
    assigned = _select_shard(specs, args.shard_index, args.num_shards)
    if not assigned:
        print(f"[exp4_1 shard {args.shard_index}/{args.num_shards}] nothing assigned", flush=True)
        return

    targets = ygen.select_exp4_1_reference_targets(REPO_ROOT, n_targets=args.exp4_1_pool_size)
    if args.max_items is not None:
        targets = targets[: args.max_items]
    wt_bb_dir = _resolve_path(args.wt_backbones_dir)
    out_dir = _resolve_path(args.work_dir) / "exp4_1" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in assigned:
        out_p = out_dir / f"{spec.file_slug}.jsonl"
        if args.skip_existing and not args.overwrite and _line_count(out_p) >= len(targets):
            print(f"[exp4_1] {spec.entry_key}: cached ({_line_count(out_p)} rows), skipping", flush=True)
            continue

        t0 = time.monotonic()
        rows: list[dict[str, Any]] = []
        skipped_missing_bb = 0

        if spec.backend == "dplm2":
            adapter = pool.get_dplm2(spec.arm_id, spec.mode)
            for idx, tgt in enumerate(targets):
                seed = ygen.deterministic_seed("exp4_1", spec.entry_key, tgt["seq_id"], base_seed=args.base_seed)
                coords = None
                if spec.requires_backbone_3d:
                    coords = _load_bb_coords(wt_bb_dir, tgt["sequence"])
                    if coords is None:
                        skipped_missing_bb += 1
                        continue
                steps = min(args.dplm2_steps, int(tgt["length"]))
                smode = spec.diffusion_mode or "uniform_random"
                res = adapter.sample_diffusion_full(
                    length=int(tgt["length"]),
                    seed=seed,
                    steps=steps,
                    temperature=spec.temperature,
                    top_p=0.95,
                    sampling_mode=smode,
                    coords=coords,
                )
                seq = str(res["sequence"]).strip().upper()
                rows.append(
                    {
                        "seq_id": f"{spec.file_slug}_{idx:04d}",
                        "entry_key": spec.entry_key,
                        "arm_id": spec.arm_id,
                        "mode": spec.mode,
                        "conditioning_regime": spec.conditioning_regime,
                        "diffusion_mode": smode,
                        "temperature": spec.temperature,
                        "seed": seed,
                        "template_id": tgt["seq_id"],
                        "template_sha256": tgt["sequence_sha256"],
                        "sequence": seq,
                        "sequence_sha256": ygen.seq_sha256(seq),
                        "length": len(seq),
                        "template_sequence_recovery": (
                            ygen.sequence_identity_exact(seq, tgt["sequence"])
                            if spec.requires_backbone_3d
                            else None
                        ),
                    }
                )
        elif spec.backend == "dima":
            adapter = pool.get_dima(spec.arm_id, load_encoder=False)
            ode_flag = spec.mode == "ode"
            bsz = max(1, int(args.dima_batch_size))
            for start in range(0, len(targets), bsz):
                chunk = targets[start : start + bsz]
                lengths = [int(t["length"]) for t in chunk]
                chunk_seed = ygen.deterministic_seed(
                    "exp4_1", spec.entry_key, chunk[0]["seq_id"], base_seed=args.base_seed
                )
                with torch.no_grad():
                    x_mean, attn_mask, _ = adapter._run_reverse_diffusion(
                        lengths,
                        steps=args.dima_steps,
                        ode_sampling=ode_flag,
                        seed=chunk_seed,
                    )
                    seqs, threedis, _, _ = adapter.decode_sequences_from_latents(
                        x_mean, lengths, attention_mask=attn_mask
                    )
                for offset_i, (tgt, seq, td) in enumerate(zip(chunk, seqs, threedis)):
                    idx = start + offset_i
                    seq_u = str(seq).strip().upper()
                    row_d: dict[str, Any] = {
                        "seq_id": f"{spec.file_slug}_{idx:04d}",
                        "entry_key": spec.entry_key,
                        "arm_id": spec.arm_id,
                        "mode": spec.mode,
                        "conditioning_regime": spec.conditioning_regime,
                        "diffusion_mode": spec.diffusion_mode,
                        "temperature": spec.temperature,
                        "seed": chunk_seed + offset_i,
                        "template_id": tgt["seq_id"],
                        "template_sha256": tgt["sequence_sha256"],
                        "sequence": seq_u,
                        "sequence_sha256": ygen.seq_sha256(seq_u),
                        "length": len(seq_u),
                        "template_sequence_recovery": None,
                    }
                    if td is not None:
                        row_d["threedi_sequence"] = td
                    rows.append(row_d)
        else:
            adapter = pool.get_proteinmpnn(spec.arm_id)
            for idx, tgt in enumerate(targets):
                coords = _load_bb_coords(wt_bb_dir, tgt["sequence"])
                if coords is None:
                    skipped_missing_bb += 1
                    continue
                seed = ygen.deterministic_seed("exp4_1", spec.entry_key, tgt["seq_id"], base_seed=args.base_seed)
                gen = adapter.generate(
                    coords,
                    n_samples=1,
                    seed=seed,
                    temperature=spec.temperature,
                    omit_X=True,
                )
                seq = str(gen[0]["sequence"]).strip().upper()
                rows.append(
                    {
                        "seq_id": f"{spec.file_slug}_{idx:04d}",
                        "entry_key": spec.entry_key,
                        "arm_id": spec.arm_id,
                        "mode": spec.mode,
                        "conditioning_regime": spec.conditioning_regime,
                        "diffusion_mode": None,
                        "temperature": spec.temperature,
                        "seed": seed,
                        "template_id": tgt["seq_id"],
                        "template_sha256": tgt["sequence_sha256"],
                        "sequence": seq,
                        "sequence_sha256": ygen.seq_sha256(seq),
                        "length": len(seq),
                        "template_sequence_recovery": ygen.sequence_identity_exact(seq, tgt["sequence"]),
                    }
                )

        if rows:
            ygen.atomic_write_jsonl(out_p, rows)
        print(
            f"[exp4_1 shard {args.shard_index}/{args.num_shards}] {spec.entry_key}: "
            f"{len(rows)} seqs (missing_bb={skipped_missing_bb}) in {time.monotonic() - t0:.1f}s",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Stage 2: R5a Local Infilling Generation
# --------------------------------------------------------------------------- #


def run_r5a_stage(args: argparse.Namespace, pool: _AdapterPool) -> None:
    allowed_arms, allowed_modes = _parse_model_filter(args.models)
    specs = [
        s
        for s in ygen.R5A_NEW_SPECS
        if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes)
    ]
    assigned = _select_shard(specs, args.shard_index, args.num_shards)
    if not assigned:
        print(f"[r5a shard {args.shard_index}/{args.num_shards}] nothing assigned", flush=True)
        return

    backbones = ygen.load_r5a_ladder_backbones(REPO_ROOT)
    if args.max_items is not None:
        backbones = backbones[: args.max_items]
    wt_bb_dir = _resolve_path(args.wt_backbones_dir)
    out_dir = _resolve_path(args.work_dir) / "r5a" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = len(backbones) * len(ygen.R5A_RUNGS) * ygen.R5A_REPLICATES

    for spec in assigned:
        out_p = out_dir / f"{spec.file_slug}.jsonl"
        if args.skip_existing and not args.overwrite and _line_count(out_p) >= expected_rows:
            print(f"[r5a] {spec.entry_key}: cached ({_line_count(out_p)} rows), skipping", flush=True)
            continue

        t0 = time.monotonic()
        rows: list[dict[str, Any]] = []
        skipped_missing_bb = 0
        adapter = (
            pool.get_dplm2(spec.arm_id, spec.mode)
            if spec.backend == "dplm2"
            else (
                pool.get_dima(spec.arm_id, load_encoder=True)
                if spec.backend == "dima"
                else pool.get_proteinmpnn(spec.arm_id)
            )
        )

        for bb in backbones:
            parent_seq = bb["sequence"]
            coords = _load_bb_coords(wt_bb_dir, parent_seq) if spec.requires_backbone_3d else None
            if spec.requires_backbone_3d and coords is None:
                skipped_missing_bb += 1
                continue
            st_tokens = (
                adapter.encode_backbone(coords)
                if (spec.backend == "dplm2" and spec.requires_backbone_3d and coords is not None)
                else None
            )
            for rung in ygen.R5A_RUNGS:
                span = bb["window_spans"].get(rung)
                if span is None:
                    continue
                start_idx, end_idx = int(span[0]), int(span[1])
                extent = end_idx - start_idx
                wt_win = parent_seq[start_idx:end_idx]
                for rep in range(ygen.R5A_REPLICATES):
                    seed = ygen.deterministic_seed(
                        "r5a", spec.entry_key, bb["backbone_id"], rung, rep, base_seed=args.base_seed
                    )
                    if spec.backend == "dplm2":
                        res = adapter.infill_window(
                            parent_seq,
                            start=start_idx,
                            extent=extent,
                            seed=seed,
                            steps=min(extent, args.dplm2_steps),
                            temperature=spec.temperature,
                            top_p=0.95,
                            sampling_mode="uniform_random",
                            struct_tokens=st_tokens,
                            mode=spec.mode,
                        )
                        seq = str(res["sequence"]).strip().upper()
                    elif spec.backend == "dima":
                        res = adapter.infill(
                            parent_seq,
                            start=start_idx,
                            extent=extent,
                            steps=args.dima_steps,
                            ode_sampling=False,
                            seed=seed,
                        )
                        seq = str(res["sequence"]).strip().upper()
                    else:
                        res = adapter.infill_window(
                            parent_seq,
                            coords,
                            start=start_idx,
                            extent=extent,
                            seed=seed,
                            temperature=spec.temperature,
                        )
                        seq = str(res["sequence"]).strip().upper()
                    win_seq = seq[start_idx:end_idx]
                    rows.append(
                        {
                            "seq_id": f"{spec.file_slug}__{bb['backbone_id']}__{rung}__r{rep}",
                            "entry_key": spec.entry_key,
                            "arm_id": spec.arm_id,
                            "mode": spec.mode,
                            "conditioning_regime": spec.conditioning_regime,
                            "backbone_id": bb["backbone_id"],
                            "parent_sequence_sha256": bb["sequence_sha256"],
                            "rung": rung,
                            "window_start": start_idx,
                            "window_end": end_idx,
                            "extent": extent,
                            "replicate": rep,
                            "seed": seed,
                            "sequence": seq,
                            "sequence_sha256": ygen.seq_sha256(seq),
                            "length": len(seq),
                            "window_sequence_recovery": ygen.sequence_identity_exact(win_seq, wt_win),
                        }
                    )

        if rows:
            ygen.atomic_write_jsonl(out_p, rows)
        print(
            f"[r5a shard {args.shard_index}/{args.num_shards}] {spec.entry_key}: "
            f"{len(rows)} seqs (missing_bb={skipped_missing_bb}) in {time.monotonic() - t0:.1f}s",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Stage 3: Experiment 4.2 Eight-Domain Phase Space Generation
# --------------------------------------------------------------------------- #


def run_exp4_2_stage(args: argparse.Namespace, pool: _AdapterPool) -> None:
    allowed_arms, allowed_modes = _parse_model_filter(args.models)
    specs = [
        s
        for s in ygen.EXP4_2_NEW_SPECS
        if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes)
    ]
    assigned = _select_shard(specs, args.shard_index, args.num_shards)
    if not assigned:
        print(f"[exp4_2 shard {args.shard_index}/{args.num_shards}] nothing assigned", flush=True)
        return

    family_ref_dir = _resolve_path(exp42.DEFAULT_FAMILY_REF_DIR)
    wt_bb_dir = _resolve_path(args.wt_backbones_dir)
    _, wt_seqs, _ = exp42.load_frozen_wt_ca_traces(_resolve_path(exp42.DEFAULT_WT_REF_MANIFEST))
    domains = list(exp42.SEALED_8_DOMAINS)
    if args.max_items is not None:
        domains = domains[: args.max_items]
    family_models = {d: load_family_model(d, family_ref_dir) for d in domains}
    out_dir = _resolve_path(args.work_dir) / "exp4_2" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_size = args.exp4_2_pool_size

    for spec in assigned:
        adapter = (
            pool.get_dplm2(spec.arm_id, spec.mode)
            if spec.backend == "dplm2"
            else (
                pool.get_dima(spec.arm_id, load_encoder=False)
                if spec.backend == "dima"
                else pool.get_proteinmpnn(spec.arm_id)
            )
        )
        for dom in domains:
            fm = family_models[dom]
            coords = _load_bb_coords(wt_bb_dir, wt_seqs[dom]) if spec.requires_backbone_3d else None
            if spec.requires_backbone_3d and coords is None:
                raise FileNotFoundError(f"Exp 4.2 missing WT backbone for {dom}")

            for arm_key in spec.arm_keys:
                out_p = out_dir / f"{spec.file_slug}__{dom}__{arm_key}.jsonl"
                if args.skip_existing and not args.overwrite and _line_count(out_p) >= pool_size:
                    continue
                t0 = time.monotonic()
                sseed = exp42.cell_stream_seed(spec.entry_key, dom, arm_key, base_seed=args.base_seed)

                if spec.backend == "dplm2" and spec.mode in ("seq_only", "struct_conditioned"):
                    arm_spec = exp42.PHASE_SPACE_ARM_MAP[arm_key]
                    records = ygen.sample_guided_dplm2(
                        adapter=adapter,
                        model_id=spec.entry_key,
                        family_model=fm,
                        arm_spec=arm_spec,
                        count=pool_size,
                        batch_size=args.dplm2_batch_size,
                        stream_seed=sseed,
                        mode=spec.mode,
                        coords=coords,
                    )
                elif spec.backend == "dima":
                    arm_spec = exp42.PHASE_SPACE_ARM_MAP[arm_key]
                    records = ygen.sample_guided_dima(
                        adapter=adapter,
                        model_id=spec.entry_key,
                        family_model=fm,
                        arm_spec=arm_spec,
                        count=pool_size,
                        batch_size=args.dima_batch_size,
                        stream_seed=sseed,
                        steps=args.dima_steps,
                        ode_sampling=False,
                    )
                elif spec.backend == "dplm2" and spec.mode == "co_generation":
                    arm_spec = exp42.PHASE_SPACE_ARM_MAP["N"]
                    records = []
                    for i in range(pool_size):
                        res = adapter.sample_diffusion_full(
                            length=fm.length,
                            seed=sseed + i,
                            steps=min(fm.length, args.dplm2_steps),
                            temperature=1.0,
                            top_p=0.95,
                            sampling_mode="co_generation",
                        )
                        seq = str(res["sequence"]).strip().upper()
                        records.append(
                            exp42._make_generated_record(
                                model_id=spec.entry_key,
                                domain=dom,
                                arm_spec=arm_spec,
                                seed=sseed + i,
                                sequence_index=i,
                                sequence=seq,
                                applied_bias=0.0,
                                fields_score=None,
                                coupling_score=None,
                                mean_step_entropy=0.0,
                                sampler="dplm2_co_generation",
                                token_steps=int(res["steps"]),
                            )
                        )
                else:
                    # ProteinMPNN inverse folding on the WT backbone (T=0.1 for N, T=1.0 for N_T10).
                    temp = 1.0 if arm_key == "N_T10" else 0.1
                    gen = adapter.generate(
                        coords,
                        n_samples=pool_size,
                        seed=sseed,
                        temperature=temp,
                        omit_X=True,
                    )
                    arm_spec = exp42.PhaseSpaceArmSpec(
                        arm_key=arm_key,
                        guidance_arm="inverse_folding_backbone3d",
                        lambda_fields=0.0,
                        lambda_coupling=0.0,
                        description=f"ProteinMPNN 3D-backbone inverse folding T={temp}",
                    )
                    records = []
                    for i, gen_row in enumerate(gen):
                        seq = str(gen_row["sequence"]).strip().upper()
                        records.append(
                            exp42._make_generated_record(
                                model_id=spec.entry_key,
                                domain=dom,
                                arm_spec=arm_spec,
                                seed=sseed + i,
                                sequence_index=i,
                                sequence=seq,
                                applied_bias=0.0,
                                fields_score=None,
                                coupling_score=None,
                                mean_step_entropy=0.0,
                                sampler=f"proteinmpnn_T{temp:.1f}",
                                token_steps=fm.length,
                            )
                        )

                for rec in records:
                    rec["entry_key"] = spec.entry_key
                    rec["arm_id"] = spec.arm_id
                    rec["conditioning_regime"] = spec.conditioning_regime
                ygen.atomic_write_jsonl(out_p, records)
                print(
                    f"[exp4_2 shard {args.shard_index}/{args.num_shards}] "
                    f"{spec.entry_key}/{dom}/{arm_key}: {len(records)} seqs in {time.monotonic() - t0:.1f}s",
                    flush=True,
                )


# --------------------------------------------------------------------------- #
# Stage 4: Result 5 Downstream 2 Continuation Generation
# --------------------------------------------------------------------------- #


def run_r4_cont_stage(args: argparse.Namespace, pool: _AdapterPool) -> None:
    allowed_arms, allowed_modes = _parse_model_filter(args.models)
    specs = [
        s
        for s in ygen.R4_CONT_NEW_SPECS
        if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes)
    ]
    assigned = _select_shard(specs, args.shard_index, args.num_shards)
    if not assigned:
        print(f"[r4_cont shard {args.shard_index}/{args.num_shards}] nothing assigned", flush=True)
        return

    contexts = json.loads(_resolve_path(r4c.DEFAULT_HOMOLOGS_JSON).read_text(encoding="utf-8"))
    plans = r4c.build_canonical_target_plans(contexts, budget=r4c.BUDGET_TOKENS)
    wt_bb_dir = _resolve_path(args.wt_backbones_dir)
    out_dir = _resolve_path(args.work_dir) / "r4_cont" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ids = list(r4c.ALL_22_TARGETS)
    if args.max_items is not None:
        target_ids = target_ids[: args.max_items]

    for spec in assigned:
        out_p = out_dir / f"{spec.file_slug}.jsonl"
        adapter = (
            pool.get_dplm2(spec.arm_id, spec.mode)
            if spec.backend == "dplm2"
            else (
                pool.get_dima(spec.arm_id, load_encoder=True)
                if spec.backend == "dima"
                else pool.get_proteinmpnn(spec.arm_id)
            )
        )

        if spec.conditioning_regime == "unaligned_homolog_prefix":
            cells = [
                c
                for c in r4c.required_target_condition_cells()
                if c[0] in set(target_ids)
            ]
            expected = len(cells) * args.r4_attempts_per_cell
            if args.skip_existing and not args.overwrite and _line_count(out_p) >= expected:
                print(f"[r4_cont] {spec.entry_key}: cached ({_line_count(out_p)} rows), skipping", flush=True)
                continue
            t0 = time.monotonic()
            rows: list[dict[str, Any]] = []
            for tid, cond in cells:
                plan = plans[tid]
                ctx_dicts = plan["items"].get(cond, [])
                ctx_seqs = [str(it["sequence"]) for it in ctx_dicts]
                ctx_subjects = [str(it["subject"]) for it in ctx_dicts]
                gen_len = min(len(str(plan["wildtype"])), r4c.DEFAULT_MAX_NEW_TOKENS)
                seed = r4c.cell_seed(spec.entry_key, tid, cond, base=args.base_seed)
                if spec.backend == "dima":
                    seqs, n_ctx_used, ctx_res = ygen.sample_dima_homolog_continuation(
                        adapter=adapter,
                        context_items=ctx_seqs,
                        gen_length=gen_len,
                        n_samples=args.r4_attempts_per_cell,
                        seed=seed,
                        budget_tokens=ygen.DIMA_MAX_RESIDUES,
                        steps=args.dima_steps,
                        batch_size=min(args.dima_batch_size, args.r4_attempts_per_cell),
                    )
                else:
                    seqs, n_ctx_used, ctx_res = ygen.sample_dplm2_homolog_continuation(
                        adapter=adapter,
                        context_items=ctx_seqs,
                        gen_length=gen_len,
                        n_samples=args.r4_attempts_per_cell,
                        seed=seed,
                        budget_tokens=r4c.BUDGET_TOKENS,
                        steps=min(gen_len, r4c.DIFFUSION_MAX_STEPS),
                        batch_size=min(args.dplm2_batch_size, args.r4_attempts_per_cell),
                    )
                used_ctx_seqs = ctx_seqs[:n_ctx_used]
                used_ctx_subjs = ctx_subjects[:n_ctx_used]
                for idx, seq in enumerate(seqs):
                    copy_stats = r4c.H.copy_statistics(seq, used_ctx_seqs)
                    rows.append(
                        {
                            "schema_version": r4c.ATTEMPT_SCHEMA,
                            "attempt_id": f"{spec.file_slug}__{tid}__{cond}__{idx:02d}",
                            "model": spec.entry_key,
                            "arm_id": spec.arm_id,
                            "conditioning_regime": spec.conditioning_regime,
                            "target_id": tid,
                            "cluster": plan["cluster"],
                            "condition": cond,
                            "declared_condition": r4c.CONDITION_TO_E09_BIN[cond],
                            "attempt_index": idx,
                            "seed": seed + idx,
                            "sequence": seq,
                            "sequence_sha256": ygen.seq_sha256(seq),
                            "residues": len(seq),
                            "valid_aa20": True,
                            "stopped_by_eos": True,
                            "stop_status": (
                                "native_terminal"
                                if len(seq) >= r4c.MIN_PRODUCT_RESIDUES
                                else "empty_or_noncanonical"
                            ),
                            "prompt_tokens": int(ctx_res),
                            "context_items_used": int(n_ctx_used),
                            "context_residues_used": int(ctx_res),
                            "context_subjects": used_ctx_subjs,
                            "shannon_H_bits": r4c.shannon_entropy_bits(seq),
                            "distinct_5mer": r4c.distinct_kmer_fraction(seq, 5),
                            **copy_stats,
                        }
                    )
            ygen.atomic_write_jsonl(out_p, rows)
            print(
                f"[r4_cont shard {args.shard_index}/{args.num_shards}] {spec.entry_key}: "
                f"{len(rows)} attempts in {time.monotonic() - t0:.1f}s",
                flush=True,
            )
        else:
            # 3D-backbone 50%-WT-prefix suffix continuation on the 22 targets (8 attempts/target -> 176 total).
            n_per_target = min(args.r4_attempts_per_cell, 8)
            expected = len(target_ids) * n_per_target
            if args.skip_existing and not args.overwrite and _line_count(out_p) >= expected:
                print(f"[r4_cont] {spec.entry_key}: cached ({_line_count(out_p)} rows), skipping", flush=True)
                continue
            t0 = time.monotonic()
            rows = []
            skipped_bb = 0
            for tid in target_ids:
                plan = plans[tid]
                wt = str(plan["wildtype"])
                coords = _load_bb_coords(wt_bb_dir, wt)
                if coords is None:
                    skipped_bb += 1
                    continue
                prefix_len = len(wt) // 2
                prefix = wt[:prefix_len]
                wt_suffix = wt[prefix_len:]
                seed = r4c.cell_seed(spec.entry_key, tid, "wt_prefix_50pct", base=args.base_seed)
                if spec.backend == "dplm2":
                    st_tokens = adapter.encode_backbone(coords)
                    seqs = []
                    for idx in range(n_per_target):
                        res = adapter.continue_suffix(
                            prefix=prefix,
                            target_length=len(wt),
                            seed=seed + idx,
                            steps=min(len(wt_suffix), args.dplm2_steps),
                            temperature=1.0,
                            top_p=0.95,
                            sampling_mode="uniform_random",
                            struct_tokens=st_tokens,
                            mode="struct_conditioned",
                        )
                        seqs.append(str(res["sequence"]).strip().upper())
                else:
                    gen = adapter.continue_suffix(
                        wt,
                        coords,
                        prefix_len=prefix_len,
                        n_samples=n_per_target,
                        seed=seed,
                        temperature=0.1,
                        n_to_c_order=True,
                    )
                    seqs = [str(row["sequence"]).strip().upper() for row in gen]
                for idx, seq in enumerate(seqs):
                    rows.append(
                        {
                            "schema_version": r4c.ATTEMPT_SCHEMA,
                            "attempt_id": f"{spec.file_slug}__{tid}__wt50__{idx:02d}",
                            "model": spec.entry_key,
                            "arm_id": spec.arm_id,
                            "conditioning_regime": spec.conditioning_regime,
                            "target_id": tid,
                            "cluster": plan["cluster"],
                            "condition": "wt_prefix_50pct",
                            "prefix_len": prefix_len,
                            "attempt_index": idx,
                            "seed": seed + idx,
                            "sequence": seq,
                            "sequence_sha256": ygen.seq_sha256(seq),
                            "residues": len(seq),
                            "valid_aa20": True,
                            "suffix_sequence_recovery": ygen.sequence_identity_exact(
                                seq[prefix_len:], wt_suffix
                            ),
                        }
                    )
            if rows:
                ygen.atomic_write_jsonl(out_p, rows)
            print(
                f"[r4_cont shard {args.shard_index}/{args.num_shards}] {spec.entry_key}: "
                f"{len(rows)} suffix attempts (missing_bb={skipped_bb}) in {time.monotonic() - t0:.1f}s",
                flush=True,
            )


# --------------------------------------------------------------------------- #
# Stage 5: ESMFold2 3D Folding (OOM-safe, 1 worker/GPU, deduplicated)
# --------------------------------------------------------------------------- #


def _load_esmfold2(esmfold2_dir: Path, device: str) -> tuple[Any, torch.device]:
    from transformers import EsmFold2Model

    se.require_esmfold2_checkpoint(esmfold2_dir)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = (
        EsmFold2Model.from_pretrained(esmfold2_dir, local_files_only=True, dtype=torch.bfloat16)
        .eval()
        .to(dev)
    )
    return model, dev


def _fold_one_seq_with_backbone(
    model: Any, dev: torch.device, seq: str, *, base_seed: int, diffusion_samples: int
) -> tuple[dict[str, float], np.ndarray]:
    from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features

    seed = se.fold_seed(seq, base_seed=base_seed, rule=FOLD_SEED_RULE)
    samples_tried = [int(diffusion_samples)]
    if len(seq) > 700 and int(diffusion_samples) > 8:
        samples_tried.extend([8, 4])

    last_oom: Exception | None = None
    out = None
    feats = None
    used_samples = int(diffusion_samples)
    for n_samp in samples_tried:
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        try:
            with torch.inference_mode():
                feats = prepare_protein_features(seq, device=dev)
                out = model.fold(**feats, num_diffusion_samples=n_samp)
            used_samples = n_samp
            last_oom = None
            break
        except torch.OutOfMemoryError as exc:
            last_oom = exc
            out = None
            feats = None
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    if last_oom is not None or out is None or feats is None:
        raise last_oom or RuntimeError(f"Failed to fold sequence of length {len(seq)}")

    summ = se.summarize_esmfold2(out, sequence=seq)
    s_idx = int(summ["diffusion_sample_index"])
    pdb_text = output_to_pdb(out, feats, sample_idx=s_idx)
    bb_coords = ygen.parse_pdb_backbone_ncaco(pdb_text, expected_len=len(seq))
    metrics = {
        "mean_ca_plddt": float(summ["mean_ca_plddt"]),
        "ptm": float(summ["ptm"]),
        "mean_pae_angstrom": float(summ["mean_pae_angstrom"]),
        "diffusion_sample_index": s_idx,
        "diffusion_samples_used": used_samples,
    }
    del out, feats
    if len(seq) > 600 and dev.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, bb_coords


def run_fold_stage(args: argparse.Namespace) -> None:
    work_dir = _resolve_path(args.work_dir)
    wt_bb_dir = _resolve_path(args.wt_backbones_dir)
    allowed_arms, allowed_modes = _parse_model_filter(args.models)
    fold_targets = {t.strip() for t in args.fold_benchmarks.split(",") if t.strip()}

    # Build a deterministic flat list of folding jobs across the requested benchmarks.
    jobs: list[tuple[str, str, Path, Path, dict[str, Any]]] = []

    if "exp4_1" in fold_targets:
        for s in ygen.EXP4_1_NEW_SPECS:
            if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes):
                gp = work_dir / "exp4_1" / "generated" / f"{s.file_slug}.jsonl"
                fp = work_dir / "exp4_1" / "folded" / f"{s.file_slug}.jsonl"
                if gp.is_file():
                    jobs.append(("exp4_1", s.file_slug, gp, fp, {"spec": s}))

    if "r5a" in fold_targets:
        for s in ygen.R5A_NEW_SPECS:
            if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes):
                gp = work_dir / "r5a" / "generated" / f"{s.file_slug}.jsonl"
                fp = work_dir / "r5a" / "folded" / f"{s.file_slug}.jsonl"
                if gp.is_file():
                    jobs.append(("r5a", s.file_slug, gp, fp, {"spec": s}))

    if "exp4_2" in fold_targets:
        for s in ygen.EXP4_2_NEW_SPECS:
            if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes):
                for dom in exp42.SEALED_8_DOMAINS:
                    for ak in s.arm_keys:
                        stem = f"{s.file_slug}__{dom}__{ak}"
                        gp = work_dir / "exp4_2" / "generated" / f"{stem}.jsonl"
                        fp = work_dir / "exp4_2" / "folded" / f"{stem}.jsonl"
                        if gp.is_file():
                            jobs.append(("exp4_2", stem, gp, fp, {"spec": s, "domain": dom, "arm_key": ak}))

    if "r4_cont" in fold_targets:
        for s in ygen.R4_CONT_NEW_SPECS:
            if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes):
                gp = work_dir / "r4_cont" / "generated" / f"{s.file_slug}.jsonl"
                fp = work_dir / "r4_cont" / "folded" / f"{s.file_slug}.jsonl"
                if gp.is_file():
                    jobs.append(("r4_cont", s.file_slug, gp, fp, {"spec": s}))

    assigned = _select_shard(jobs, args.shard_index, args.num_shards)
    todo = [
        j
        for j in assigned
        if not (args.skip_existing and not args.overwrite and j[3].is_file() and _line_count(j[3]) > 0)
    ]
    print(
        f"[fold shard {args.shard_index}/{args.num_shards}] assigned={len(assigned)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        return

    # Load reference traces needed for TM-score calculation.
    wt_exp4_2_ca, wt_exp4_2_seqs, _ = exp42.load_frozen_wt_ca_traces(
        _resolve_path(exp42.DEFAULT_WT_REF_MANIFEST)
    )
    exp4_2_fms = {
        d: load_family_model(d, _resolve_path(exp42.DEFAULT_FAMILY_REF_DIR))
        for d in exp42.SEALED_8_DOMAINS
    }
    msa_dir = _resolve_path(exp42.DEFAULT_MSA_DIR)
    exp4_2_msa_seqs: dict[str, list[str]] = {}
    for d in exp42.SEALED_8_DOMAINS:
        msa_p = msa_dir / f"{d}.a3m"
        if msa_p.is_file():
            exp4_2_msa_seqs[d] = [
                ln.strip().upper().replace("-", "")
                for ln in msa_p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith(">")
            ]

    r4_wt_ca: dict[str, np.ndarray] = {}
    r4_wt_npz = _resolve_path(r4c.DEFAULT_WORK_DIR) / "wt_ca_traces.npz"
    if r4_wt_npz.is_file():
        npz = np.load(r4_wt_npz, allow_pickle=False)
        r4_wt_ca = {k: np.asarray(npz[k], dtype=np.float64) for k in npz.files}

    exp4_1_templates = {
        t["seq_id"]: t["sequence"]
        for t in ygen.select_exp4_1_reference_targets(REPO_ROOT, n_targets=400)
    }
    r5a_parents = {b["backbone_id"]: b["sequence"] for b in ygen.load_r5a_ladder_backbones(REPO_ROOT)}

    esmfold2_dir = _resolve_esmfold2_dir(args.esmfold2_dir)
    model, dev = _load_esmfold2(esmfold2_dir, args.device)
    fold_cache: dict[str, tuple[dict[str, float], np.ndarray]] = {}

    try:
        for bench, stem, gp, fp, meta in todo:
            t0 = time.monotonic()
            gen_rows = ygen.read_jsonl(gp)
            if bench == "r4_cont":
                spec: ygen.R4ContNewSpec = meta["spec"]
                if spec.conditioning_regime == "unaligned_homolog_prefix":
                    gen_rows = r4c.select_fold_candidates(gen_rows, max_per_cell=r4c.FOLD_CANDIDATES_PER_CELL)
                else:
                    # Select up to 8 folded attempts per target (<= 176 folds/model).
                    gen_rows = gen_rows[:176]

            folded_rows: list[dict[str, Any]] = []
            refolded_bb_map: dict[str, np.ndarray] = {}

            for rec in gen_rows:
                seq = str(rec["sequence"]).strip().upper()
                sha = str(rec.get("sequence_sha256") or ygen.seq_sha256(seq))
                cached = fold_cache.get(sha)
                if cached is None:
                    cached = _fold_one_seq_with_backbone(
                        model,
                        dev,
                        seq,
                        base_seed=args.base_seed,
                        diffusion_samples=args.diffusion_samples,
                    )
                    fold_cache[sha] = cached
                s_met, bb_coords = cached
                mob_ca = bb_coords[:, 1, :].astype(np.float64)

                if bench == "exp4_1":
                    sc_tm = None
                    tid = rec.get("template_id")
                    if tid and tid in exp4_1_templates:
                        t_coords = _load_bb_coords(wt_bb_dir, exp4_1_templates[tid])
                        if t_coords is not None and t_coords.shape[0] == mob_ca.shape[0]:
                            sc_tm = float(fg.tm_score(mob_ca, t_coords[:, 1, :].astype(np.float64))["tm_score"])
                    refolded_bb_map[str(rec["seq_id"])] = bb_coords.astype(np.float32)
                    folded_rows.append(
                        {
                            "seq_id": rec["seq_id"],
                            "sequence": seq,
                            "sequence_sha256": sha,
                            "length": len(seq),
                            "status": "ok",
                            "backbone_sc_tm": sc_tm,
                            **s_met,
                        }
                    )
                elif bench == "r5a":
                    tm_parent = None
                    bb_id = str(rec["backbone_id"])
                    if bb_id in r5a_parents:
                        p_coords = _load_bb_coords(wt_bb_dir, r5a_parents[bb_id])
                        if p_coords is not None and p_coords.shape[0] == mob_ca.shape[0]:
                            tm_parent = float(fg.tm_score(mob_ca, p_coords[:, 1, :].astype(np.float64))["tm_score"])
                    folded_rows.append(
                        {
                            "seq_id": rec["seq_id"],
                            "backbone_id": bb_id,
                            "rung": rec["rung"],
                            "sequence_sha256": sha,
                            "status": "ok",
                            "tm_score_to_parent": tm_parent,
                            **s_met,
                        }
                    )
                elif bench == "exp4_2":
                    dom = meta["domain"]
                    fm = exp4_2_fms[dom]
                    wt_seq = wt_exp4_2_seqs[dom]
                    tm_val = float(fg.tm_score(mob_ca, wt_exp4_2_ca[dom])["tm_score"])
                    e_full, e_fields, e_couplings = exp42.compute_potts_energies(fm, seq)
                    wt_id = ygen.sequence_identity_exact(seq, wt_seq)
                    msa_list = exp4_2_msa_seqs.get(dom) or [wt_seq]
                    msa_max = max(
                        (ygen.sequence_identity_exact(seq, m) for m in msa_list if len(m) == len(seq)),
                        default=wt_id,
                    )
                    folded_rows.append(
                        {
                            **rec,
                            "schema_version": exp42.FOLDED_RECORD_SCHEMA,
                            "status": "ok",
                            "tm_score": tm_val,
                            "plddt_mean": s_met["mean_ca_plddt"],
                            "mean_ca_plddt": s_met["mean_ca_plddt"],
                            "ptm": s_met["ptm"],
                            "mean_pae_angstrom": s_met["mean_pae_angstrom"],
                            "potts_energy_full": e_full,
                            "potts_energy_fields": e_fields,
                            "potts_energy_couplings": e_couplings,
                            "wt_identity_pct": wt_id,
                            "msa_max_identity_pct": msa_max,
                            "shannon_H_bits": exp42.compute_shannon_entropy_bits(seq),
                        }
                    )
                elif bench == "r4_cont":
                    tid = str(rec["target_id"])
                    tm_val = (
                        r4c.structural_tm_score(mob_ca, r4_wt_ca[tid])
                        if tid in r4_wt_ca
                        else None
                    )
                    folded_rows.append(
                        {
                            "schema_version": r4c.FOLDED_SCHEMA,
                            "attempt_id": rec["attempt_id"],
                            "model": rec["model"],
                            "target_id": tid,
                            "cluster": rec["cluster"],
                            "condition": rec["condition"],
                            "sequence_sha256": sha,
                            "residues": len(seq),
                            "plddt_mean": s_met["mean_ca_plddt"],
                            "ptm": s_met["ptm"],
                            "mean_pae_angstrom": s_met["mean_pae_angstrom"],
                            "tm_score_to_target_wt": tm_val,
                        }
                    )

            ygen.atomic_write_jsonl(fp, folded_rows)
            if bench == "exp4_1" and refolded_bb_map:
                npz_dir = work_dir / "exp4_1" / "refolded_coords"
                npz_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(npz_dir / f"{stem}.npz", **refolded_bb_map)

            print(
                f"[fold shard {args.shard_index}/{args.num_shards}] {bench}/{stem}: "
                f"{len(folded_rows)} seqs in {time.monotonic() - t0:.1f}s (cache={len(fold_cache)})",
                flush=True,
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Stage 6: Experiment 4.3 Native Self-Likelihood Scoring
# --------------------------------------------------------------------------- #


def run_exp4_3_stage(args: argparse.Namespace, pool: _AdapterPool) -> None:
    allowed_arms, allowed_modes = _parse_model_filter(args.models)
    work_dir = _resolve_path(args.work_dir)
    wt_bb_dir = _resolve_path(args.wt_backbones_dir)
    scores_dir = work_dir / "exp4_3" / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        s
        for s in ygen.EXP4_1_NEW_SPECS
        if s.arm_id in allowed_arms and (allowed_modes is None or s.mode in allowed_modes)
    ]
    assigned = _select_shard(specs, args.shard_index, args.num_shards)
    if not assigned:
        print(f"[exp4_3 shard {args.shard_index}/{args.num_shards}] nothing assigned", flush=True)
        return

    exp4_1_templates = {
        t["seq_id"]: t["sequence"]
        for t in ygen.select_exp4_1_reference_targets(REPO_ROOT, n_targets=400)
    }

    for spec in assigned:
        gen_p = work_dir / "exp4_1" / "generated" / f"{spec.file_slug}.jsonl"
        fold_p = work_dir / "exp4_1" / "folded" / f"{spec.file_slug}.jsonl"
        npz_p = work_dir / "exp4_1" / "refolded_coords" / f"{spec.file_slug}.npz"
        if not (gen_p.is_file() and fold_p.is_file()):
            print(f"[exp4_3] {spec.entry_key}: missing generated/folded files, skipping", flush=True)
            continue

        gen_by_id = {str(r["seq_id"]): r for r in ygen.read_jsonl(gen_p)}
        fold_rows = ygen.read_jsonl(fold_p)
        refold_npz = np.load(npz_p, allow_pickle=False) if npz_p.is_file() else None

        if spec.backend == "dplm2":
            # Score with DPLM-2 exact_pll (seq_only for unconditional; struct_conditioned on refolded backbone for struct/cogen).
            score_modes = [("seq_only", "none", spec.entry_key)]
            if spec.mode in ("struct_conditioned", "co_generation") and refold_npz is not None:
                score_modes.append(
                    ("struct_conditioned", "esmfold2_refolded", f"{spec.entry_key}__refolded_ll")
                )
            for smode, bb_src, pool_id in score_modes:
                out_p = scores_dir / f"{ygen.entry_file_slug(pool_id)}.jsonl"
                if args.skip_existing and not args.overwrite and _line_count(out_p) >= len(fold_rows):
                    continue
                t0 = time.monotonic()
                adapter = pool.get_dplm2(spec.arm_id, smode)
                out_rows: list[dict[str, Any]] = []
                for frow in fold_rows:
                    sid = str(frow["seq_id"])
                    seq = str(frow["sequence"]).strip().upper()
                    coords = (
                        np.asarray(refold_npz[sid], dtype=np.float32)
                        if (smode == "struct_conditioned" and refold_npz is not None and sid in refold_npz.files)
                        else None
                    )
                    pv = adapter.exact_pll(seq, coords=coords, mode=smode, batch_size=32)
                    nll_res = float(pv.nll_sum) / max(1, len(seq))
                    out_rows.append(
                        {
                            "pool_id": pool_id,
                            "arm_id": spec.arm_id,
                            "conditioning_regime": spec.conditioning_regime,
                            "scoring_backbone_source": bb_src,
                            "seq_id": sid,
                            "sequence": seq,
                            "sequence_sha256": str(frow["sequence_sha256"]),
                            "plddt": float(frow["mean_ca_plddt"]),
                            "ptm": float(frow["ptm"]),
                            "mean_nll_per_scored_token": nll_res,
                            "mean_nll_per_residue": nll_res,
                            "mean_pll_per_residue": float(-nll_res),
                        }
                    )
                ygen.atomic_write_jsonl(out_p, out_rows)
                print(
                    f"[exp4_3 shard {args.shard_index}/{args.num_shards}] {pool_id}: "
                    f"{len(out_rows)} scored in {time.monotonic() - t0:.1f}s",
                    flush=True,
                )
        elif spec.backend == "dima":
            # Score with DiMA 8-fold stride-masked denoising PLL + latent denoising MSE diagnostic.
            out_p = scores_dir / f"{ygen.entry_file_slug(spec.entry_key)}.jsonl"
            mse_pool_id = f"{spec.entry_key}__latent_mse"
            mse_p = scores_dir / f"{ygen.entry_file_slug(mse_pool_id)}.jsonl"
            if (
                args.skip_existing
                and not args.overwrite
                and _line_count(out_p) >= len(fold_rows)
                and _line_count(mse_p) >= len(fold_rows)
            ):
                continue
            t0 = time.monotonic()
            adapter = pool.get_dima(spec.arm_id, load_encoder=True)
            out_rows = []
            mse_rows = []
            for frow in fold_rows:
                sid = str(frow["seq_id"])
                seq = str(frow["sequence"]).strip().upper()
                pv = adapter.pll_8fold(seq)
                nll_res = float(pv.nll_sum) / max(1, len(seq))
                mse_val = float(adapter.latent_denoising_mse(seq, seed=args.base_seed))
                base_row = {
                    "arm_id": spec.arm_id,
                    "conditioning_regime": spec.conditioning_regime,
                    "scoring_backbone_source": "none",
                    "seq_id": sid,
                    "sequence": seq,
                    "sequence_sha256": str(frow["sequence_sha256"]),
                    "plddt": float(frow["mean_ca_plddt"]),
                    "ptm": float(frow["ptm"]),
                    "latent_denoising_mse": mse_val,
                }
                out_rows.append(
                    {
                        **base_row,
                        "pool_id": spec.entry_key,
                        "mean_nll_per_scored_token": nll_res,
                        "mean_nll_per_residue": nll_res,
                        "mean_pll_per_residue": float(-nll_res),
                    }
                )
                mse_rows.append(
                    {
                        **base_row,
                        "pool_id": mse_pool_id,
                        "mean_nll_per_scored_token": mse_val,
                        "mean_nll_per_residue": mse_val,
                        "mean_pll_per_residue": float(-mse_val),
                    }
                )
            ygen.atomic_write_jsonl(out_p, out_rows)
            ygen.atomic_write_jsonl(mse_p, mse_rows)
            print(
                f"[exp4_3 shard {args.shard_index}/{args.num_shards}] {spec.entry_key} (+latent_mse): "
                f"{len(out_rows)} scored in {time.monotonic() - t0:.1f}s",
                flush=True,
            )
        else:
            # ProteinMPNN: score BOTH on ESMFold2-refolded backbone (primary, zero template leakage)
            # AND on input template backbone (template-conditioned companion).
            adapter = pool.get_proteinmpnn(spec.arm_id)
            variants = [
                ("esmfold2_refolded", spec.entry_key),
                ("input_template", f"{spec.entry_key}__template_ll"),
            ]
            for bb_src, pool_id in variants:
                out_p = scores_dir / f"{ygen.entry_file_slug(pool_id)}.jsonl"
                if args.skip_existing and not args.overwrite and _line_count(out_p) >= len(fold_rows):
                    continue
                t0 = time.monotonic()
                out_rows = []
                for frow in fold_rows:
                    sid = str(frow["seq_id"])
                    seq = str(frow["sequence"]).strip().upper()
                    if bb_src == "esmfold2_refolded":
                        if refold_npz is None or sid not in refold_npz.files:
                            continue
                        coords = np.asarray(refold_npz[sid], dtype=np.float32)
                    else:
                        tid = (gen_by_id.get(sid) or {}).get("template_id")
                        if not tid or tid not in exp4_1_templates:
                            continue
                        coords = _load_bb_coords(wt_bb_dir, exp4_1_templates[tid])
                        if coords is None:
                            continue
                    ll_arr = adapter.exact_sequence_ll_batch(
                        [seq], coords, n_permutations=4, seed=args.base_seed
                    )
                    mean_ll = float(ll_arr[0]) / max(1, len(seq))
                    nll_res = float(-mean_ll)
                    out_rows.append(
                        {
                            "pool_id": pool_id,
                            "arm_id": spec.arm_id,
                            "conditioning_regime": spec.conditioning_regime,
                            "scoring_backbone_source": bb_src,
                            "seq_id": sid,
                            "sequence": seq,
                            "sequence_sha256": str(frow["sequence_sha256"]),
                            "plddt": float(frow["mean_ca_plddt"]),
                            "ptm": float(frow["ptm"]),
                            "mean_nll_per_scored_token": nll_res,
                            "mean_nll_per_residue": nll_res,
                            "mean_pll_per_residue": mean_ll,
                        }
                    )
                if out_rows:
                    ygen.atomic_write_jsonl(out_p, out_rows)
                print(
                    f"[exp4_3 shard {args.shard_index}/{args.num_shards}] {pool_id}: "
                    f"{len(out_rows)} scored in {time.monotonic() - t0:.1f}s",
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage",
        choices=(*ALL_STAGES, "all"),
        required=True,
        help="Execution stage (or 'all' to run generate -> fold -> exp4_3 sequentially).",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="all",
        help="Model subset: all | dplm2 | dplm2_seq | dplm2_struct | proteinmpnn | dima | <csv arm_ids>",
    )
    ap.add_argument(
        "--fold-benchmarks",
        type=str,
        default="exp4_1,r5a,exp4_2,r4_cont",
        help="Comma-separated benchmarks to fold during --stage fold",
    )
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    ap.add_argument("--dima-root", type=Path, default=None)
    ap.add_argument("--esmfold2-dir", type=Path, default=None)
    ap.add_argument("--wt-backbones-dir", type=Path, default=ygen.WT_BACKBONES_REL)
    ap.add_argument("--work-dir", type=Path, default=ygen.WORK_DIR_REL)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--dplm2-steps", type=int, default=64)
    ap.add_argument("--dplm2-batch-size", type=int, default=16)
    ap.add_argument("--dima-steps", type=int, default=250)
    ap.add_argument("--dima-batch-size", type=int, default=16)
    ap.add_argument("--exp4-1-pool-size", type=int, default=ygen.EXP4_1_POOL_SIZE)
    ap.add_argument("--exp4-2-pool-size", type=int, default=ygen.EXP4_2_POOL_SIZE)
    ap.add_argument("--r4-attempts-per-cell", type=int, default=r4c.ATTEMPTS_PER_CELL)
    ap.add_argument("--diffusion-samples", type=int, default=32)
    ap.add_argument("--base-seed", type=int, default=ygen.DEFAULT_SEED)
    ap.add_argument("--max-items", type=int, default=None, help="Cap targets/domains for smoke tests")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index {args.shard_index} outside [0, {args.num_shards})")

    stages = list(ALL_STAGES) if args.stage == "all" else [args.stage]
    t0 = time.monotonic()
    pool = _AdapterPool(
        model_root=args.model_root,
        device=args.device,
        dtype_override=args.dtype,
        dima_root=args.dima_root,
    )
    try:
        for st in stages:
            if st == "exp4_1":
                run_exp4_1_stage(args, pool)
            elif st == "r5a":
                run_r5a_stage(args, pool)
            elif st == "exp4_2":
                run_exp4_2_stage(args, pool)
            elif st == "r4_cont":
                run_r4_cont_stage(args, pool)
            elif st == "fold":
                pool.release_all()
                run_fold_stage(args)
            elif st == "exp4_3":
                run_exp4_3_stage(args, pool)
    finally:
        pool.release_all()

    receipt_dir = _resolve_path(args.work_dir) / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "stage": args.stage,
        "models": args.models,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "elapsed_seconds": time.monotonic() - t0,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (receipt_dir / f"receipt_{args.stage}_s{args.shard_index:02d}_of_{args.num_shards:02d}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
