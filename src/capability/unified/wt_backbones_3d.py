"""Unified leakage-free 3D wild-type & context backbone cache for ProteinMPNN and DPLM-2.

Inventories 100% of unique wild-type, template, parent, homolog, and unrelated
context sequences required across all Results 1-5 benchmarks:

* **R1**: All 217 ProteinGym DMS WTs (``data/proteingym_raw/DMS_substitutions.csv``)
  plus 174 statistical-ladder WTs.
* **R2**: All 101 MegaScale stability WTs and 428 Domainome abundance WTs.
* **R3**: All 64 double-mutant epistasis WTs and all admitted E01 contact WTs.
* **Result 4**: Experiment 4.1 natural reference fragments (400), R5a local
  infilling parent backbones (16), and Experiment 4.2 sealed domains (8/12).
* **Result 5**: Main Table (1,000 units: 800 retained + 200 high-local-overlap
  targets, filler, and budget-1024 homolog/unrelated contexts), Downstream 1
  (``rescue_30_80`` + ``e09_five_band`` WTs and planned contexts), and
  Downstream 2 (22 continuation WTs and planned contexts).

Every entry is strictly content-addressed by ``sha256(uppercase_sequence)`` and
stores ONLY wild-type / template / context backbones (never mutant sequences),
preventing any structural data leakage.

Canonical ``.npz`` Backbone Contract
------------------------------------
Each cached ``backbones/<sha[:2]>/<sha>.npz`` stores:

* ``coords``: ``float32`` array of shape ``(L, 4, 3)`` in exact
  ``[N, CA, C, O]`` atom order (Angstroms; ``NaN`` only for genuinely missing
  experimental atoms). Directly compatible with both ``ProteinMPNN``
  (full-backbone ``[N, CA, C, O]`` and CA-only ``coords[:, 1, :]``) and
  ``DPLM-2 StructTokenizerEncoder`` (``coords[:, :3, :]`` = ``[N, CA, C]``).
* ``res_mask``: ``bool`` array of shape ``(L,)`` (``True`` where ``N``, ``CA``,
  ``C`` are all finite).
* ``ca_mask``: ``bool`` array of shape ``(L,)`` (``True`` where ``CA`` is finite).
* ``plddt``: ``float32`` array of shape ``(L,)`` (per-residue pLDDT in ``[0, 100]``
  for AlphaFold / ESMFold2 structures; ``NaN`` for experimental structures).
* ``sequence``: exact canonical amino-acid string of length ``L``.
* ``source_kind``: provenance tag (``exp_freeze_pdb_exact``, ``afdb_v6_exact``,
  ``afdb_v6_substring``, ``exp_mmcif_exact``, ``exp_mmcif_substring``,
  ``cath_s40_exact``, ``cath_s40_substring``, ``esmfold2_direct``,
  ``esmfold2_kabsch_stitched``).

Unmatched sequences are emitted into ``pending_wt_esmfold2.jsonl`` sorted by
priority tier (``tier_1_r1_r4_wt``, ``tier_2_r5_targets_and_primary_ctx``,
``tier_3_r5_extra_contexts``), with sequences longer than 1,024 residues
automatically split into overlapping 1,024-aa windows (``overlap = 256`` aa)
and seamlessly reassembled via exact proper-rotation Kabsch superposition
(``stitch_windowed_backbones``).
"""

from __future__ import annotations

import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import shlex
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.capability.context import homology_context as H
from src.capability.core.io import sha256_file
from src.capability.unified import r4_continuation_yardstick as r4c
from src.capability.unified import r4_scoring_yardstick as r4s
from src.capability.unified import r5_dms_context_yardstick as r5d
from src.capability.unified.roster import REPO_ROOT, UNIFIED_OUTPUT_DIR

SCHEMA_MANIFEST = "wt_backbones_3d_manifest_v1"
SCHEMA_PENDING = "wt_backbones_3d_pending_esmfold2_v1"

DEFAULT_CACHE_DIR = UNIFIED_OUTPUT_DIR / "wt_backbones_3d"
ATOM_ORDER: tuple[str, str, str, str] = ("N", "CA", "C", "O")
ATOM_INDEX: dict[str, int] = {name: i for i, name in enumerate(ATOM_ORDER)}

ESMFOLD_MAX_LEN = 1024
WINDOW_OVERLAP = 256
MIN_SEED_KMER = 12

#: Standard 3-letter to 1-letter amino acid code map (including common modified residues).
AA3_TO_1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}

#: Priority Tier 1: core R1/R2/R3/R4 WTs + R5 D1/D2 WTs (immediately unblocks Results 1-4).
TIER1_R1_R4_TAGS: frozenset[str] = frozenset(
    {
        "r1_proteingym_217",
        "r1_ladder_support",
        "r2_megascale_101",
        "r2_domainome_428",
        "r3_epistasis_64",
        "r3_e01_contacts",
        "r4_exp4_1_natural_400",
        "r4_r5a_backbones_16",
        "r4_exp4_2_domains_12",
        "r5_d1_rescue_wt",
        "r5_d1_e09_wt",
        "r5_d2_continuation_wt",
    }
)

#: Priority Tier 2: R5 Main Table targets/filler + primary 1st homolog/unrelated contexts.
TIER2_R5_PRIMARY_TAGS: frozenset[str] = frozenset(
    {
        "r5_main_target",
        "r5_main_filler",
        "r5_main_homolog_1st",
        "r5_main_unrelated_1st",
        "r5_d1_rescue_homolog",
        "r5_d1_rescue_unrelated",
        "r5_d1_e09_ctx_1st",
        "r5_d2_continuation_ctx_1st",
    }
)

TIER_ORDER: tuple[str, str, str] = (
    "tier_1_r1_r4_wt",
    "tier_2_r5_targets_and_primary_ctx",
    "tier_3_r5_extra_contexts",
)

DEFAULT_FREEZE_PDB_DIR = Path("results/R6/e20_wild_type_references_20261013/freeze")
DEFAULT_AFDB_DIR = Path("data/alphafold")
DEFAULT_EXP_CONTACT_CIF_DIR = Path(
    "archive/logs/shared/gate_structure_contact_20260924/structures"
)
DEFAULT_BINDING_CIF_DIR = Path("data/phenotype_followups_20261007/binding-structures")
DEFAULT_CATH_S40_DIR = Path("/Data/public/datasets/cath-s40")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seq_sha256(sequence: str) -> str:
    """Deterministic SHA-256 hex digest of a canonical uppercase sequence."""
    return hashlib.sha256(sequence.strip().upper().encode("utf-8")).hexdigest()


def backbone_npz_relpath(digest: str) -> Path:
    """Content-addressed relative path ``backbones/<sha[:2]>/<sha>.npz``."""
    return Path("backbones") / digest[:2] / f"{digest}.npz"


def atomic_write_json_with_sha256(path: Path, payload: Any) -> str:
    """Atomically write sorted JSON and a ``sha256sum -c`` compatible ``.sha256`` sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    digest = sha256_file(path)
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def atomic_write_jsonl_with_sha256(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Atomically write JSONL and its ``.sha256`` sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    digest = sha256_file(path)
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def save_backbone_npz(
    path: Path,
    *,
    sequence: str,
    coords: np.ndarray,
    plddt: np.ndarray | None,
    source_kind: str,
    source_path: str = "",
    dplm2_struct_tokens: np.ndarray | None = None,
    dplm2_lfq_bits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate and atomically save one ``(L, 4, 3)`` ``[N, CA, C, O]`` backbone ``.npz``."""
    seq = sequence.strip().upper()
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 3 or arr.shape != (len(seq), 4, 3):
        raise ValueError(
            f"Backbone coords shape {arr.shape} disagrees with sequence length {len(seq)} (expected (L, 4, 3))"
        )
    res_mask = np.isfinite(arr[:, :3, :]).all(axis=(1, 2))
    ca_mask = np.isfinite(arr[:, 1, :]).all(axis=1)
    if plddt is None:
        plddt_arr = np.full(len(seq), np.nan, dtype=np.float32)
    else:
        plddt_arr = np.asarray(plddt, dtype=np.float32)
        if plddt_arr.shape != (len(seq),):
            raise ValueError(f"pLDDT shape {plddt_arr.shape} disagrees with length {len(seq)}")

    payload: dict[str, Any] = {
        "coords": arr,
        "res_mask": res_mask.astype(np.bool_),
        "ca_mask": ca_mask.astype(np.bool_),
        "plddt": plddt_arr,
        "sequence": np.asarray(seq),
        "atom_order": np.asarray(ATOM_ORDER),
        "source_kind": np.asarray(source_kind),
        "source_path": np.asarray(source_path),
    }
    if dplm2_struct_tokens is not None:
        toks = np.asarray(dplm2_struct_tokens, dtype=np.int32)
        if toks.shape != (len(seq),):
            raise ValueError(f"dplm2_struct_tokens shape {toks.shape} != ({len(seq)},)")
        payload["dplm2_struct_tokens"] = toks
    if dplm2_lfq_bits is not None:
        bits = np.asarray(dplm2_lfq_bits, dtype=np.int8)
        if bits.shape[0] != len(seq):
            raise ValueError(f"dplm2_lfq_bits shape {bits.shape} disagrees with length {len(seq)}")
        payload["dplm2_lfq_bits"] = bits

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    valid_plddt = plddt_arr[np.isfinite(plddt_arr)]
    return {
        "length": len(seq),
        "n_valid_ncac": int(res_mask.sum()),
        "n_valid_ca": int(ca_mask.sum()),
        "coverage_fraction": float(res_mask.mean()) if len(seq) else 0.0,
        "mean_plddt": float(valid_plddt.mean()) if len(valid_plddt) else None,
    }


def load_backbone_npz(cache_dir: Path, seq_or_sha: str) -> dict[str, Any]:
    """Load a cached 3D backbone by raw sequence or 64-hex SHA-256 digest."""
    key = seq_or_sha.strip()
    digest = key.lower() if len(key) == 64 and all(c in "0123456789abcdef" for c in key.lower()) else seq_sha256(key)
    path = Path(cache_dir) / backbone_npz_relpath(digest)
    if not path.is_file():
        raise FileNotFoundError(f"Backbone {digest} not found in {cache_dir}")
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, Any] = {
            "seq_sha256": digest,
            "sequence": str(data["sequence"].item()),
            "coords": np.asarray(data["coords"], dtype=np.float32),
            "res_mask": np.asarray(data["res_mask"], dtype=np.bool_),
            "ca_mask": np.asarray(data["ca_mask"], dtype=np.bool_),
            "plddt": np.asarray(data["plddt"], dtype=np.float32),
            "source_kind": str(data["source_kind"].item()),
            "source_path": str(data["source_path"].item()),
        }
        if "dplm2_struct_tokens" in data:
            out["dplm2_struct_tokens"] = np.asarray(data["dplm2_struct_tokens"], dtype=np.int32)
        if "dplm2_lfq_bits" in data:
            out["dplm2_lfq_bits"] = np.asarray(data["dplm2_lfq_bits"], dtype=np.int8)
    return out


# --------------------------------------------------------------------------- #
# Pure-Python PDB / mmCIF Parsers (exact [N, CA, C, O] coordinates)
# --------------------------------------------------------------------------- #


def _open_text_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def scan_pdb_ca_sequence(path: Path) -> str:
    """Ultra-fast CA-only sequence scan of model 1 chain A of a PDB / ``.pdb.gz`` file."""
    residues: list[str] = []
    seen_res: set[tuple[str, str, str]] = set()
    first_chain: str | None = None
    with _open_text_maybe_gz(path) as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  "):
                continue
            if line[12:16] != " CA ":
                continue
            alt = line[16]
            if alt not in (" ", "A", "1"):
                continue
            chain = line[21]
            if first_chain is None:
                first_chain = chain
            elif chain != first_chain:
                continue
            res_key = (chain, line[22:26], line[26])
            if res_key in seen_res:
                continue
            seen_res.add(res_key)
            residues.append(AA3_TO_1.get(line[17:20].strip(), "X"))
    return "".join(residues)


def parse_pdb_backbone_lines(
    lines: Iterable[str],
    chain: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse exact ``(sequence, coords[L,4,3], bfactor_or_plddt[L])`` from PDB text lines."""
    res_order: list[tuple[str, str, str]] = []
    res_aa: dict[tuple[str, str, str], str] = {}
    res_atoms: dict[tuple[str, str, str], dict[str, tuple[float, float, float]]] = {}
    res_bfac: dict[tuple[str, str, str], float] = {}
    target_chain = chain

    for line in lines:
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ATOM_INDEX:
            continue
        alt = line[16]
        if alt not in (" ", "A", "1"):
            continue
        ch = line[21]
        if target_chain is None:
            target_chain = ch
        elif ch != target_chain:
            continue
        res_key = (ch, line[22:26], line[26])
        if res_key not in res_atoms:
            res_order.append(res_key)
            res_aa[res_key] = AA3_TO_1.get(line[17:20].strip(), "X")
            res_atoms[res_key] = {}
        if atom_name not in res_atoms[res_key]:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            res_atoms[res_key][atom_name] = (x, y, z)
            if atom_name == "CA":
                try:
                    res_bfac[res_key] = float(line[60:66])
                except ValueError:
                    res_bfac[res_key] = float("nan")

    n_res = len(res_order)
    coords = np.full((n_res, 4, 3), np.nan, dtype=np.float32)
    plddt = np.full(n_res, np.nan, dtype=np.float32)
    seq_chars: list[str] = []
    for i, rk in enumerate(res_order):
        seq_chars.append(res_aa[rk])
        atoms = res_atoms[rk]
        for aname, aidx in ATOM_INDEX.items():
            if aname in atoms:
                coords[i, aidx] = atoms[aname]
        if rk in res_bfac:
            plddt[i] = res_bfac[rk]
    return "".join(seq_chars), coords, plddt


def parse_pdb_backbone(path: Path, chain: str | None = None) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse exact ``(sequence, coords[L,4,3], plddt[L])`` from model 1 of a PDB / ``.pdb.gz`` file."""
    with _open_text_maybe_gz(path) as handle:
        return parse_pdb_backbone_lines(handle, chain=chain)


def parse_mmcif_chains(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """Parse model-1 per-chain ``(sequence, coords[L,4,3])`` from an mmCIF / ``.cif.gz`` file."""
    chains_order: dict[str, list[tuple[int | str, str]]] = defaultdict(list)
    chains_aa: dict[str, dict[tuple[int | str, str], str]] = defaultdict(dict)
    chains_atoms: dict[str, dict[tuple[int | str, str], dict[str, tuple[float, float, float]]]] = (
        defaultdict(dict)
    )

    with _open_text_maybe_gz(path) as handle:
        in_atom_loop = False
        cols: list[str] = []
        col_idx: dict[str, int] = {}
        first_model: str | None = None

        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                if in_atom_loop and cols:
                    in_atom_loop = False
                    cols = []
                    col_idx = {}
                continue
            if line == "loop_":
                in_atom_loop = False
                cols = []
                col_idx = {}
                continue
            if line.startswith("_atom_site."):
                in_atom_loop = True
                cols.append(line.split(".", 1)[1])
                continue
            if line.startswith("_"):
                if in_atom_loop:
                    in_atom_loop = False
                    cols = []
                    col_idx = {}
                continue
            if not in_atom_loop or not cols:
                continue
            if not col_idx:
                col_idx = {name: i for i, name in enumerate(cols)}
                required = {"group_PDB", "label_atom_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z"}
                if not required.issubset(col_idx):
                    in_atom_loop = False
                    cols = []
                    col_idx = {}
                    continue

            parts = shlex.split(line) if ("'" in line or '"' in line) else line.split()
            if len(parts) < len(cols):
                continue
            if parts[col_idx["group_PDB"]] != "ATOM":
                continue
            atom_name = parts[col_idx["label_atom_id"]]
            if atom_name not in ATOM_INDEX:
                continue
            if "pdbx_PDB_model_num" in col_idx:
                mnum = parts[col_idx["pdbx_PDB_model_num"]]
                if first_model is None:
                    first_model = mnum
                elif mnum != first_model:
                    continue
            if "label_alt_id" in col_idx:
                alt = parts[col_idx["label_alt_id"]]
                if alt not in (".", "?", "A", "1"):
                    continue

            ch_col = "auth_asym_id" if "auth_asym_id" in col_idx else "label_asym_id"
            seq_col = "label_seq_id" if "label_seq_id" in col_idx else "auth_seq_id"
            ch = parts[col_idx[ch_col]]
            seq_s = parts[col_idx[seq_col]]
            if seq_s in (".", "?"):
                continue
            ins = parts[col_idx["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in col_idx else "."
            if ins in (".", "?"):
                ins = ""
            res_key: tuple[int | str, str] = (int(seq_s) if seq_s.lstrip("-").isdigit() else seq_s, ins)
            if res_key not in chains_atoms[ch]:
                chains_order[ch].append(res_key)
                chains_aa[ch][res_key] = AA3_TO_1.get(parts[col_idx["label_comp_id"]], "X")
                chains_atoms[ch][res_key] = {}
            if atom_name not in chains_atoms[ch][res_key]:
                try:
                    xyz = (
                        float(parts[col_idx["Cartn_x"]]),
                        float(parts[col_idx["Cartn_y"]]),
                        float(parts[col_idx["Cartn_z"]]),
                    )
                except ValueError:
                    continue
                chains_atoms[ch][res_key][atom_name] = xyz

    result: dict[str, tuple[str, np.ndarray]] = {}
    for ch, rkeys in chains_order.items():
        n_res = len(rkeys)
        if n_res == 0:
            continue
        coords = np.full((n_res, 4, 3), np.nan, dtype=np.float32)
        seq_chars = []
        for i, rk in enumerate(rkeys):
            seq_chars.append(chains_aa[ch][rk])
            atoms = chains_atoms[ch][rk]
            for aname, aidx in ATOM_INDEX.items():
                if aname in atoms:
                    coords[i, aidx] = atoms[aname]
        result[ch] = ("".join(seq_chars), coords)
    return result


# --------------------------------------------------------------------------- #
# Window Planning & Exact Kabsch Superposition Stitching (> 1,024 aa)
# --------------------------------------------------------------------------- #


def plan_overlapping_windows(
    sequence: str,
    *,
    max_len: int = ESMFOLD_MAX_LEN,
    overlap: int = WINDOW_OVERLAP,
) -> list[tuple[int, int]]:
    """Split a long sequence (``L > max_len``) into overlapping ``[start, end)`` windows."""
    length = len(sequence)
    if length <= max_len:
        return [(0, length)]
    if overlap <= 0 or overlap >= max_len:
        raise ValueError(f"Invalid overlap {overlap} for max_len {max_len}")
    stride = max_len - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + max_len, length)
        if end == length:
            start = max(0, length - max_len)
            windows.append((start, length))
            break
        windows.append((start, end))
        start += stride
    return windows


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute exact proper-rotation Kabsch transform ``(R, t, rmsd)`` minimizing ``||mobile @ R + t - target||``.

    Guarantees ``det(R) == +1`` (proper rotation, never a mirror reflection).
    """
    p = np.asarray(mobile, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    if p.ndim != 2 or p.shape != q.shape or p.shape[1] != 3 or len(p) < 3:
        raise ValueError(f"Kabsch requires matching (N>=3, 3) point clouds, got {p.shape} vs {q.shape}")
    p_mean = p.mean(axis=0)
    q_mean = q.mean(axis=0)
    pc = p - p_mean
    qc = q - q_mean
    cov = pc.T @ qc
    u, _s, vt = np.linalg.svd(cov)
    d = np.linalg.det(u @ vt)
    sign = np.diag([1.0, 1.0, -1.0 if d < 0.0 else 1.0])
    rot = u @ sign @ vt
    trans = q_mean - p_mean @ rot
    aligned = p @ rot + trans
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=1))))
    return rot, trans, rmsd


def stitch_windowed_backbones(
    windows: Sequence[Mapping[str, Any]],
    parent_length: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Progressively Kabsch-align overlapping ``(L_w, 4, 3)`` windows into one ``(L, 4, 3)`` backbone.

    Each element of ``windows`` requires ``window_start``, ``window_end``,
    ``coords`` ``(L_w, 4, 3)``, and optional ``plddt`` ``(L_w,)``.
    Overlap regions are smoothly blended with linear ramp weights so backbone
    geometry transitions without seam discontinuities.
    """
    ordered = sorted(windows, key=lambda w: (int(w["window_start"]), int(w["window_end"])))
    if not ordered:
        raise ValueError("Cannot stitch empty window list")
    if int(ordered[0]["window_start"]) != 0 or int(ordered[-1]["window_end"]) != parent_length:
        raise ValueError(
            f"Windows do not span [0, {parent_length}): "
            f"start={ordered[0]['window_start']}, end={ordered[-1]['window_end']}"
        )

    out_coords = np.full((parent_length, 4, 3), np.nan, dtype=np.float64)
    out_plddt = np.full(parent_length, np.nan, dtype=np.float64)
    filled = np.zeros(parent_length, dtype=np.bool_)

    w0 = ordered[0]
    s0, e0 = int(w0["window_start"]), int(w0["window_end"])
    c0 = np.asarray(w0["coords"], dtype=np.float64)
    if c0.shape != (e0 - s0, 4, 3):
        raise ValueError(f"Window 0 coords shape {c0.shape} != ({e0 - s0}, 4, 3)")
    out_coords[s0:e0] = c0
    if w0.get("plddt") is not None:
        out_plddt[s0:e0] = np.asarray(w0["plddt"], dtype=np.float64)
    filled[s0:e0] = True

    overlap_rmsds: list[float] = []
    for win in ordered[1:]:
        s, e = int(win["window_start"]), int(win["window_end"])
        cw = np.asarray(win["coords"], dtype=np.float64)
        if cw.shape != (e - s, 4, 3):
            raise ValueError(f"Window [{s},{e}) coords shape {cw.shape} != ({e - s}, 4, 3)")
        pw = (
            np.asarray(win["plddt"], dtype=np.float64)
            if win.get("plddt") is not None
            else np.full(e - s, np.nan, dtype=np.float64)
        )

        # Overlap positions where BOTH assembled backbone and current window have finite CA atoms
        ov_global = np.arange(s, e)
        ov_mask = filled[s:e] & np.isfinite(out_coords[s:e, 1, :]).all(axis=1) & np.isfinite(cw[:, 1, :]).all(axis=1)
        if int(ov_mask.sum()) < 3:
            raise ValueError(f"Window [{s},{e}) has fewer than 3 finite CA overlap residues ({int(ov_mask.sum())})")

        mob_ca = cw[ov_mask, 1, :]
        tgt_ca = out_coords[ov_global[ov_mask], 1, :]
        rot, trans, rmsd = kabsch_align(mob_ca, tgt_ca)
        overlap_rmsds.append(rmsd)

        aligned_cw = np.einsum("lai,ij->laj", cw, rot) + trans[None, None, :]

        # Linear ramp blending across the overlap span; direct copy beyond overlap
        ov_indices = np.flatnonzero(filled[s:e])
        n_ov = len(ov_indices)
        if n_ov > 0:
            ramp = np.linspace(0.0, 1.0, n_ov + 2, dtype=np.float64)[1:-1]
            for k_local, alpha in zip(ov_indices, ramp):
                g_idx = s + int(k_local)
                old_ok = np.isfinite(out_coords[g_idx]).all()
                new_ok = np.isfinite(aligned_cw[k_local]).all()
                if old_ok and new_ok:
                    out_coords[g_idx] = (1.0 - alpha) * out_coords[g_idx] + alpha * aligned_cw[k_local]
                elif new_ok:
                    out_coords[g_idx] = aligned_cw[k_local]
                if np.isfinite(out_plddt[g_idx]) and np.isfinite(pw[k_local]):
                    out_plddt[g_idx] = (1.0 - alpha) * out_plddt[g_idx] + alpha * pw[k_local]
                elif np.isfinite(pw[k_local]):
                    out_plddt[g_idx] = pw[k_local]

        new_indices = np.flatnonzero(~filled[s:e])
        for k_local in new_indices:
            g_idx = s + int(k_local)
            out_coords[g_idx] = aligned_cw[k_local]
            out_plddt[g_idx] = pw[k_local]
            filled[g_idx] = True

    return out_coords.astype(np.float32), out_plddt.astype(np.float32), overlap_rmsds


# --------------------------------------------------------------------------- #
# Complete 5-Benchmark WT / Context Sequence Inventory
# --------------------------------------------------------------------------- #


@dataclass
class SequenceEntry:
    """One deduplicated wild-type / template / context sequence across all benchmarks."""

    seq_sha256: str
    sequence: str
    tags: set[str]
    example_ids: list[str]

    @property
    def priority_tier(self) -> str:
        if self.tags & TIER1_R1_R4_TAGS:
            return "tier_1_r1_r4_wt"
        if self.tags & TIER2_R5_PRIMARY_TAGS:
            return "tier_2_r5_targets_and_primary_ctx"
        return "tier_3_r5_extra_contexts"


def collect_all_benchmark_sequences(
    *,
    root: Path | None = None,
    include_r5_extra_b1024_contexts: bool = True,
) -> dict[str, SequenceEntry]:
    """Inventory 100% of unique WT, parent, template, homolog, and unrelated sequences across Results 1-5."""
    base = REPO_ROOT if root is None else Path(root)
    registry: dict[str, SequenceEntry] = {}

    def register(seq: str, tag: str, example_id: str) -> None:
        cleaned = str(seq).strip().upper()
        if not cleaned:
            return
        digest = seq_sha256(cleaned)
        entry = registry.get(digest)
        if entry is None:
            registry[digest] = SequenceEntry(
                seq_sha256=digest,
                sequence=cleaned,
                tags={tag},
                example_ids=[example_id],
            )
        else:
            entry.tags.add(tag)
            if len(entry.example_ids) < 5 and example_id not in entry.example_ids:
                entry.example_ids.append(example_id)

    # 1. R1: All 217 ProteinGym DMS WTs + 174 ladder support WTs
    pg_csv = base / "data/proteingym_raw/DMS_substitutions.csv"
    with pg_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            register(row["target_seq"], "r1_proteingym_217", f"pg:{row['DMS_id']}")
    ladder_sup = json.loads(
        (base / "results/extensions/statistical_ladder_20261106/inputs/support.json").read_text(
            encoding="utf-8"
        )
    )
    wt_entries = ladder_sup["wildtypes"]
    if isinstance(wt_entries, dict):
        for wid, item in wt_entries.items():
            s = item.get("wildtype") or item.get("sequence") if isinstance(item, dict) else None
            if s:
                register(s, "r1_ladder_support", f"r1_sup:{wid}")
    elif isinstance(wt_entries, list):
        for idx, item in enumerate(wt_entries):
            s = item.get("wildtype") or item.get("sequence") if isinstance(item, dict) else None
            if s:
                register(s, "r1_ladder_support", f"r1_sup:{idx}")

    # 2. R2: All 101 MegaScale WTs + All 428 Domainome WTs
    ms_cohort = json.loads(
        (base / "results/R4/gate_stability_20260924/cohort/cohort.json").read_text(encoding="utf-8")
    )
    for bg in ms_cohort["backgrounds"]:
        register(bg["wildtype"], "r2_megascale_101", f"ms:{bg['name']}")

    dm_cohort = json.loads(
        (base / "results/R1/external_confirmation_20260924/cohort.json").read_text(encoding="utf-8")
    )
    for dom in dm_cohort["domains"]:
        register(dom["wildtype"], "r2_domainome_428", f"dm:{dom['name']}")

    # 3. R3: All 64 Epistasis WTs + All admitted E01 Contact WTs
    ep_plan = json.loads(
        (base / "archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json").read_text(
            encoding="utf-8"
        )
    )
    for bg in ep_plan["backgrounds"]:
        register(bg["wildtype"], "r3_epistasis_64", f"ep:{bg['name']}")

    with gzip.open(
        base / "results/extensions/mutation_structure_20261006/structure/annotation/sites.json.gz",
        "rt",
        encoding="utf-8",
    ) as handle:
        sites = json.load(handle)
    admitted_e01_assays = {row["assay_id"] for row in sites if row.get("status") == "admitted"}
    rescue_cohort = json.loads(
        (base / "results/R5/context_mutation_rescue_20260923/cohort.json").read_text(
            encoding="utf-8"
        )
    )
    by_rescue_assay = {a["assay"]: a for a in rescue_cohort["assays"]}
    for assay_id in sorted(admitted_e01_assays):
        if assay_id in by_rescue_assay:
            register(by_rescue_assay[assay_id]["wildtype"], "r3_e01_contacts", f"e01:{assay_id}")

    # 4. Result 4: Exp 4.1 natural fragments (400), R5a backbones (16), Exp 4.2 domains (8/12)
    exp41_nat = base / "results/extensions/unified_yardstick_20261124/exp4_1_pools/natural_fragments_fold_400.jsonl"
    for line in exp41_nat.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            register(row["sequence"], "r4_exp4_1_natural_400", f"exp41:{row['seq_id']}")

    r5a_bb = base / "results/audit_20261011_h200/ladder_original/ladder_backbones.jsonl"
    for line in r5a_bb.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            register(row["sequence"], "r4_r5a_backbones_16", f"r5a:{row['backbone_id']}")

    exp42_ref = json.loads(
        (base / "results/R6/e20_wild_type_references_20261013/freeze/wild_type_references.json").read_text(
            encoding="utf-8"
        )
    )
    for dom_id, info in sorted((exp42_ref.get("domains") or {}).items()):
        if isinstance(info, dict) and info.get("sequence"):
            register(info["sequence"], "r4_exp4_2_domains_12", f"exp42:{dom_id}")

    # 5a. Result 5 Main Table: 1,000 units (800 retained + 200 high_local_overlap)
    hom_cohort = json.loads(
        (base / "results/R5/context_homologue_20260826/cohort.json").read_text(encoding="utf-8")
    )
    prot = hom_cohort["protein"]
    recs = prot["records"]
    rel_map = {int(k): {int(x) for x in v} for k, v in prot["related"].items()}
    filler_idx = int(prot["filler"]["index"])
    register(str(recs[filler_idx]), "r5_main_filler", f"r5_filler:{filler_idx}")

    for unit in prot["units"]:
        t_idx = int(unit["target"])
        tgt_seq = str(recs[t_idx])
        ukey = str(unit["key"])
        register(tgt_seq, "r5_main_target", f"r5_tgt:{ukey}")
        partners = [int(p) for p in unit.get("partners") or []]
        unrs = r4s.pick_unrelated_partners(unit, recs, rel_map, len(partners), filler_idx=filler_idx)
        rem = 1024 - (len(tgt_seq) + 2)
        for pos_i, (p_idx, u_seq) in enumerate(zip(partners, unrs)):
            h_seq = str(recs[p_idx])
            if pos_i == 0:
                register(h_seq, "r5_main_homolog_1st", f"r5_hom1:{ukey}")
                register(u_seq, "r5_main_unrelated_1st", f"r5_unr1:{ukey}")
            if len(h_seq) + 1 <= rem:
                if include_r5_extra_b1024_contexts and pos_i > 0:
                    register(h_seq, "r5_main_homolog_b1024", f"r5_hom{pos_i+1}:{ukey}")
                    register(u_seq, "r5_main_unrelated_b1024", f"r5_unr{pos_i+1}:{ukey}")
                rem -= len(h_seq) + 1
            else:
                break

    # 5b. Result 5 Downstream 1: rescue_30_80 + e09_five_band
    for assay_row in rescue_cohort["assays"]:
        aid = str(assay_row["assay"])
        register(assay_row["wildtype"], "r5_d1_rescue_wt", f"rescue_wt:{aid}")
        plan, _ = r5d.plan_rescue_30_80_assay(
            assay_row,
            token_length_fn=len,
            overhead_tokens=3,
            budget_tokens=1024,
            multi_item_icl=False,
        )
        if plan:
            for seq in plan["homolog_items"]:
                register(seq, "r5_d1_rescue_homolog", f"rescue_hom:{aid}")
            for seq in plan["unrelated_items"]:
                register(seq, "r5_d1_rescue_unrelated", f"rescue_unr:{aid}")

    e09_doc = json.loads(
        (
            base / "results/extensions/homology_context_20261008/uniref90_2026_03/bin/homology_context.json"
        ).read_text(encoding="utf-8")
    )
    e09_by_tid: dict[str, Mapping[str, Any]] = {}
    for target in e09_doc["targets"]:
        tid = str(target["target_id"])
        e09_by_tid[tid] = target
        wt = str(target["wildtype"])
        register(wt, "r5_d1_e09_wt", f"e09_wt:{tid}")
        a_rows = [by_rescue_assay[aid] for aid in target["assays"] if aid in by_rescue_assay]
        e09_plan = r5d.plan_e09_target(
            target,
            a_rows,
            token_length_fn=len,
            overhead_tokens=3,
            budget_tokens=1024,
        )
        for cond_name, item_list in (e09_plan.get("items") or {}).items():
            for idx_it, item in enumerate(item_list):
                tag = "r5_d1_e09_ctx_1st" if idx_it == 0 else "r5_d1_e09_ctx_extra"
                if idx_it == 0 or include_r5_extra_b1024_contexts:
                    register(item["sequence"], tag, f"e09_{cond_name}:{tid}")

    # 5c. Result 5 Downstream 2: 22 continuation targets + planned contexts
    wf_path = base / "results/extensions/unified_yardstick_20261124/r4_continuation_pools/wt_folds.json"
    if wf_path.is_file():
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        for tid, info in sorted(wf.items()):
            if isinstance(info, dict) and info.get("sequence"):
                register(info["sequence"], "r5_d2_continuation_wt", f"cont_wt:{tid}")
    for tid in r4c.ALL_22_TARGETS:
        target = e09_by_tid.get(tid)
        if target is None:
            continue
        wt = str(target["wildtype"])
        register(wt, "r5_d2_continuation_wt", f"cont_wt:{tid}")
        referent = len(wt) + 2
        room = r4c.BUDGET_TOKENS - referent
        if room > 0:
            conds, _ = H.plan_conditions(
                wildtype=wt,
                bins=target["bins"],
                donors=target["unrelated_pool"],
                self_copy=target.get("self_copy"),
                token_length=lambda s: len(s) + 1,
                referent=referent,
                room=room,
            )
            for cond_label, e09_name in r4c.CONDITION_TO_E09_BIN.items():
                plan_obj = conds.get(e09_name)
                if plan_obj is not None and plan_obj.status == "present":
                    for idx_it, it in enumerate(plan_obj.items):
                        tag = "r5_d2_continuation_ctx_1st" if idx_it == 0 else "r5_d2_continuation_ctx_extra"
                        if idx_it == 0 or include_r5_extra_b1024_contexts:
                            register(it["sequence"], tag, f"cont_{cond_label}:{tid}")

    return registry


# --------------------------------------------------------------------------- #
# Fast Local Structure Indexer (Exact + Exact Contiguous Substring Slicing)
# --------------------------------------------------------------------------- #


def _scan_pdb_chunk(paths: Sequence[Path]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in paths:
        seq = scan_pdb_ca_sequence(p)
        if seq:
            out.append((str(p), seq))
    return out


def _scan_mmcif_chunk(paths: Sequence[Path]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for p in paths:
        chains = parse_mmcif_chains(p)
        for ch, (seq, _coords) in chains.items():
            if seq:
                out.append((str(p), ch, seq))
    return out


def _match_targets_against_corpus(
    unmatched: Mapping[str, SequenceEntry],
    corpus: Sequence[tuple[str, str, str]],
) -> dict[str, tuple[str, str, int, int, bool]]:
    """Match ``unmatched`` sequences against ``corpus = [(path, chain_or_empty, seq), ...]``.

    Returns ``{seq_sha256: (source_path, chain, slice_start, slice_end, is_exact)}``.
    Exact full-length matches always take precedence over contiguous substring matches.
    Uses a 12-mer seed-and-verify inverted index for $O(\\sum |\\text{corpus}|)$ speed
    with zero false negatives for all sequences of length >= 12.
    """
    if not unmatched or not corpus:
        return {}

    exact_by_seq: dict[str, str] = {entry.sequence: digest for digest, entry in unmatched.items()}
    matches: dict[str, tuple[str, str, int, int, bool]] = {}

    # Pass 1: O(1) exact full-length match
    for src_path, ch, full_seq in corpus:
        digest = exact_by_seq.get(full_seq)
        if digest is not None and digest not in matches:
            matches[digest] = (src_path, ch, 0, len(full_seq), True)

    remaining = {d: e for d, e in unmatched.items() if d not in matches}
    if not remaining:
        return matches

    # Pass 2: 12-mer seed index for exact contiguous substring matches
    seed_to_targets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    short_targets: list[tuple[str, str]] = []
    for digest, entry in remaining.items():
        if len(entry.sequence) >= MIN_SEED_KMER:
            seed_to_targets[entry.sequence[:MIN_SEED_KMER]].append((digest, entry.sequence))
        else:
            short_targets.append((digest, entry.sequence))

    for src_path, ch, full_seq in corpus:
        if len(matches) == len(unmatched):
            break
        flen = len(full_seq)
        if flen >= MIN_SEED_KMER and seed_to_targets:
            for pos in range(0, flen - MIN_SEED_KMER + 1):
                cands = seed_to_targets.get(full_seq[pos : pos + MIN_SEED_KMER])
                if not cands:
                    continue
                for digest, tseq in cands:
                    if digest in matches:
                        continue
                    end = pos + len(tseq)
                    if end <= flen and full_seq[pos:end] == tseq:
                        matches[digest] = (src_path, ch, pos, end, False)
        for digest, tseq in short_targets:
            if digest in matches:
                continue
            idx = full_seq.find(tseq)
            if idx >= 0:
                matches[digest] = (src_path, ch, idx, idx + len(tseq), False)

    return matches


def _extract_pdb_matches_worker(
    items: Sequence[tuple[str, list[tuple[str, str, int, int, bool, str]]]],
    cache_dir_str: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Parse each matched PDB once and write sliced ``.npz`` backbones."""
    cache_dir = Path(cache_dir_str)
    out: list[tuple[str, dict[str, Any]]] = []
    for src_path_s, hits in items:
        src_path = Path(src_path_s)
        full_seq, full_coords, full_plddt = parse_pdb_backbone(src_path)
        for digest, expected_seq, start, end, is_exact, kind_prefix in hits:
            sliced_seq = full_seq[start:end]
            if sliced_seq != expected_seq:
                raise ValueError(
                    f"Sequence mismatch slicing {src_path}[{start}:{end}]: "
                    f"len {len(sliced_seq)} vs {len(expected_seq)}"
                )
            coords = full_coords[start:end]
            plddt = full_plddt[start:end]
            kind = f"{kind_prefix}_exact" if is_exact else f"{kind_prefix}_substring"
            rel = backbone_npz_relpath(digest)
            stats = save_backbone_npz(
                cache_dir / rel,
                sequence=expected_seq,
                coords=coords,
                plddt=plddt,
                source_kind=kind,
                source_path=src_path_s,
            )
            out.append(
                (
                    digest,
                    {
                        "resolved": True,
                        "npz_relpath": str(rel),
                        "source_kind": kind,
                        "source_path": src_path_s,
                        "chain": "",
                        "slice_start": int(start),
                        "slice_end": int(end),
                        **stats,
                    },
                )
            )
    return out


def _extract_mmcif_matches_worker(
    items: Sequence[tuple[str, list[tuple[str, str, str, int, int, bool, str]]]],
    cache_dir_str: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Parse each matched mmCIF once and write sliced ``.npz`` backbones."""
    cache_dir = Path(cache_dir_str)
    out: list[tuple[str, dict[str, Any]]] = []
    for src_path_s, hits in items:
        src_path = Path(src_path_s)
        chains = parse_mmcif_chains(src_path)
        for digest, ch, expected_seq, start, end, is_exact, kind_prefix in hits:
            full_seq, full_coords = chains[ch]
            sliced_seq = full_seq[start:end]
            if sliced_seq != expected_seq:
                raise ValueError(f"Sequence mismatch slicing {src_path} chain {ch}[{start}:{end}]")
            coords = full_coords[start:end]
            kind = f"{kind_prefix}_exact" if is_exact else f"{kind_prefix}_substring"
            rel = backbone_npz_relpath(digest)
            stats = save_backbone_npz(
                cache_dir / rel,
                sequence=expected_seq,
                coords=coords,
                plddt=None,
                source_kind=kind,
                source_path=src_path_s,
            )
            out.append(
                (
                    digest,
                    {
                        "resolved": True,
                        "npz_relpath": str(rel),
                        "source_kind": kind,
                        "source_path": src_path_s,
                        "chain": ch,
                        "slice_start": int(start),
                        "slice_end": int(end),
                        **stats,
                    },
                )
            )
    return out


def build_wt_3d_backbone_cache(
    *,
    root: Path | None = None,
    cache_dir: Path | None = None,
    workers: int = 8,
    include_r5_extra_b1024_contexts: bool = True,
    scan_afdb: bool = True,
    scan_experimental: bool = True,
) -> dict[str, Any]:
    """Build or update the unified 3D WT backbone cache and emit ``pending_wt_esmfold2.jsonl``."""
    base = REPO_ROOT if root is None else Path(root)
    target_dir = (base / DEFAULT_CACHE_DIR) if cache_dir is None else Path(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, min(int(workers), 16))

    registry = collect_all_benchmark_sequences(
        root=base,
        include_r5_extra_b1024_contexts=include_r5_extra_b1024_contexts,
    )

    # Reuse any already-cached .npz backbones cleanly (e.g., previously ingested ESMFold2 folds).
    resolved: dict[str, dict[str, Any]] = {}
    manifest_path = target_dir / "manifest.json"
    existing_entries: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        old_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_entries = dict(old_doc.get("entries") or {})

    for digest, entry in registry.items():
        rel = backbone_npz_relpath(digest)
        npz_path = target_dir / rel
        if npz_path.is_file():
            prev = existing_entries.get(digest)
            if prev and prev.get("resolved"):
                resolved[digest] = dict(prev)
            else:
                loaded = load_backbone_npz(target_dir, digest)
                if loaded["sequence"] == entry.sequence:
                    valid_p = loaded["plddt"][np.isfinite(loaded["plddt"])]
                    resolved[digest] = {
                        "resolved": True,
                        "npz_relpath": str(rel),
                        "source_kind": loaded["source_kind"],
                        "source_path": loaded["source_path"],
                        "chain": "",
                        "slice_start": 0,
                        "slice_end": len(entry.sequence),
                        "length": len(entry.sequence),
                        "n_valid_ncac": int(loaded["res_mask"].sum()),
                        "n_valid_ca": int(loaded["ca_mask"].sum()),
                        "coverage_fraction": float(loaded["res_mask"].mean()),
                        "mean_plddt": float(valid_p.mean()) if len(valid_p) else None,
                    }

    mp_ctx = multiprocessing.get_context("fork")

    def _run_pdb_source(pdb_files: Sequence[Path], kind_prefix: str) -> None:
        unmatched = {d: e for d, e in registry.items() if d not in resolved}
        if not unmatched or not pdb_files:
            return
        chunks = [pdb_files[i::n_workers] for i in range(n_workers) if pdb_files[i::n_workers]]
        if n_workers == 1 or len(pdb_files) < 32:
            scanned = _scan_pdb_chunk(pdb_files)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(chunks), mp_context=mp_ctx
            ) as pool:
                scanned = [row for sub in pool.map(_scan_pdb_chunk, chunks) for row in sub]
        corpus = [(p_s, "", seq) for p_s, seq in scanned]
        hits = _match_targets_against_corpus(unmatched, corpus)
        if not hits:
            return
        by_file: dict[str, list[tuple[str, str, int, int, bool, str]]] = defaultdict(list)
        for digest, (src_p, _ch, s, e, exact) in hits.items():
            by_file[src_p].append((digest, registry[digest].sequence, s, e, exact, kind_prefix))
        file_items = list(by_file.items())
        if n_workers == 1 or len(file_items) < 16:
            written = _extract_pdb_matches_worker(file_items, str(target_dir))
        else:
            f_chunks = [file_items[i::n_workers] for i in range(n_workers) if file_items[i::n_workers]]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(f_chunks), mp_context=mp_ctx
            ) as pool:
                futs = [
                    pool.submit(_extract_pdb_matches_worker, ch_items, str(target_dir))
                    for ch_items in f_chunks
                ]
                written = [row for fut in futs for row in fut.result()]
        for digest, rec in written:
            resolved[digest] = rec

    def _run_mmcif_source(cif_files: Sequence[Path], kind_prefix: str) -> None:
        unmatched = {d: e for d, e in registry.items() if d not in resolved}
        if not unmatched or not cif_files:
            return
        if n_workers == 1 or len(cif_files) < 16:
            corpus = _scan_mmcif_chunk(cif_files)
        else:
            chunks = [cif_files[i::n_workers] for i in range(n_workers) if cif_files[i::n_workers]]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(chunks), mp_context=mp_ctx
            ) as pool:
                corpus = [row for sub in pool.map(_scan_mmcif_chunk, chunks) for row in sub]
        hits = _match_targets_against_corpus(unmatched, corpus)
        if not hits:
            return
        by_file: dict[str, list[tuple[str, str, str, int, int, bool, str]]] = defaultdict(list)
        for digest, (src_p, ch, s, e, exact) in hits.items():
            by_file[src_p].append((digest, ch, registry[digest].sequence, s, e, exact, kind_prefix))
        file_items = list(by_file.items())
        written = _extract_mmcif_matches_worker(file_items, str(target_dir))
        for digest, rec in written:
            resolved[digest] = rec

    # Resolution priority:
    # 1. Sealed Exp 4.2 freeze WT PDBs (exact 12 WTs, complete atoms)
    freeze_dir = base / DEFAULT_FREEZE_PDB_DIR
    if freeze_dir.is_dir():
        _run_pdb_source(sorted(freeze_dir.glob("*.pdb")), "exp_freeze_pdb")

    # 2. AlphaFold DB v6 (all 23,586 F1..F99 complete backbones; exact + exact contiguous substring slicing)
    afdb_dir = base / DEFAULT_AFDB_DIR
    if scan_afdb and afdb_dir.is_dir():
        _run_pdb_source(sorted(afdb_dir.glob("AF-*-model_v6.pdb.gz")), "afdb_v6")

    # 3. Experimental contact mmCIFs (64 structures) + binding mmCIFs (315 structures)
    if scan_experimental:
        exp_dir = base / DEFAULT_EXP_CONTACT_CIF_DIR
        if exp_dir.is_dir():
            _run_mmcif_source(sorted(exp_dir.glob("*.cif.gz")), "exp_mmcif")
        bind_dir = base / DEFAULT_BINDING_CIF_DIR
        if bind_dir.is_dir():
            _run_mmcif_source(sorted(bind_dir.glob("*.cif.gz")), "exp_binding_mmcif")

    # 4. Local CATH-S40 non-redundant experimental PDB domains (34,653 domains via FASTA index)
    cath_fa = DEFAULT_CATH_S40_DIR / "cath-dataset-nonredundant-S40.atom.fa"
    cath_tgz = DEFAULT_CATH_S40_DIR / "cath-dataset-nonredundant-S40.pdb.tgz"
    if scan_experimental and cath_fa.is_file() and cath_tgz.is_file():
        unmatched = {d: e for d, e in registry.items() if d not in resolved}
        if unmatched:
            cath_corpus: list[tuple[str, str, str]] = []
            cur_dom: str | None = None
            cur_buf: list[str] = []
            with cath_fa.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith(">"):
                        if cur_dom is not None:
                            cath_corpus.append((cur_dom, "", "".join(cur_buf).strip().upper()))
                        token = line[1:].split()[0]
                        cur_dom = token.split("|")[-1].split("/")[0]
                        cur_buf = []
                    else:
                        cur_buf.append(line.strip())
                if cur_dom is not None:
                    cath_corpus.append((cur_dom, "", "".join(cur_buf).strip().upper()))
            cath_hits = _match_targets_against_corpus(unmatched, cath_corpus)
            if cath_hits:
                by_dom: dict[str, list[tuple[str, str, int, int, bool]]] = defaultdict(list)
                for digest, (dom_id, _ch, s, e, exact) in cath_hits.items():
                    by_dom[dom_id].append((digest, registry[digest].sequence, s, e, exact))
                wanted_members = {f"dompdb/{dom_id}": dom_id for dom_id in by_dom}
                with tarfile.open(cath_tgz, "r:gz") as tf:
                    remaining = set(wanted_members)
                    for member in tf:
                        if member.name not in remaining:
                            continue
                        dom_id = wanted_members[member.name]
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            continue
                        raw_lines = fobj.read().decode("utf-8", errors="replace").splitlines()
                        full_seq, full_coords, _ = parse_pdb_backbone_lines(raw_lines)
                        for digest, expected_seq, _s, _e, _exact in by_dom[dom_id]:
                            idx = full_seq.find(expected_seq)
                            if idx < 0:
                                continue
                            s_act = idx
                            e_act = idx + len(expected_seq)
                            is_exact_act = len(full_seq) == len(expected_seq)
                            kind = "cath_s40_exact" if is_exact_act else "cath_s40_substring"
                            rel = backbone_npz_relpath(digest)
                            src_label = f"{cath_tgz}:{member.name}"
                            stats = save_backbone_npz(
                                target_dir / rel,
                                sequence=expected_seq,
                                coords=full_coords[s_act:e_act],
                                plddt=None,
                                source_kind=kind,
                                source_path=src_label,
                            )
                            resolved[digest] = {
                                "resolved": True,
                                "npz_relpath": str(rel),
                                "source_kind": kind,
                                "source_path": src_label,
                                "chain": "",
                                "slice_start": int(s_act),
                                "slice_end": int(e_act),
                                **stats,
                            }
                        remaining.discard(member.name)
                        if not remaining:
                            break

    # Build pending_wt_esmfold2.jsonl for all unmatched sequences (3 priority tiers, windowed if L > 1024)
    tier_rank = {name: idx for idx, name in enumerate(TIER_ORDER)}
    unmatched_entries = [e for d, e in registry.items() if d not in resolved]
    unmatched_entries.sort(
        key=lambda e: (tier_rank[e.priority_tier], len(e.sequence), e.seq_sha256)
    )
    pending_rows: list[dict[str, Any]] = []
    for entry in unmatched_entries:
        windows = plan_overlapping_windows(
            entry.sequence, max_len=ESMFOLD_MAX_LEN, overlap=WINDOW_OVERLAP
        )
        is_windowed = len(windows) > 1
        for w_idx, (w_start, w_end) in enumerate(windows):
            sub_seq = entry.sequence[w_start:w_end]
            job_id = (
                f"{entry.seq_sha256}__w{w_idx:02d}_{w_start}_{w_end}"
                if is_windowed
                else entry.seq_sha256
            )
            pending_rows.append(
                {
                    "schema": SCHEMA_PENDING,
                    "job_id": job_id,
                    "parent_seq_sha256": entry.seq_sha256,
                    "parent_length": len(entry.sequence),
                    "sequence": sub_seq,
                    "length": len(sub_seq),
                    "is_windowed": is_windowed,
                    "window_index": w_idx,
                    "num_windows": len(windows),
                    "window_start": w_start,
                    "window_end": w_end,
                    "priority_tier": entry.priority_tier,
                    "tags": sorted(entry.tags),
                    "example_ids": entry.example_ids[:3],
                }
            )

    pending_path = target_dir / "pending_wt_esmfold2.jsonl"
    pending_sha = atomic_write_jsonl_with_sha256(pending_path, pending_rows)

    # Assemble manifest & per-benchmark coverage summary
    manifest_entries: dict[str, dict[str, Any]] = {}
    tag_total: Counter[str] = Counter()
    tag_resolved: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    tier_summary: dict[str, dict[str, int]] = {
        name: {"total": 0, "resolved": 0, "pending_seqs": 0, "pending_windows": 0}
        for name in TIER_ORDER
    }

    for digest, entry in sorted(registry.items()):
        tier = entry.priority_tier
        tier_summary[tier]["total"] += 1
        is_res = digest in resolved
        for tag in entry.tags:
            tag_total[tag] += 1
            if is_res:
                tag_resolved[tag] += 1
        if is_res:
            tier_summary[tier]["resolved"] += 1
            rec = dict(resolved[digest])
            source_counts[str(rec["source_kind"])] += 1
            manifest_entries[digest] = {
                "seq_sha256": digest,
                "length": len(entry.sequence),
                "priority_tier": tier,
                "tags": sorted(entry.tags),
                "example_ids": entry.example_ids,
                **rec,
            }
        else:
            tier_summary[tier]["pending_seqs"] += 1
            wins = plan_overlapping_windows(
                entry.sequence, max_len=ESMFOLD_MAX_LEN, overlap=WINDOW_OVERLAP
            )
            tier_summary[tier]["pending_windows"] += len(wins)
            manifest_entries[digest] = {
                "seq_sha256": digest,
                "length": len(entry.sequence),
                "priority_tier": tier,
                "tags": sorted(entry.tags),
                "example_ids": entry.example_ids,
                "resolved": False,
                "npz_relpath": None,
                "source_kind": "pending_esmfold2",
                "num_pending_windows": len(wins),
            }

    by_benchmark = {
        tag: {
            "total": int(tag_total[tag]),
            "resolved": int(tag_resolved[tag]),
            "pending": int(tag_total[tag] - tag_resolved[tag]),
            "coverage_pct": round(100.0 * tag_resolved[tag] / max(1, tag_total[tag]), 2),
        }
        for tag in sorted(tag_total)
    }

    manifest_doc = {
        "schema": SCHEMA_MANIFEST,
        "created_utc": _now_utc(),
        "atom_order": list(ATOM_ORDER),
        "leakage_policy": "wild-type, template, parent, homolog, and unrelated sequences ONLY; zero mutant backbones",
        "esmfold_windowing": {"max_len": ESMFOLD_MAX_LEN, "overlap": WINDOW_OVERLAP},
        "totals": {
            "n_unique_sequences": len(registry),
            "n_resolved": len(resolved),
            "n_pending_sequences": len(unmatched_entries),
            "n_pending_esmfold2_jobs": len(pending_rows),
            "n_pending_long_over_1024": sum(1 for e in unmatched_entries if len(e.sequence) > ESMFOLD_MAX_LEN),
        },
        "by_priority_tier": tier_summary,
        "by_source_kind": dict(sorted(source_counts.items())),
        "by_benchmark": by_benchmark,
        "pending_jsonl": {
            "path": str(pending_path.name),
            "sha256": pending_sha,
            "n_rows": len(pending_rows),
        },
        "entries": manifest_entries,
    }
    manifest_sha = atomic_write_json_with_sha256(manifest_path, manifest_doc)
    manifest_doc["manifest_sha256"] = manifest_sha
    return manifest_doc


def ingest_esmfold2_folded_records(
    folded_inputs: Iterable[Path],
    *,
    cache_dir: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Ingest H200 ESMFold2 folded ``(N, CA, C, O)`` outputs (direct + Kabsch-stitched windows).

    Each input JSONL line (or ``.npz`` companion) must carry:
    ``parent_seq_sha256``, ``sequence``, ``window_start``, ``window_end``,
    ``num_windows``, ``coords`` (shape ``(L_w, 4, 3)`` nested list or ``coords_npz`` path),
    and optional ``plddt`` (shape ``(L_w,)``).
    """
    target_dir = Path(cache_dir)
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for src in folded_inputs:
        src = Path(src)
        files = sorted(src.glob("*.jsonl")) if src.is_dir() else [src]
        for fpath in files:
            for line in fpath.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "coords" not in rec and rec.get("coords_npz"):
                    with np.load(rec["coords_npz"], allow_pickle=False) as d:
                        rec["coords"] = d["coords"]
                        if "plddt" in d:
                            rec["plddt"] = d["plddt"]
                if "coords" not in rec and rec.get("pdb_path"):
                    _s, c_arr, p_arr = parse_pdb_backbone(Path(rec["pdb_path"]))
                    rec["coords"] = c_arr
                    rec["plddt"] = p_arr
                by_parent[str(rec["parent_seq_sha256"])].append(rec)

    n_direct = 0
    n_stitched = 0
    max_stitch_rmsd = 0.0

    for parent_sha, wins in sorted(by_parent.items()):
        expected_n = int(wins[0].get("num_windows", 1))
        if len(wins) < expected_n:
            continue
        rel = backbone_npz_relpath(parent_sha)
        if expected_n == 1:
            w0 = wins[0]
            seq = str(w0["sequence"]).strip().upper()
            if seq_sha256(seq) != parent_sha:
                raise ValueError(f"Sequence SHA256 mismatch for direct ESMFold2 job {parent_sha}")
            save_backbone_npz(
                target_dir / rel,
                sequence=seq,
                coords=np.asarray(w0["coords"], dtype=np.float32),
                plddt=None if w0.get("plddt") is None else np.asarray(w0["plddt"], dtype=np.float32),
                source_kind="esmfold2_direct",
                source_path=str(w0.get("source_path", "esmfold2_h200")),
            )
            n_direct += 1
        else:
            parent_len = int(wins[0]["parent_length"])
            ordered = sorted(wins, key=lambda w: int(w["window_start"]))
            chars = [""] * parent_len
            for w in ordered:
                s, e = int(w["window_start"]), int(w["window_end"])
                win_seq = str(w["sequence"]).strip().upper()
                if e - s != len(win_seq):
                    raise ValueError(
                        f"Window [{s}, {e}) length mismatch ({e - s} != {len(win_seq)}) for {parent_sha}"
                    )
                for offset, ch in enumerate(win_seq):
                    chars[s + offset] = ch
            parent_seq = "".join(chars)
            if len(parent_seq) != parent_len or seq_sha256(parent_seq) != parent_sha:
                raise ValueError(f"Reconstructed parent sequence mismatch for {parent_sha}")
            stitched_coords, stitched_plddt, rmsds = stitch_windowed_backbones(ordered, parent_len)
            if rmsds:
                max_stitch_rmsd = max(max_stitch_rmsd, max(rmsds))
            save_backbone_npz(
                target_dir / rel,
                sequence=parent_seq,
                coords=stitched_coords,
                plddt=stitched_plddt,
                source_kind="esmfold2_kabsch_stitched",
                source_path=f"esmfold2_h200:{len(ordered)}w_max_rmsd={max(rmsds, default=0.0):.3f}",
            )
            n_stitched += 1

    # Rebuild manifest cleanly reflecting newly ingested backbones
    manifest = build_wt_3d_backbone_cache(
        root=root,
        cache_dir=target_dir,
        workers=4,
        scan_afdb=False,
        scan_experimental=False,
    )
    return {
        "n_ingested_direct": n_direct,
        "n_ingested_kabsch_stitched": n_stitched,
        "max_stitch_overlap_rmsd_angstrom": max_stitch_rmsd,
        "totals": manifest["totals"],
    }


def _finite_check(obj: Any) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_finite_check(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_finite_check(v) for v in obj)
    return True
