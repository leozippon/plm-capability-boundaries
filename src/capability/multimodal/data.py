"""Paired (AA, 3Di, C_beta geometry) 3D multimodal data store, Foldseek 3Di extraction, and fold-disjoint splits."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..supply_demand.corpus import (
    cath_topology_of,
    seq_sha256,
)
from ..supply_demand.objectives import PackedSequenceStore, encode_aa_sequence
from .spec import (
    AA20,
    GEOM_FEAT_DIM,
    PAD_TOKEN_ID,
    STRUCT_PAD_TOKEN_ID,
    THREEDI_ALPHABET,
    THREEDI_TO_ID,
    VOCAB_3DI,
)

MULTIMODAL_BUNDLE_SCHEMA = "multimodal_3d_bundle_v1"
AA20_SET: frozenset[str] = frozenset(AA20)


def encode_3di_sequence(three_di: str) -> np.ndarray:
    """Encode a 20-letter Foldseek 3Di string into uint8 indices in [0, 19]."""
    s = three_di.strip().upper()
    if not s:
        raise ValueError("Cannot encode empty 3Di sequence")
    out = np.empty(len(s), dtype=np.uint8)
    for idx, ch in enumerate(s):
        tok = THREEDI_TO_ID.get(ch)
        if tok is None:
            # Map unknown/X 3Di character deterministically to state 0 ('A')
            tok = 0
        out[idx] = tok
    return out


def compute_virtual_cb(coords_ncaco: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ideal virtual C_beta coordinates from (L, >=3, 3) [N, CA, C, ...] backbone coordinates."""
    arr = np.asarray(coords_ncaco, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] < 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected coords of shape (L, >=3, 3), got {arr.shape}")
    n_xyz = arr[:, 0, :]
    ca_xyz = arr[:, 1, :]
    c_xyz = arr[:, 2, :]
    valid = np.isfinite(n_xyz).all(axis=1) & np.isfinite(ca_xyz).all(axis=1) & np.isfinite(c_xyz).all(axis=1)
    cb = np.zeros_like(ca_xyz, dtype=np.float32)
    if np.any(valid):
        b = ca_xyz[valid] - n_xyz[valid]
        c = c_xyz[valid] - ca_xyz[valid]
        a = np.cross(b, c)
        cb[valid] = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_xyz[valid]
    return cb, valid.astype(np.bool_)


def _dihedral_angles(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Vectorized dihedral angle (radians) for batches of 4 points (N, 3)."""
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_u = b1 / np.maximum(b1_norm, 1e-8)
    v = b0 - np.sum(b0 * b1_u, axis=-1, keepdims=True) * b1_u
    w = b2 - np.sum(b2 * b1_u, axis=-1, keepdims=True) * b1_u
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_u, v) * w, axis=-1)
    return np.arctan2(y, x)


def compute_backbone_geom_features(
    cb_coords: np.ndarray,
    cb_mask: np.ndarray,
    coords_ncaco: np.ndarray | None = None,
) -> np.ndarray:
    """Compute rotation/translation-invariant (L, GEOM_FEAT_DIM=18) float32 residue features.

    Channels:
      0..7:  sequential CB inverse-distances ``1 / (1 + d(i, i+k) / 8.0)`` for k in (-4,-3,-2,-1,+1,+2,+3,+4)
      8..13: sin/cos of backbone dihedrals (phi, psi, omega) when N/CA/C provided (else 0)
      14..16: non-local (|i-j| >= 6) contact densities within 8A, 12A, 16A (normalized by 16.0, clipped to [0, 2])
      17:    finite CB coordinate validity flag (0.0 or 1.0)
    """
    cb = np.asarray(cb_coords, dtype=np.float32)
    mask = np.asarray(cb_mask, dtype=np.bool_)
    L = int(cb.shape[0])
    feats = np.zeros((L, GEOM_FEAT_DIM), dtype=np.float32)
    if L == 0:
        return feats

    safe_cb = np.where(mask[:, None], cb, 0.0)
    offsets = (-4, -3, -2, -1, 1, 2, 3, 4)
    for col, k in enumerate(offsets):
        if abs(k) >= L:
            continue
        if k > 0:
            pair_ok = mask[:-k] & mask[k:]
            dist = np.linalg.norm(safe_cb[:-k] - safe_cb[k:], axis=-1)
            feats[:-k, col] = np.where(pair_ok, 1.0 / (1.0 + dist / 8.0), 0.0)
        else:
            s = -k
            pair_ok = mask[s:] & mask[:-s]
            dist = np.linalg.norm(safe_cb[s:] - safe_cb[:-s], axis=-1)
            feats[s:, col] = np.where(pair_ok, 1.0 / (1.0 + dist / 8.0), 0.0)

    if coords_ncaco is not None and coords_ncaco.shape[0] == L and coords_ncaco.shape[1] >= 3 and L >= 2:
        n_xyz = np.asarray(coords_ncaco[:, 0, :], dtype=np.float64)
        ca_xyz = np.asarray(coords_ncaco[:, 1, :], dtype=np.float64)
        c_xyz = np.asarray(coords_ncaco[:, 2, :], dtype=np.float64)
        ncac_ok = (
            np.isfinite(n_xyz).all(axis=1)
            & np.isfinite(ca_xyz).all(axis=1)
            & np.isfinite(c_xyz).all(axis=1)
        )
        # phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i) for i = 1..L-1
        phi_ok = ncac_ok[:-1] & ncac_ok[1:]
        if np.any(phi_ok):
            phi = _dihedral_angles(c_xyz[:-1][phi_ok], n_xyz[1:][phi_ok], ca_xyz[1:][phi_ok], c_xyz[1:][phi_ok])
            idx = np.flatnonzero(phi_ok) + 1
            feats[idx, 8] = np.sin(phi).astype(np.float32)
            feats[idx, 9] = np.cos(phi).astype(np.float32)
        # psi_i = dihedral(N_i, CA_i, C_i, N_{i+1}) for i = 0..L-2
        psi_ok = ncac_ok[:-1] & ncac_ok[1:]
        if np.any(psi_ok):
            psi = _dihedral_angles(n_xyz[:-1][psi_ok], ca_xyz[:-1][psi_ok], c_xyz[:-1][psi_ok], n_xyz[1:][psi_ok])
            idx = np.flatnonzero(psi_ok)
            feats[idx, 10] = np.sin(psi).astype(np.float32)
            feats[idx, 11] = np.cos(psi).astype(np.float32)
        # omega_i = dihedral(CA_{i-1}, C_{i-1}, N_i, CA_i) for i = 1..L-1
        if np.any(phi_ok):
            omega = _dihedral_angles(ca_xyz[:-1][phi_ok], c_xyz[:-1][phi_ok], n_xyz[1:][phi_ok], ca_xyz[1:][phi_ok])
            idx = np.flatnonzero(phi_ok) + 1
            feats[idx, 12] = np.sin(omega).astype(np.float32)
            feats[idx, 13] = np.cos(omega).astype(np.float32)

    # Non-local contact densities (|i-j| >= 6)
    if L >= 7 and int(mask.sum()) >= 2:
        diff = safe_cb[:, None, :] - safe_cb[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        ii, jj = np.indices((L, L))
        nonlocal_valid = (np.abs(ii - jj) >= 6) & mask[:, None] & mask[None, :]
        for c_idx, cutoff in enumerate((8.0, 12.0, 16.0)):
            cnt = ((dmat < cutoff) & nonlocal_valid).sum(axis=-1).astype(np.float32)
            feats[:, 14 + c_idx] = np.clip(cnt / 16.0, 0.0, 2.0)

    feats[:, 17] = mask.astype(np.float32)
    return feats


def discretize_local_3di_fallback(
    cb_coords: np.ndarray,
    cb_mask: np.ndarray,
) -> str:
    """Deterministic 20-state geometric fallback tokenizer when Foldseek binary is not used."""
    feats = compute_backbone_geom_features(cb_coords, cb_mask, coords_ncaco=None)
    # Combine local i+2, i+3, i+4 compaction and 8A contact density into a 20-state code
    score = (
        4.0 * feats[:, 5]
        + 6.0 * feats[:, 6]
        + 5.0 * feats[:, 7]
        + 5.0 * feats[:, 14]
    )
    bins = np.clip(np.floor(score * 2.0).astype(np.int64), 0, VOCAB_3DI - 1)
    return "".join(THREEDI_ALPHABET[int(b)] for b in bins)


def ensure_foldseek_binary(
    tarball_path: Path,
    scratch_dir: Path = Path("/tmp/multimodal_foldseek_tool"),
) -> Path:
    """Extract Foldseek binary from tarball into scratch_dir if not already extracted."""
    binary = scratch_dir / "foldseek/bin/foldseek"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    if not tarball_path.is_file():
        raise FileNotFoundError(f"Foldseek tarball not found: {tarball_path}")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tf:
        tf.extractall(scratch_dir)
    if not binary.is_file():
        raise FileNotFoundError(f"Expected Foldseek binary at {binary} after extracting {tarball_path}")
    return binary


def extract_foldseek_3di_map(
    pdb_dir: Path,
    *,
    foldseek_bin: Path,
    work_dir: Path,
    threads: int = 16,
) -> dict[str, tuple[str, str]]:
    """Run ``foldseek structureto3didescriptor`` over a directory of PDB files.

    Returns ``{stem: (aa_seq, three_di_seq)}``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    desc_tsv = work_dir / "descriptors.tsv"
    subprocess.run(
        [
            str(foldseek_bin),
            "structureto3didescriptor",
            str(pdb_dir),
            str(desc_tsv),
            "--threads",
            str(int(threads)),
            "-v",
            "1",
        ],
        check=True,
    )
    out: dict[str, tuple[str, str]] = {}
    with desc_tsv.open("r", encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            raw_name = cols[0].split()[0]
            stem = raw_name
            for suf in (".pdb.gz", ".cif.gz", ".pdb", ".cif"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
            aa_s = cols[1].strip().upper()
            tdi_s = cols[2].strip().upper()
            if aa_s and len(aa_s) == len(tdi_s):
                out[stem] = (aa_s, tdi_s)
    return out


@dataclass(frozen=True)
class MultimodalDomainRecord:
    """One paired 3D protein domain with AA sequence, 3Di tokens, C_beta coordinates, and CATH fold."""

    domain_id: str
    sequence: str
    three_di: str
    cb_coords: np.ndarray     # (L, 3) float32
    cb_mask: np.ndarray       # (L,) bool
    geom_feats: np.ndarray    # (L, GEOM_FEAT_DIM) float32
    cath_h: str
    cath_t: str

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class PackedMultimodalStore:
    """Compact contiguous multimodal store holding paired (AA, 3Di, C_beta, local geom) tracks."""

    aa_flat: np.ndarray         # (total_tokens,) uint8 in 0..19
    three_di_flat: np.ndarray   # (total_tokens,) uint8 in 0..19
    cb_coords_flat: np.ndarray  # (total_tokens, 3) float32
    cb_mask_flat: np.ndarray    # (total_tokens,) bool
    geom_feats_flat: np.ndarray # (total_tokens, GEOM_FEAT_DIM) float32
    offsets: np.ndarray         # (n_sequences + 1,) int64
    lengths: np.ndarray         # (n_sequences,) int32
    ids: np.ndarray | None = None
    cath_h: np.ndarray | None = None
    cath_t: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.aa_flat.ndim != 1 or self.aa_flat.dtype != np.uint8:
            raise ValueError("aa_flat must be 1D uint8")
        if self.three_di_flat.shape != self.aa_flat.shape or self.three_di_flat.dtype != np.uint8:
            raise ValueError("three_di_flat must match aa_flat shape with uint8 dtype")
        n_tok = int(self.aa_flat.size)
        if self.cb_coords_flat.shape != (n_tok, 3):
            raise ValueError(f"cb_coords_flat shape {self.cb_coords_flat.shape} != ({n_tok}, 3)")
        if self.cb_mask_flat.shape != (n_tok,):
            raise ValueError(f"cb_mask_flat shape {self.cb_mask_flat.shape} != ({n_tok},)")
        if self.geom_feats_flat.shape != (n_tok, GEOM_FEAT_DIM):
            raise ValueError(
                f"geom_feats_flat shape {self.geom_feats_flat.shape} != ({n_tok}, {GEOM_FEAT_DIM})"
            )

    @property
    def n_sequences(self) -> int:
        return int(self.lengths.size)

    @property
    def total_tokens(self) -> int:
        return int(self.aa_flat.size)

    def to_aa_sequence_store(self) -> PackedSequenceStore:
        """Zero-copy view of the AA sequence track as a standard Stage-2 PackedSequenceStore."""
        return PackedSequenceStore(
            tokens_flat=self.aa_flat,
            offsets=self.offsets,
            lengths=self.lengths,
        )

    def collate_indices(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        device: torch.device | str | None = None,
        pad_to_multiple: int = 8,
    ) -> dict[str, torch.Tensor]:
        """Collate a mini-batch of paired multimodal tracks padded to a multiple of pad_to_multiple."""
        idx_arr = np.asarray(indices, dtype=np.int64)
        bsz = int(idx_arr.size)
        if bsz == 0:
            raise ValueError("Cannot collate empty index batch")
        lens = self.lengths[idx_arr]
        max_len = int(lens.max())
        if pad_to_multiple > 1:
            padded_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        else:
            padded_len = max_len

        aa_buf = np.full((bsz, padded_len), PAD_TOKEN_ID, dtype=np.uint8)
        s3_buf = np.full((bsz, padded_len), STRUCT_PAD_TOKEN_ID, dtype=np.uint8)
        cb_buf = np.zeros((bsz, padded_len, 3), dtype=np.float32)
        cbm_buf = np.zeros((bsz, padded_len), dtype=np.bool_)
        geom_buf = np.zeros((bsz, padded_len, GEOM_FEAT_DIM), dtype=np.float32)

        for row_i, seq_idx in enumerate(idx_arr):
            s = int(self.offsets[seq_idx])
            e = int(self.offsets[seq_idx + 1])
            L = e - s
            aa_buf[row_i, :L] = self.aa_flat[s:e]
            s3_buf[row_i, :L] = self.three_di_flat[s:e]
            cb_buf[row_i, :L] = self.cb_coords_flat[s:e]
            cbm_buf[row_i, :L] = self.cb_mask_flat[s:e]
            geom_buf[row_i, :L] = self.geom_feats_flat[s:e]

        return {
            "aa_tokens": torch.from_numpy(aa_buf).to(device=device, dtype=torch.long),
            "struct_tokens": torch.from_numpy(s3_buf).to(device=device, dtype=torch.long),
            "cb_coords": torch.from_numpy(cb_buf).to(device=device, dtype=torch.float32),
            "cb_mask": torch.from_numpy(cbm_buf).to(device=device, dtype=torch.bool),
            "geom_feats": torch.from_numpy(geom_buf).to(device=device, dtype=torch.float32),
            "lengths": torch.from_numpy(lens.astype(np.int64, copy=False)).to(
                device=device, dtype=torch.long
            ),
        }

    @classmethod
    def from_records(cls, records: Sequence[MultimodalDomainRecord]) -> PackedMultimodalStore:
        n_seq = len(records)
        lengths = np.asarray([r.length for r in records], dtype=np.int32)
        offsets = np.zeros(n_seq + 1, dtype=np.int64)
        if n_seq > 0:
            np.cumsum(lengths, out=offsets[1:])
            aa_flat = np.concatenate([encode_aa_sequence(r.sequence) for r in records]).astype(
                np.uint8, copy=False
            )
            s3_flat = np.concatenate([encode_3di_sequence(r.three_di) for r in records]).astype(
                np.uint8, copy=False
            )
            cb_flat = np.concatenate([np.asarray(r.cb_coords, dtype=np.float32) for r in records], axis=0)
            cbm_flat = np.concatenate([np.asarray(r.cb_mask, dtype=np.bool_) for r in records], axis=0)
            geom_flat = np.concatenate(
                [np.asarray(r.geom_feats, dtype=np.float32) for r in records], axis=0
            )
        else:
            aa_flat = np.zeros(0, dtype=np.uint8)
            s3_flat = np.zeros(0, dtype=np.uint8)
            cb_flat = np.zeros((0, 3), dtype=np.float32)
            cbm_flat = np.zeros(0, dtype=np.bool_)
            geom_flat = np.zeros((0, GEOM_FEAT_DIM), dtype=np.float32)
        return cls(
            aa_flat=aa_flat,
            three_di_flat=s3_flat,
            cb_coords_flat=cb_flat,
            cb_mask_flat=cbm_flat,
            geom_feats_flat=geom_flat,
            offsets=offsets,
            lengths=lengths,
            ids=np.asarray([r.domain_id for r in records], dtype=str),
            cath_h=np.asarray([r.cath_h for r in records], dtype=str),
            cath_t=np.asarray([r.cath_t for r in records], dtype=str),
        )


def save_packed_multimodal_npz(
    path: Path,
    records: Sequence[MultimodalDomainRecord],
) -> dict[str, Any]:
    """Atomically save paired multimodal records to a compressed .npz file."""
    store = PackedMultimodalStore.from_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(
            tmp_path,
            aa_flat=store.aa_flat,
            three_di_flat=store.three_di_flat,
            cb_coords_flat=store.cb_coords_flat,
            cb_mask_flat=store.cb_mask_flat,
            geom_feats_flat=store.geom_feats_flat,
            offsets=store.offsets,
            lengths=store.lengths,
            ids=store.ids if store.ids is not None else np.zeros(0, dtype=str),
            cath_h=store.cath_h if store.cath_h is not None else np.zeros(0, dtype=str),
            cath_t=store.cath_t if store.cath_t is not None else np.zeros(0, dtype=str),
        )
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return {
        "file": path.name,
        "n_sequences": int(store.n_sequences),
        "total_tokens": int(store.total_tokens),
        "n_distinct_cath_h": len({r.cath_h for r in records if r.cath_h}),
        "n_distinct_cath_t": len({r.cath_t for r in records if r.cath_t}),
        "size_bytes": int(path.stat().st_size),
    }


def load_packed_multimodal_npz(path: Path) -> PackedMultimodalStore:
    """Load a compressed multimodal .npz split."""
    with np.load(path, allow_pickle=False) as data:
        return PackedMultimodalStore(
            aa_flat=np.asarray(data["aa_flat"], dtype=np.uint8),
            three_di_flat=np.asarray(data["three_di_flat"], dtype=np.uint8),
            cb_coords_flat=np.asarray(data["cb_coords_flat"], dtype=np.float32),
            cb_mask_flat=np.asarray(data["cb_mask_flat"], dtype=np.bool_),
            geom_feats_flat=np.asarray(data["geom_feats_flat"], dtype=np.float32),
            offsets=np.asarray(data["offsets"], dtype=np.int64),
            lengths=np.asarray(data["lengths"], dtype=np.int32),
            ids=np.asarray(data["ids"], dtype=str) if "ids" in data else None,
            cath_h=np.asarray(data["cath_h"], dtype=str) if "cath_h" in data else None,
            cath_t=np.asarray(data["cath_t"], dtype=str) if "cath_t" in data else None,
        )


def build_fold_disjoint_multimodal_splits(
    records: Sequence[MultimodalDomainRecord],
    *,
    unseen_fold_frac: float = 0.15,
    unseen_homolog_frac: float = 0.10,
    n_train_sub_eval: int = 512,
    excluded_seq_hashes: set[str] | frozenset[str] = frozenset(),
    seed: int = 20261128,
) -> dict[str, Any]:
    """Partition paired 3D domains into strict C.A.T fold-disjoint train / unseen_homolog / unseen_fold splits."""
    rng = np.random.default_rng(int(seed))
    seen_hashes: set[str] = set()
    clean: list[MultimodalDomainRecord] = []
    for rec in records:
        if not rec.cath_h:
            continue
        sh = seq_sha256(rec.sequence)
        if sh in seen_hashes or sh in excluded_seq_hashes:
            continue
        seen_hashes.add(sh)
        cath_t = rec.cath_t or cath_topology_of(rec.cath_h)
        clean.append(
            MultimodalDomainRecord(
                domain_id=rec.domain_id,
                sequence=rec.sequence,
                three_di=rec.three_di,
                cb_coords=rec.cb_coords,
                cb_mask=rec.cb_mask,
                geom_feats=rec.geom_feats,
                cath_h=rec.cath_h,
                cath_t=cath_t,
            )
        )

    by_topo: dict[str, list[MultimodalDomainRecord]] = defaultdict(list)
    for rec in clean:
        by_topo[rec.cath_t].append(rec)

    topos = sorted(by_topo.keys())
    if len(topos) < 2:
        raise ValueError(f"Need >= 2 distinct CATH C.A.T topologies for fold-disjoint split, got {len(topos)}")

    perm_topos = list(topos)
    rng.shuffle(perm_topos)
    n_unseen_t = max(1, min(len(perm_topos) - 1, int(round(float(unseen_fold_frac) * len(perm_topos)))))
    unseen_topo_set = set(perm_topos[:n_unseen_t])

    eval_unseen_fold: list[MultimodalDomainRecord] = []
    train_by_h: dict[str, list[MultimodalDomainRecord]] = defaultdict(list)
    for rec in clean:
        if rec.cath_t in unseen_topo_set:
            eval_unseen_fold.append(rec)
        else:
            train_by_h[rec.cath_h].append(rec)

    train_recs: list[MultimodalDomainRecord] = []
    eval_unseen_homolog: list[MultimodalDomainRecord] = []
    for h, recs in sorted(train_by_h.items()):
        shuf = list(recs)
        rng.shuffle(shuf)
        if len(shuf) >= 2:
            n_hom = max(1, int(round(float(unseen_homolog_frac) * len(shuf))))
            n_hom = min(n_hom, len(shuf) - 1)
            eval_unseen_homolog.extend(shuf[:n_hom])
            train_recs.extend(shuf[n_hom:])
        else:
            train_recs.extend(shuf)

    rng.shuffle(train_recs)
    rng.shuffle(eval_unseen_homolog)
    rng.shuffle(eval_unseen_fold)
    eval_train_sub = list(train_recs[: min(int(n_train_sub_eval), len(train_recs))])

    # Hard fold-disjointness assertion
    train_topos = {r.cath_t for r in train_recs}
    hom_topos = {r.cath_t for r in eval_unseen_homolog}
    unf_topos = {r.cath_t for r in eval_unseen_fold}
    assert train_topos.isdisjoint(unf_topos), "Leakage: unseen_fold topology present in train!"
    assert hom_topos.isdisjoint(unf_topos), "Leakage: unseen_fold topology present in unseen_homolog!"
    train_ids = {r.domain_id for r in train_recs}
    hom_ids = {r.domain_id for r in eval_unseen_homolog}
    assert train_ids.isdisjoint(hom_ids), "Leakage: train and unseen_homolog overlap!"

    return {
        "train": train_recs,
        "eval_train_sub": eval_train_sub,
        "eval_unseen_homolog": eval_unseen_homolog,
        "eval_unseen_fold": eval_unseen_fold,
        "unseen_fold_topologies": sorted(unf_topos),
        "train_topologies": sorted(train_topos),
    }


class Multimodal3DBundle:
    """Loader for a portable paired 3D multimodal dataset + downstream benchmark bundle."""

    def __init__(self, bundle_dir: Path | str) -> None:
        self.bundle_dir = Path(bundle_dir)
        manifest_path = self.bundle_dir / "bundle_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing bundle_manifest.json in {self.bundle_dir}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

    def load_splits(self) -> dict[str, PackedMultimodalStore]:
        splits_dir = self.bundle_dir / "splits"
        return {
            "train": load_packed_multimodal_npz(splits_dir / "train.npz"),
            "eval_train_sub": load_packed_multimodal_npz(splits_dir / "eval_train_sub.npz"),
            "eval_unseen_homolog": load_packed_multimodal_npz(splits_dir / "eval_unseen_homolog.npz"),
            "eval_unseen_fold": load_packed_multimodal_npz(splits_dir / "eval_unseen_fold.npz"),
        }

    def load_megascale(self) -> dict[str, Any] | None:
        p = self.bundle_dir / "benchmarks/megascale_ddg.json.gz"
        if not p.is_file():
            return None
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def load_proteingym(self) -> dict[str, Any] | None:
        p = self.bundle_dir / "benchmarks/proteingym_substitutions.json.gz"
        if not p.is_file():
            return None
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def load_contacts(self) -> list[dict[str, Any]]:
        meta_p = self.bundle_dir / "benchmarks/unseen_fold_contacts_meta.json"
        npz_p = self.bundle_dir / "benchmarks/unseen_fold_contacts.npz"
        if not (meta_p.is_file() and npz_p.is_file()):
            return []
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        out: list[dict[str, Any]] = []
        with np.load(npz_p, allow_pickle=False) as data:
            for dom in meta.get("domains", []):
                did = str(dom["domain_id"])
                entry = dict(dom)
                entry["cb_coords"] = np.asarray(data[f"cb_{did}"], dtype=np.float32)
                entry["cb_mask"] = np.asarray(data[f"mask_{did}"], dtype=np.bool_)
                if f"three_di_{did}" in data:
                    entry["three_di_tokens"] = np.asarray(data[f"three_di_{did}"], dtype=np.uint8)
                out.append(entry)
        return out
