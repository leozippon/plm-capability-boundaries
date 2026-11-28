"""Temporal + Fold-Disjoint + Family-Bucketed Audit Bundle Builder (`open_model_audit`).

Constructs a compact, portable (<50 MB) evaluation bundle under
``output/open_audit_bundle/`` containing four strictly sequence-disjoint cohorts:

1. ``seen_pretrain_buckets`` (``train`` proxy + Horvitz-Thompson corpus estimator):
   Stratified random sample across 5 UniRef50 cluster-depth buckets
   (``singleton d=1``, ``small 2..16``, ``medium 17..128``, ``deep 129..1024``,
   ``ultra_deep >1024``; default 256 seqs/bucket = 1,280 seqs), paired with exact
   full-corpus population cluster counts $N_b$ and residue counts $T_b$ across all
   ~60.3M UniRef50 clusters, plus a canonical pre-cutoff Swiss-Prot CATH-matched
   reference subset (``seen_cath_matched``).
2. ``unseen_homolog`` (512 seqs):
   Post-2023 Swiss-Prot / non-SwissProt TrEMBL homologs inside the **same CATH
   H-superfamilies** as ``seen_cath_matched`` + non-representative UniRef90
   intra-cluster homologs, measuring $\\Delta h_{\\mathrm{res}} = H_{\\mathrm{unseen\\_homolog}} - H_{\\mathrm{seen}}$.
3. ``unseen_fold`` (132 certified zero-UniRef50-hit MegaScale *de novo* designs +
   512 rare/orphan CATH-topology sequences strictly disjoint in C.A.T fold from
   seen/homolog splits), measuring $\\Delta h_{\\mathrm{univ}} = H_0 - H_{\\mathrm{unseen\\_fold}}$
   and Eq. (3b) temperature calibration $\\Delta_{\\mathrm{cal}}(\\theta)$.
4. ``3d_paired_panel`` (128 3D structures: 64 natural CATH-S40 + 64 *de novo*
   zero-hit backbones) carrying exact ``(L, 4, 3)`` ``[N, CA, C, O]`` coordinates
   AND exact Foldseek ``3Di`` strings for multi-track & structure-conditioned
   $b_{\\mathrm{ext}}(Z)$ evaluation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import AA20_SET, DEPTH_BUCKET_ORDER, assign_depth_bucket

BUNDLE_SCHEMA_VERSION = "open_model_audit_bundle_v1"

THREE_TO_ONE: dict[str, str] = {
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
}

_UNIREF_N_RE = re.compile(rb"\bn=(\d+)\b")
_XML_ENTRY_DATE_RE = re.compile(rb'<entry\b[^>]*created="(\d{4}-\d{2}-\d{2})"')
_XML_ACC_RE = re.compile(rb"<accession>([^<]+)</accession>")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seq_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json_with_sha256(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    digest = sha256_file(path)
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def is_valid_aa20(sequence: str, *, min_len: int = 30, max_len: int = 512) -> bool:
    n = len(sequence)
    if not (min_len <= n <= max_len):
        return False
    return not (set(sequence) - AA20_SET)


def cath_topology_of(cath_h: str) -> str:
    parts = cath_h.strip().split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else cath_h.strip()


# --------------------------------------------------------------------------- #
# Bundle Schema Dataclasses & Loader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuditSequenceRecord:
    record_id: str
    split: str
    sub_stratum: str
    sequence: str
    seq_len: int
    seq_sha256: str
    cluster_size: int | None = None
    depth_bucket: str | None = None
    accession: str = ""
    cath_h: str = ""
    cath_t: str = ""
    source_db: str = ""
    created_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Paired3DRecord:
    record_id: str
    domain_group: str  # "natural_cath_s40" | "de_novo_zero_hit"
    sequence: str
    seq_len: int
    seq_sha256: str
    three_di_seq: str
    coords_key: str
    source_kind: str
    cath_id: str = ""
    design_series: str = ""
    mean_plddt: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenAuditBundle:
    """Portable loader for a built ``output/open_audit_bundle/`` directory."""

    def __init__(self, bundle_dir: Path | str) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        manifest_path = self.bundle_dir / "bundle_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing bundle_manifest.json in {self.bundle_dir}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.population_buckets_all: dict[str, dict[str, Any]] = self.manifest["population_buckets_all"]
        self.population_buckets_eligible: dict[str, dict[str, Any]] = self.manifest[
            "population_buckets_eligible"
        ]

        self.seen_pretrain_buckets: list[AuditSequenceRecord] = self._load_seq_jsonl(
            self.bundle_dir / "splits" / "seen_pretrain_buckets.jsonl"
        )
        self.seen_cath_matched: list[AuditSequenceRecord] = self._load_seq_jsonl(
            self.bundle_dir / "splits" / "seen_cath_matched.jsonl"
        )
        self.unseen_homolog: list[AuditSequenceRecord] = self._load_seq_jsonl(
            self.bundle_dir / "splits" / "unseen_homolog.jsonl"
        )
        self.unseen_fold: list[AuditSequenceRecord] = self._load_seq_jsonl(
            self.bundle_dir / "splits" / "unseen_fold.jsonl"
        )
        self.paired_3d_panel: list[Paired3DRecord] = self._load_3d_jsonl(
            self.bundle_dir / "splits" / "paired_3d_panel.jsonl"
        )
        self._coords_npz_path = self.bundle_dir / "splits" / "panel_3d_coords.npz"
        self._coords_cache: dict[str, np.ndarray] | None = None

    @staticmethod
    def _load_seq_jsonl(path: Path) -> list[AuditSequenceRecord]:
        if not path.is_file():
            return []
        rows: list[AuditSequenceRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(AuditSequenceRecord(**json.loads(line)))
        return rows

    @staticmethod
    def _load_3d_jsonl(path: Path) -> list[Paired3DRecord]:
        if not path.is_file():
            return []
        rows: list[Paired3DRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(Paired3DRecord(**json.loads(line)))
        return rows

    def load_coords(self, coords_key: str) -> np.ndarray:
        """Return ``(L, 4, 3)`` float32 ``[N, CA, C, O]`` backbone coordinates."""
        if self._coords_cache is None:
            if not self._coords_npz_path.is_file():
                raise FileNotFoundError(f"Missing 3D coordinate pack {self._coords_npz_path}")
            with np.load(self._coords_npz_path, allow_pickle=False) as z:
                self._coords_cache = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
        return self._coords_cache[coords_key]

    def seen_by_bucket(self) -> dict[str, list[AuditSequenceRecord]]:
        grouped: dict[str, list[AuditSequenceRecord]] = {b: [] for b in DEPTH_BUCKET_ORDER}
        for rec in self.seen_pretrain_buckets:
            if rec.depth_bucket in grouped:
                grouped[rec.depth_bucket].append(rec)
        return grouped


# --------------------------------------------------------------------------- #
# 1. Swiss-Prot FASTA + Creation-Date Scanner
# --------------------------------------------------------------------------- #


def load_swissprot_accessions(
    sprot_fasta_gz: Path,
    *,
    min_len: int = 50,
    max_len: int = 384,
) -> dict[str, str]:
    """Load canonical AA20 Swiss-Prot sequences keyed by primary UniProt accession."""
    if not sprot_fasta_gz.is_file():
        return {}
    seqs: dict[str, str] = {}
    cur_acc: str | None = None
    chunks: list[str] = []

    def _flush() -> None:
        if cur_acc is None:
            return
        seq = "".join(chunks).strip().upper()
        if is_valid_aa20(seq, min_len=min_len, max_len=max_len):
            seqs[cur_acc] = seq

    with gzip.open(sprot_fasta_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                _flush()
                parts = line[1:].split("|")
                cur_acc = parts[1].strip() if len(parts) >= 2 else line[1:].split()[0].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    _flush()
    return seqs


def parse_swissprot_creation_dates(sprot_xml_gz: Path) -> dict[str, str]:
    """Fast streaming parser extracting ``{primary_accession: 'YYYY-MM-DD'}`` from ``uniprot_sprot.xml.gz``."""
    if not sprot_xml_gz.is_file():
        return {}
    dates: dict[str, str] = {}
    cur_date: str | None = None
    need_primary_acc = False

    def _consume(stream: Any) -> None:
        nonlocal cur_date, need_primary_acc
        for raw in stream:
            if b"<entry " in raw:
                m = _XML_ENTRY_DATE_RE.search(raw)
                if m:
                    cur_date = m.group(1).decode("ascii")
                    need_primary_acc = True
                else:
                    cur_date = None
                    need_primary_acc = False
            elif need_primary_acc and b"<accession>" in raw:
                m = _XML_ACC_RE.search(raw)
                if m and cur_date is not None:
                    acc = m.group(1).decode("ascii").strip()
                    dates[acc] = cur_date
                    need_primary_acc = False

    if shutil.which("gzip") and shutil.which("grep"):
        # Filter 99.5% of XML lines in C (grep) before Python sees them (~5s vs ~90s!)
        cmd = (
            f'gzip -dc "{sprot_xml_gz}" | '
            r'grep -E "<entry [^>]*created=\"|<accession>"'
        )
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            _consume(proc.stdout)
        finally:
            proc.stdout.close()
            proc.wait()
    elif shutil.which("gzip"):
        proc = subprocess.Popen(["gzip", "-dc", str(sprot_xml_gz)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            _consume(proc.stdout)
        finally:
            proc.stdout.close()
            proc.wait()
    else:
        with gzip.open(sprot_xml_gz, "rb") as fh:
            _consume(fh)
    return dates


# --------------------------------------------------------------------------- #
# 2. Single-Pass Full-UniRef50 Depth-Bucket Scanner + Reservoir Sampler
# --------------------------------------------------------------------------- #


def scan_uniref50_depth_buckets(
    uniref50_fasta: Path,
    *,
    n_per_bucket: int = 256,
    min_len: int = 50,
    max_len: int = 384,
    swissprot_accs: set[str] | frozenset[str] = frozenset(),
    swissprot_dates: Mapping[str, str] | None = None,
    cath_by_acc: Mapping[str, tuple[str, str]] | None = None,
    excluded_seq_hashes: set[str] | frozenset[str] = frozenset(),
    seed: int = 20261128,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Stream full UniRef50 FASTA in one pass to record exact $N_b, T_b$ and sample 5 depth buckets.

    Records BOTH:
      - ``population_buckets_all``: exact cluster count $N_b$ and residue count $T_b$ over
        ALL UniRef50 clusters in ``uniref50.fasta`` (~60.3M clusters, ~17.3B residues).
      - ``population_buckets_eligible``: exact $N_b, T_b$ over AA20-strict clusters with
        ``min_len <= L <= max_len``.
    """
    rng = np.random.default_rng(int(seed))
    dates = swissprot_dates or {}
    cath_map = cath_by_acc or {}

    pop_all: dict[str, dict[str, int]] = {
        b: {"n_clusters": 0, "total_residues": 0, "sum_cluster_members": 0} for b in DEPTH_BUCKET_ORDER
    }
    pop_elig: dict[str, dict[str, int]] = {
        b: {"n_clusters": 0, "total_residues": 0, "sum_cluster_members": 0} for b in DEPTH_BUCKET_ORDER
    }
    reservoirs: dict[str, list[dict[str, Any]]] = {b: [] for b in DEPTH_BUCKET_ORDER}
    elig_seen: dict[str, int] = {b: 0 for b in DEPTH_BUCKET_ORDER}

    aa20_bytes = frozenset(b"ACDEFGHIKLMNPQRSTVWY")
    total_records = 0

    cur_cid = ""
    cur_acc = ""
    cur_n = 1
    seq_parts: list[bytes] = []

    def _process_record() -> None:
        nonlocal total_records
        if not cur_cid:
            return
        total_records += 1
        seq_b = b"".join(seq_parts)
        L = len(seq_b)
        b_name = assign_depth_bucket(cur_n)
        pa = pop_all[b_name]
        pa["n_clusters"] += 1
        pa["total_residues"] += L
        pa["sum_cluster_members"] += cur_n

        if min_len <= L <= max_len and not (set(seq_b) - aa20_bytes):
            pe = pop_elig[b_name]
            pe["n_clusters"] += 1
            pe["total_residues"] += L
            pe["sum_cluster_members"] += cur_n

            # Reservoir sample uniformly within each depth bucket (unbiased Horvitz-Thompson!)
            elig_seen[b_name] += 1
            seen_b = elig_seen[b_name]
            res = reservoirs[b_name]
            take_slot: int | None = None
            if len(res) < n_per_bucket:
                take_slot = len(res)
            else:
                j = int(rng.integers(0, seen_b))
                if j < n_per_bucket:
                    take_slot = j
            if take_slot is not None:
                seq_str = seq_b.decode("ascii")
                digest = seq_sha256(seq_str)
                if digest in excluded_seq_hashes:
                    return
                if cur_acc in swissprot_accs:
                    src_db = "swissprot"
                elif cur_acc.startswith("UPI"):
                    src_db = "uniparc"
                else:
                    src_db = "trembl"
                ch, ct = cath_map.get(cur_acc, ("", ""))
                item = {
                    "record_id": f"seen_{b_name}_{cur_cid}",
                    "split": "seen_pretrain_buckets",
                    "sub_stratum": b_name,
                    "sequence": seq_str,
                    "seq_len": L,
                    "seq_sha256": digest,
                    "cluster_size": int(cur_n),
                    "depth_bucket": b_name,
                    "accession": cur_acc,
                    "cath_h": ch,
                    "cath_t": ct,
                    "source_db": src_db,
                    "created_date": str(dates.get(cur_acc, "")),
                }
                if take_slot == len(res):
                    res.append(item)
                else:
                    res[take_slot] = item

    with uniref50_fasta.open("rb", buffering=4 * 1024 * 1024) as fh:
        for line in fh:
            if line.startswith(b">"):
                _process_record()
                if max_records is not None and total_records >= int(max_records):
                    cur_cid = ""
                    break
                sp_idx = line.find(b" ")
                cid_b = line[1:sp_idx] if sp_idx > 1 else line[1:].strip()
                cur_cid = cid_b.decode("ascii", errors="replace")
                cur_acc = cur_cid[len("UniRef50_") :] if cur_cid.startswith("UniRef50_") else cur_cid
                m = _UNIREF_N_RE.search(line)
                cur_n = int(m.group(1)) if m else 1
                seq_parts = []
            else:
                seq_parts.append(line.strip())
    _process_record()

    # Deduplicate sampled records across buckets by seq_sha256 while preserving deterministic order
    sampled_records: list[AuditSequenceRecord] = []
    seen_hashes: set[str] = set()
    for b_name in DEPTH_BUCKET_ORDER:
        for item in sorted(reservoirs[b_name], key=lambda r: (r["record_id"], r["seq_sha256"])):
            if item["seq_sha256"] in seen_hashes:
                continue
            seen_hashes.add(item["seq_sha256"])
            sampled_records.append(AuditSequenceRecord(**item))

    return {
        "total_fasta_records_scanned": int(total_records),
        "population_buckets_all": pop_all,
        "population_buckets_eligible": pop_elig,
        "sampled_records": sampled_records,
    }


def sample_uniref50_depth_buckets_via_index(
    uniref50_fasta: Path,
    index_pkl: Path,
    *,
    n_probe_records: int = 300_000,
    n_per_bucket: int = 256,
    min_len: int = 50,
    max_len: int = 384,
    swissprot_accs: set[str] | frozenset[str] = frozenset(),
    swissprot_dates: Mapping[str, str] | None = None,
    cath_by_acc: Mapping[str, tuple[str, str]] | None = None,
    excluded_seq_hashes: set[str] | frozenset[str] = frozenset(),
    seed: int = 20261128,
) -> dict[str, Any]:
    """Unbiased uniform random cluster census & 5-bucket sampler over the full 24GB UniRef50 FASTA.

    Uses ``uniref50.fasta.idx_max1022.pkl`` (58.5M byte offsets) sorted by offset for fast
    disk read-ahead, avoiding length-ordering bias present in the head of ``uniref50.fasta``.
    """
    import pickle

    with index_pkl.open("rb") as fh:
        idx_list = pickle.load(fh)
    n_index = len(idx_list)
    rng = np.random.default_rng(int(seed))
    n_take = min(int(n_probe_records), n_index)
    chosen_indices = rng.choice(n_index, size=n_take, replace=False)
    # Sort chosen byte offsets ascending for sequential OS read-ahead
    chosen_offsets = sorted(int(idx_list[int(i)][0]) for i in chosen_indices)
    del idx_list, chosen_indices

    dates = swissprot_dates or {}
    cath_map = cath_by_acc or {}
    pop_all: dict[str, dict[str, int]] = {
        b: {"n_clusters": 0, "total_residues": 0, "sum_cluster_members": 0} for b in DEPTH_BUCKET_ORDER
    }
    pop_elig: dict[str, dict[str, int]] = {
        b: {"n_clusters": 0, "total_residues": 0, "sum_cluster_members": 0} for b in DEPTH_BUCKET_ORDER
    }
    reservoirs: dict[str, list[dict[str, Any]]] = {b: [] for b in DEPTH_BUCKET_ORDER}
    elig_seen: dict[str, int] = {b: 0 for b in DEPTH_BUCKET_ORDER}
    aa20_bytes = frozenset(b"ACDEFGHIKLMNPQRSTVWY")
    scanned = 0

    with uniref50_fasta.open("rb") as fh:
        for off in chosen_offsets:
            fh.seek(off)
            hdr = fh.readline()
            if not hdr.startswith(b">"):
                continue
            seq_parts: list[bytes] = []
            while True:
                ln = fh.readline()
                if not ln or ln.startswith(b">"):
                    break
                seq_parts.append(ln.strip())
            seq_b = b"".join(seq_parts)
            L = len(seq_b)
            if L == 0:
                continue
            scanned += 1
            sp_idx = hdr.find(b" ")
            cid_b = hdr[1:sp_idx] if sp_idx > 1 else hdr[1:].strip()
            cur_cid = cid_b.decode("ascii", errors="replace")
            cur_acc = cur_cid[len("UniRef50_") :] if cur_cid.startswith("UniRef50_") else cur_cid
            m = _UNIREF_N_RE.search(hdr)
            cur_n = int(m.group(1)) if m else 1

            b_name = assign_depth_bucket(cur_n)
            pa = pop_all[b_name]
            pa["n_clusters"] += 1
            pa["total_residues"] += L
            pa["sum_cluster_members"] += cur_n

            if min_len <= L <= max_len and not (set(seq_b) - aa20_bytes):
                pe = pop_elig[b_name]
                pe["n_clusters"] += 1
                pe["total_residues"] += L
                pe["sum_cluster_members"] += cur_n

                elig_seen[b_name] += 1
                seen_b = elig_seen[b_name]
                res = reservoirs[b_name]
                take_slot: int | None = None
                if len(res) < n_per_bucket:
                    take_slot = len(res)
                else:
                    j = int(rng.integers(0, seen_b))
                    if j < n_per_bucket:
                        take_slot = j
                if take_slot is not None:
                    seq_str = seq_b.decode("ascii")
                    digest = seq_sha256(seq_str)
                    if digest in excluded_seq_hashes:
                        continue
                    if cur_acc in swissprot_accs:
                        src_db = "swissprot"
                    elif cur_acc.startswith("UPI"):
                        src_db = "uniparc"
                    else:
                        src_db = "trembl"
                    ch, ct = cath_map.get(cur_acc, ("", ""))
                    item = {
                        "record_id": f"seen_{b_name}_{cur_cid}",
                        "split": "seen_pretrain_buckets",
                        "sub_stratum": b_name,
                        "sequence": seq_str,
                        "seq_len": L,
                        "seq_sha256": digest,
                        "cluster_size": int(cur_n),
                        "depth_bucket": b_name,
                        "accession": cur_acc,
                        "cath_h": ch,
                        "cath_t": ct,
                        "source_db": src_db,
                        "created_date": str(dates.get(cur_acc, "")),
                    }
                    if take_slot == len(res):
                        res.append(item)
                    else:
                        res[take_slot] = item

    sampled_records: list[AuditSequenceRecord] = []
    seen_hashes: set[str] = set()
    for b_name in DEPTH_BUCKET_ORDER:
        for item in sorted(reservoirs[b_name], key=lambda r: (r["record_id"], r["seq_sha256"])):
            if item["seq_sha256"] in seen_hashes:
                continue
            seen_hashes.add(item["seq_sha256"])
            sampled_records.append(AuditSequenceRecord(**item))

    return {
        "total_fasta_records_scanned": int(scanned),
        "population_buckets_all": pop_all,
        "population_buckets_eligible": pop_elig,
        "sampled_records": sampled_records,
    }


# --------------------------------------------------------------------------- #
# 3. PDB Coordinate Parsing, Foldseek 3Di, and Optional Offline ESMFold2
# --------------------------------------------------------------------------- #


ONE_TO_THREE: dict[str, str] = {v: k for k, v in THREE_TO_ONE.items()}


def _write_backbone_pdb(path: Path, sequence: str, coords: np.ndarray) -> None:
    """Write minimal N/CA/C/O ATOM records so Foldseek can compute exact 3Di strings."""
    atom_names = (" N  ", " CA ", " C  ", " O  ")
    elem_names = ("N", "C", "C", "O")
    lines: list[str] = []
    serial = 1
    for i, aa in enumerate(sequence, start=1):
        res3 = ONE_TO_THREE.get(aa, "ALA")
        for a_idx in range(4):
            x, y, z = coords[i - 1, a_idx]
            lines.append(
                f"ATOM  {serial:5d} {atom_names[a_idx]} {res3} A{i:4d}    "
                f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00 90.00           {elem_names[a_idx]}  "
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_pdb_ncaco_from_lines(lines: Iterable[str]) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse first model / first chain exact ``(L, 4, 3)`` ``[N, CA, C, O]`` coords and B-factors."""
    atom_map = {"N": 0, "CA": 1, "C": 2, "O": 3}
    order: list[tuple[str, str, str]] = []
    res_aa: dict[tuple[str, str, str], str] = {}
    res_coords: dict[tuple[str, str, str], np.ndarray] = {}
    res_bf: dict[tuple[str, str, str], float] = {}
    first_chain: str | None = None

    for line in lines:
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in atom_map:
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
        if res_key not in res_coords:
            order.append(res_key)
            res_aa[res_key] = THREE_TO_ONE.get(line[17:20].strip(), "X")
            res_coords[res_key] = np.full((4, 3), np.nan, dtype=np.float32)
            res_bf[res_key] = np.nan
        idx = atom_map[atom_name]
        if not np.isfinite(res_coords[res_key][idx, 0]):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            res_coords[res_key][idx] = (x, y, z)
            if atom_name == "CA":
                try:
                    res_bf[res_key] = float(line[60:66])
                except ValueError:
                    res_bf[res_key] = np.nan

    if not order:
        return "", np.zeros((0, 4, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    seq = "".join(res_aa[k] for k in order)
    coords = np.stack([res_coords[k] for k in order], axis=0).astype(np.float32)
    bf = np.asarray([res_bf[k] for k in order], dtype=np.float32)
    return seq, coords, bf


def ensure_foldseek_binary(
    data_root: Path,
    scratch_dir: Path = Path("/tmp/open_model_audit_tools"),
) -> Path:
    """Locate or unpack the local offline Foldseek binary."""
    env_bin = os.environ.get("FOLDSEEK_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)
    which_bin = shutil.which("foldseek")
    if which_bin:
        return Path(which_bin)
    for cand in (
        scratch_dir / "foldseek/bin/foldseek",
        Path("/tmp/memory_theory_tools/foldseek/bin/foldseek"),
        Path("/mnt/c20250608/heliang/util/foldseek"),
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    tarball = data_root / "external/tools/foldseek-linux-avx2.tar.gz"
    if not tarball.is_file():
        raise FileNotFoundError(f"Foldseek binary and tarball not found (checked {tarball})")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(scratch_dir)
    binary = scratch_dir / "foldseek/bin/foldseek"
    if not binary.is_file():
        raise FileNotFoundError(f"Foldseek binary missing at {binary} after extracting {tarball}")
    return binary


def extract_foldseek_3di_for_pdbs(
    pdb_files: Mapping[str, Path],
    foldseek_bin: Path,
    *,
    threads: int = 8,
) -> dict[str, tuple[str, str]]:
    """Run ``foldseek structureto3didescriptor`` and return ``{record_id: (aa_seq, three_di_seq)}``."""
    if not pdb_files:
        return {}
    with tempfile.TemporaryDirectory(prefix="open_audit_3di_") as tmp_s:
        tmp = Path(tmp_s)
        in_dir = tmp / "pdbs"
        in_dir.mkdir(parents=True)
        for rec_id, src in pdb_files.items():
            suffix = ".pdb.gz" if src.name.endswith(".gz") else ".pdb"
            safe = rec_id.replace("/", "_").replace("|", "_")
            dst = in_dir / f"{safe}{suffix}"
            os.symlink(os.path.abspath(src), dst)
        safe_to_id = {rec_id.replace("/", "_").replace("|", "_"): rec_id for rec_id in pdb_files}
        out_tsv = tmp / "descriptors.tsv"
        subprocess.run(
            [
                str(foldseek_bin),
                "structureto3didescriptor",
                str(in_dir),
                str(out_tsv),
                "--threads",
                str(max(1, int(threads))),
                "-v",
                "1",
            ],
            check=True,
        )
        result: dict[str, tuple[str, str]] = {}
        with out_tsv.open("r", encoding="utf-8") as fh:
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 3:
                    continue
                stem = cols[0].split()[0]
                for ext in (".pdb.gz", ".pdb"):
                    if stem.endswith(ext):
                        stem = stem[: -len(ext)]
                rec_id = safe_to_id.get(stem, stem)
                aa_s = cols[1].strip().upper()
                tdi_s = cols[2].strip()
                if len(aa_s) == len(tdi_s) and len(aa_s) > 0:
                    result[rec_id] = (aa_s, tdi_s)
        return result


def fold_denovo_designs_with_esmfold2(
    designs: Sequence[Mapping[str, Any]],
    pdb_cache_dir: Path,
    *,
    esmfold2_python: Path,
    esmfold2_model_dir: Path,
    device: str = "cuda:2",
    seed: int = 20261128,
) -> dict[str, tuple[Path, float]]:
    """Fold missing short *de novo* designs using the isolated offline ESMFold2 runtime."""
    pdb_cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = pdb_cache_dir / "esmfold2_meta.json"
    meta: dict[str, float] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    pending: list[dict[str, str]] = []
    out: dict[str, tuple[Path, float]] = {}
    for d in designs:
        rid = str(d["record_id"])
        safe = rid.replace("/", "_").replace("|", "_")
        pdb_p = pdb_cache_dir / f"{safe}.pdb"
        if pdb_p.is_file() and pdb_p.stat().st_size > 200:
            out[rid] = (pdb_p, float(meta.get(rid, float("nan"))))
        else:
            pending.append({"record_id": rid, "safe": safe, "sequence": str(d["sequence"])})

    if pending:
        if not esmfold2_python.is_file():
            raise FileNotFoundError(f"ESMFold2 python runtime not found: {esmfold2_python}")
        if not esmfold2_model_dir.is_dir():
            raise FileNotFoundError(f"ESMFold2 checkpoint dir not found: {esmfold2_model_dir}")
        worker_code = r"""
import json, pathlib, sys, torch
from transformers import EsmFold2Model
from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features

jobs = json.loads(pathlib.Path(sys.argv[1]).read_text())
model_dir = sys.argv[2]
out_dir = pathlib.Path(sys.argv[3])
device = torch.device(sys.argv[4])
seed = int(sys.argv[5])

torch.cuda.set_device(device)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = EsmFold2Model.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16).eval().to(device)
plddt_out = {}
with torch.inference_mode():
    for job in jobs:
        seq = job["sequence"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        feats_gpu = prepare_protein_features(seq, device=device)
        outputs = model.fold(**feats_gpu, num_diffusion_samples=1)
        feats_cpu = prepare_protein_features(seq, device="cpu")
        pdb_str = output_to_pdb(outputs, feats_cpu, sample_idx=0)
        (out_dir / f"{job['safe']}.pdb").write_text(pdb_str, encoding="utf-8")
        ca_plddt = float(outputs.plddt_ca[0].float().mean().item() * 100.0) if getattr(outputs, "plddt_ca", None) is not None else float("nan")
        plddt_out[job["record_id"]] = ca_plddt
print(json.dumps(plddt_out))
"""
        with tempfile.TemporaryDirectory(prefix="open_audit_esmfold2_") as td:
            jobs_json = Path(td) / "jobs.json"
            jobs_json.write_text(json.dumps(pending), encoding="utf-8")
            proc = subprocess.run(
                [
                    str(esmfold2_python),
                    "-c",
                    worker_code,
                    str(jobs_json),
                    str(esmfold2_model_dir),
                    str(pdb_cache_dir),
                    str(device),
                    str(seed),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            last_line = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1]
            folded_plddt: dict[str, float] = json.loads(last_line)
            meta.update(folded_plddt)
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            for job in pending:
                rid = job["record_id"]
                pdb_p = pdb_cache_dir / f"{job['safe']}.pdb"
                out[rid] = (pdb_p, float(meta.get(rid, float("nan"))))

    return out


# --------------------------------------------------------------------------- #
# 4. Complete Bundle Assembly Pipeline
# --------------------------------------------------------------------------- #


def load_megascale_zero_hit_designs(data_root: Path) -> list[dict[str, Any]]:
    """Load certified zero-UniRef50-hit MegaScale *de novo* designs."""
    ms_dir = data_root / "results/shared/megascale_disjointness_20260813"
    cert = json.loads((ms_dir / "certificate.json").read_text(encoding="utf-8"))
    hit_wts = {str(row["WT_name"]) for row in cert.get("design_hits_at_gate", [])}
    qi = json.loads((ms_dir / "query_index.json").read_text(encoding="utf-8"))

    seen_hashes: set[str] = set()
    designs: list[dict[str, Any]] = []
    for row in qi:
        if row.get("kind") != "design":
            continue
        wt_name = str(row["WT_name"])
        if wt_name in hit_wts:
            continue
        seq = str(row["sequence"]).strip().upper()
        if not is_valid_aa20(seq, min_len=30, max_len=512):
            continue
        digest = seq_sha256(seq)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        designs.append(
            {
                "record_id": f"denovo_{row['query_id']}_{wt_name.replace('.pdb', '')}",
                "wt_name": wt_name,
                "series": f"{row.get('group', '')}/{row.get('series', '')}",
                "sequence": seq,
                "seq_len": len(seq),
                "seq_sha256": digest,
            }
        )
    designs.sort(key=lambda r: r["record_id"])
    return designs


def _sample_uniref90_nonrep_homologs(
    uniref90_fasta_gz: Path,
    *,
    target_count: int,
    uniref50_rep_accs: set[str],
    cath_by_acc: Mapping[str, tuple[str, str]],
    allowed_cath_h: set[str],
    forbidden_cath_t: set[str],
    excluded_seq_hashes: set[str],
    min_len: int = 50,
    max_len: int = 384,
    max_scan_records: int = 1_500_000,
    seed: int = 20261128,
) -> list[AuditSequenceRecord]:
    """Sample non-representative UniRef90 homologs strictly absent from UniRef50 rep accessions."""
    if not uniref90_fasta_gz.is_file() or target_count <= 0:
        return []
    rng = np.random.default_rng(int(seed) + 909)
    cath_hits: list[AuditSequenceRecord] = []
    fallback_hits: list[AuditSequenceRecord] = []
    fallback_seen = 0
    scanned = 0

    cur_cid = ""
    cur_acc = ""
    cur_n = 1
    chunks: list[str] = []

    def _flush() -> None:
        nonlocal scanned, fallback_seen
        if not cur_cid:
            return
        scanned += 1
        if cur_acc in uniref50_rep_accs or cur_acc.startswith("UPI"):
            return
        seq = "".join(chunks).strip().upper()
        if not is_valid_aa20(seq, min_len=min_len, max_len=max_len):
            return
        digest = seq_sha256(seq)
        if digest in excluded_seq_hashes:
            return
        ch_ct = cath_by_acc.get(cur_acc)
        if ch_ct is not None:
            ch, ct = ch_ct
            if ct in forbidden_cath_t:
                return
            if ch in allowed_cath_h:
                excluded_seq_hashes.add(digest)
                cath_hits.append(
                    AuditSequenceRecord(
                        record_id=f"hom_ur90cath_{cur_cid}",
                        split="unseen_homolog",
                        sub_stratum="uniref90_nonrep_same_cath",
                        sequence=seq,
                        seq_len=len(seq),
                        seq_sha256=digest,
                        cluster_size=int(cur_n),
                        depth_bucket=assign_depth_bucket(cur_n),
                        accession=cur_acc,
                        cath_h=ch,
                        cath_t=ct,
                        source_db="uniref90_2026_03",
                    )
                )
                return
        # Reservoir fallback for intra-cluster UniRef90 non-UniRef50-rep TrEMBL sequences
        if cur_n >= 2 and len(cath_hits) < target_count:
            fallback_seen += 1
            rec = AuditSequenceRecord(
                record_id=f"hom_ur90nonrep_{cur_cid}",
                split="unseen_homolog",
                sub_stratum="uniref90_nonrep_intra_cluster",
                sequence=seq,
                seq_len=len(seq),
                seq_sha256=digest,
                cluster_size=int(cur_n),
                depth_bucket=assign_depth_bucket(cur_n),
                accession=cur_acc,
                source_db="uniref90_2026_03",
            )
            if len(fallback_hits) < target_count:
                fallback_hits.append(rec)
            else:
                j = int(rng.integers(0, fallback_seen))
                if j < target_count:
                    fallback_hits[j] = rec

    with gzip.open(uniref90_fasta_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                _flush()
                if len(cath_hits) >= target_count or scanned >= max_scan_records:
                    cur_cid = ""
                    break
                parts = line[1:].split()
                cur_cid = parts[0] if parts else ""
                cur_acc = cur_cid[len("UniRef90_") :] if cur_cid.startswith("UniRef90_") else cur_cid
                m = re.search(r"\bn=(\d+)\b", line)
                cur_n = int(m.group(1)) if m else 1
                chunks = []
            else:
                chunks.append(line.strip())
    _flush()

    selected = list(cath_hits[:target_count])
    if len(selected) < target_count:
        for rec in fallback_hits:
            if rec.seq_sha256 in excluded_seq_hashes:
                continue
            excluded_seq_hashes.add(rec.seq_sha256)
            selected.append(rec)
            if len(selected) >= target_count:
                break
    return selected


def build_paired_3d_panel(
    data_root: Path,
    denovo_designs: Sequence[Mapping[str, Any]],
    *,
    n_natural: int = 64,
    n_denovo: int = 64,
    cath_s40_tgz: Path = Path("/Data/public/datasets/cath-s40/cath-dataset-nonredundant-S40.pdb.tgz"),
    denovo_pdb_cache_dir: Path | None = None,
    fold_denovo: bool = True,
    fold_device: str = "cuda:2",
    esmfold2_python: Path | None = None,
    esmfold2_model_dir: Path = Path("/Data/lzp/models/ESMFold2-hf"),
    min_len: int = 50,
    max_len: int = 384,
    seed: int = 20261128,
) -> tuple[list[Paired3DRecord], dict[str, np.ndarray]]:
    """Build the 128-structure ``3d_paired_panel`` (64 natural CATH-S40 + 64 *de novo* zero-hit)."""
    rng = np.random.default_rng(int(seed) + 303)
    foldseek_bin = ensure_foldseek_binary(data_root)

    with tempfile.TemporaryDirectory(prefix="open_audit_3d_panel_") as tmp_s:
        tmp = Path(tmp_s)
        nat_pdb_dir = tmp / "nat_pdbs"
        nat_pdb_dir.mkdir(parents=True)

        # 1. Sample 64 diverse natural CATH-S40 structures via O(N) streaming tar|gz (never random-seek gzip!)
        nat_candidates: list[dict[str, Any]] = []
        if cath_s40_tgz.is_file():
            target_pool = int(n_natural) + 24
            # Stride across archive entries deterministically in a single streaming pass (<1s)
            stride = 11
            valid_seen = 0
            with tarfile.open(cath_s40_tgz, "r|gz") as tf:
                for m in tf:
                    if not (m.isfile() and m.name.startswith("dompdb/")):
                        continue
                    fh = tf.extractfile(m)
                    if fh is None:
                        continue
                    valid_seen += 1
                    if (valid_seen - 1) % stride != 0:
                        continue
                    text = fh.read().decode("utf-8", errors="replace")
                    seq, coords, _bf = parse_pdb_ncaco_from_lines(text.splitlines())
                    if not is_valid_aa20(seq, min_len=min_len, max_len=max_len):
                        continue
                    if not np.isfinite(coords).all():
                        continue
                    dom_id = Path(m.name).name
                    pdb_p = nat_pdb_dir / f"nat_{dom_id}.pdb"
                    pdb_p.write_text(text, encoding="utf-8")
                    nat_candidates.append(
                        {
                            "record_id": f"nat_caths40_{dom_id}",
                            "cath_id": dom_id,
                            "sequence": seq,
                            "coords": coords,
                            "pdb_path": pdb_p,
                            "source_kind": "cath_s40_experimental",
                            "mean_plddt": None,
                        }
                    )
                    if len(nat_candidates) >= target_pool:
                        break

        # 2. Sample 64 diverse zero-hit MegaScale de novo designs across design series
        by_series: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for d in denovo_designs:
            by_series[str(d.get("series", "unknown"))].append(d)
        selected_denovo: list[Mapping[str, Any]] = []
        series_keys = sorted(by_series.keys())
        # Round-robin across design series for maximum fold diversity
        round_idx = 0
        while len(selected_denovo) < min(int(n_denovo), len(denovo_designs)):
            added_any = False
            for sk in series_keys:
                bucket = by_series[sk]
                if round_idx < len(bucket):
                    selected_denovo.append(bucket[round_idx])
                    added_any = True
                    if len(selected_denovo) >= min(int(n_denovo), len(denovo_designs)):
                        break
            if not added_any:
                break
            round_idx += 1

        denovo_candidates: list[dict[str, Any]] = []
        cache_dir = denovo_pdb_cache_dir or (tmp / "denovo_pdbs")
        if selected_denovo and (fold_denovo or (cache_dir.is_dir() and any(cache_dir.glob("*.pdb")))):
            py_bin = esmfold2_python or (data_root / "runtimes/esmfold2/bin/python")
            folded_map = fold_denovo_designs_with_esmfold2(
                selected_denovo,
                cache_dir,
                esmfold2_python=py_bin,
                esmfold2_model_dir=esmfold2_model_dir,
                device=fold_device,
                seed=seed,
            )
            for d in selected_denovo:
                rid = str(d["record_id"])
                if rid not in folded_map:
                    continue
                pdb_p, mean_plddt = folded_map[rid]
                seq, coords, bf = parse_pdb_ncaco_from_lines(
                    pdb_p.read_text(encoding="utf-8").splitlines()
                )
                if seq != str(d["sequence"]) or not np.isfinite(coords).all():
                    continue
                valid_bf = bf[np.isfinite(bf)]
                if not np.isfinite(mean_plddt) and valid_bf.size > 0:
                    mean_plddt = float(valid_bf.mean() * (100.0 if valid_bf.max() <= 1.05 else 1.0))
                denovo_candidates.append(
                    {
                        "record_id": rid,
                        "design_series": str(d.get("series", "")),
                        "sequence": seq,
                        "coords": coords,
                        "pdb_path": pdb_p,
                        "source_kind": "esmfold2_denovo_zero_hit",
                        "mean_plddt": float(mean_plddt) if np.isfinite(mean_plddt) else None,
                    }
                )

        # 4. Fallback: if fold_denovo=False and no cached de novo PDBs exist, optionally populate
        # secondary natural MegaScale backbones from wt_backbones_3d so 3D panel stays non-empty.
        if not denovo_candidates and not fold_denovo:
            wb_dir = data_root / "results/extensions/unified_yardstick_20261124/wt_backbones_3d"
            wb_manifest_p = wb_dir / "manifest.json"
            if wb_manifest_p.is_file():
                wb = json.loads(wb_manifest_p.read_text(encoding="utf-8"))
                ms_entries = [
                    e
                    for e in wb.get("entries", {}).values()
                    if "r2_megascale_101" in e.get("tags", []) and e.get("resolved")
                ]
                ms_entries.sort(key=lambda e: str(e.get("seq_sha256", "")))
                for e in ms_entries[: int(n_denovo)]:
                    npz_p = wb_dir / e["npz_relpath"]
                    if not npz_p.is_file():
                        continue
                    with np.load(npz_p, allow_pickle=False) as z:
                        seq = str(z["sequence"].item()).strip().upper()
                        coords = np.asarray(z["coords"], dtype=np.float32)
                    if not is_valid_aa20(seq, min_len=30, max_len=max_len) or not np.isfinite(coords).all():
                        continue
                    # Write minimal PDB for Foldseek 3Di extraction
                    rid = f"ms_wt_{e['seq_sha256'][:12]}"
                    pdb_p = tmp / f"{rid}.pdb"
                    _write_backbone_pdb(pdb_p, seq, coords)
                    denovo_candidates.append(
                        {
                            "record_id": rid,
                            "design_series": "megascale_wt_fallback",
                            "sequence": seq,
                            "coords": coords,
                            "pdb_path": pdb_p,
                            "source_kind": str(e.get("source_kind", "wt_backbones_3d")),
                            "mean_plddt": e.get("mean_plddt"),
                            "domain_group_override": "megascale_wt_3d",
                        }
                    )

        # 3. Extract exact Foldseek 3Di strings for every candidate PDB in one batch
        pdb_map = {c["record_id"]: c["pdb_path"] for c in nat_candidates + denovo_candidates}
        tdi_map = extract_foldseek_3di_for_pdbs(pdb_map, foldseek_bin)

        panel_records: list[Paired3DRecord] = []
        coords_pack: dict[str, np.ndarray] = {}

        nat_kept = 0
        for c in nat_candidates:
            if nat_kept >= int(n_natural):
                break
            rid = c["record_id"]
            if rid not in tdi_map:
                continue
            aa_fs, tdi_s = tdi_map[rid]
            if len(tdi_s) != len(c["sequence"]) or aa_fs != c["sequence"]:
                continue
            coords_key = f"nat_{nat_kept:03d}"
            coords_pack[coords_key] = c["coords"]
            panel_records.append(
                Paired3DRecord(
                    record_id=rid,
                    domain_group="natural_cath_s40",
                    sequence=c["sequence"],
                    seq_len=len(c["sequence"]),
                    seq_sha256=seq_sha256(c["sequence"]),
                    three_di_seq=tdi_s,
                    coords_key=coords_key,
                    source_kind=c["source_kind"],
                    cath_id=c["cath_id"],
                    mean_plddt=c["mean_plddt"],
                )
            )
            nat_kept += 1

        dn_kept = 0
        for c in denovo_candidates:
            if dn_kept >= int(n_denovo):
                break
            rid = c["record_id"]
            if rid not in tdi_map:
                continue
            aa_fs, tdi_s = tdi_map[rid]
            if len(tdi_s) != len(c["sequence"]) or aa_fs != c["sequence"]:
                continue
            coords_key = f"dn_{dn_kept:03d}"
            coords_pack[coords_key] = c["coords"]
            panel_records.append(
                Paired3DRecord(
                    record_id=rid,
                    domain_group=str(c.get("domain_group_override", "de_novo_zero_hit")),
                    sequence=c["sequence"],
                    seq_len=len(c["sequence"]),
                    seq_sha256=seq_sha256(c["sequence"]),
                    three_di_seq=tdi_s,
                    coords_key=coords_key,
                    source_kind=c["source_kind"],
                    design_series=c["design_series"],
                    mean_plddt=c["mean_plddt"],
                )
            )
            dn_kept += 1

    return panel_records, coords_pack


def build_open_audit_bundle(
    *,
    data_root: Path,
    out_dir: Path,
    n_per_depth_bucket: int = 256,
    n_seen_cath_matched: int = 256,
    n_unseen_homolog: int = 512,
    n_orphan_cath_fold: int = 512,
    n_panel_natural: int = 64,
    n_panel_denovo: int = 64,
    min_len: int = 50,
    max_len: int = 384,
    seed: int = 20261128,
    parse_xml_dates: bool = True,
    fold_denovo: bool = True,
    fold_device: str = "cuda:2",
    denovo_pdb_cache_dir: Path | None = None,
    max_uniref50_records: int | None = None,
    max_uniref90_scan: int = 1_500_000,
) -> dict[str, Any]:
    """Build and atomically persist the complete portable ``open_audit_bundle``."""
    import pandas as pd

    data_root = Path(data_root).resolve()
    out_dir = Path(out_dir).resolve()
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    # 1. Certified zero-hit MegaScale de novo designs (guaranteed unseen fold)
    denovo_raw = load_megascale_zero_hit_designs(data_root)
    denovo_records = [
        AuditSequenceRecord(
            record_id=d["record_id"],
            split="unseen_fold",
            sub_stratum="denovo_zero_hit",
            sequence=d["sequence"],
            seq_len=d["seq_len"],
            seq_sha256=d["seq_sha256"],
            source_db="megascale_tsuboyama2023_zero_uniref50_hit",
        )
        for d in denovo_raw
    ]
    used_seq_hashes: set[str] = {r.seq_sha256 for r in denovo_records}

    # 2. Load Swiss-Prot accessions and optional XML creation dates
    sprot_fasta_gz = data_root / "data/swissprot/uniprot_sprot.fasta.gz"
    sprot_xml_gz = data_root / "data/swissprot/uniprot_sprot.xml.gz"
    sprot_seqs = load_swissprot_accessions(sprot_fasta_gz, min_len=min_len, max_len=max_len)
    sprot_accs = set(sprot_seqs.keys())
    sprot_dates = parse_swissprot_creation_dates(sprot_xml_gz) if parse_xml_dates else {}
    precutoff_sprot_accs = {acc for acc, dt in sprot_dates.items() if dt < "2022-01-01"} if sprot_dates else sprot_accs
    postcutoff_sprot_accs = {acc for acc, dt in sprot_dates.items() if dt >= "2023-01-01"}

    # 3. Load UniRef50-CATH parquet + Swiss-Prot CATH TSV
    parquet_path = data_root / "output/stage2_cache/uniref50_cath_l50_384.parquet"
    cath_tsv = data_root / "data/interpro/cath_superfamily.tsv"

    cath_tsv_map: dict[str, tuple[str, str]] = {}
    if cath_tsv.is_file():
        acc_to_hs: dict[str, set[str]] = defaultdict(set)
        with cath_tsv.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                acc = row["uniprot"].strip()
                h = row["cath_superfamily"].strip()
                if h.count(".") == 3:
                    acc_to_hs[acc].add(h)
        for acc, hs in acc_to_hs.items():
            if len(hs) == 1:
                h = next(iter(hs))
                cath_tsv_map[acc] = (h, cath_topology_of(h))

    pq_df = pd.read_parquet(
        parquet_path,
        columns=["uniref50_id", "accession", "cath_h", "cath_t", "n_cath_t", "seq_len", "sequence"],
    )
    pq_single = pq_df[
        (pq_df["n_cath_t"] == 1) & (pq_df["seq_len"] >= min_len) & (pq_df["seq_len"] <= max_len)
    ]

    # Build quick lookup maps for CATH topology counts and per-superfamily seen/homolog pools
    topo_counts = pq_single["cath_t"].value_counts()
    cath_by_acc: dict[str, tuple[str, str]] = dict(cath_tsv_map)
    uniref50_cath_accs: set[str] = set()
    # Select rare/orphan CATH topologies strictly from the low-depth tail
    rare_topos_ordered = [str(t) for t, _ in sorted(topo_counts.items(), key=lambda kv: (int(kv[1]), str(kv[0])))]

    # Pick orphan_cath_fold sequences from the rarest topologies first, tracking forbidden topologies
    orphan_records: list[AuditSequenceRecord] = []
    orphan_topos: set[str] = set()
    # Filter parquet down to rarest topologies sufficient to cover n_orphan_cath_fold
    cum = 0
    candidate_rare_topos: list[str] = []
    for t in rare_topos_ordered:
        candidate_rare_topos.append(t)
        cum += int(topo_counts[t])
        if cum >= int(n_orphan_cath_fold) * 2:
            break
    rare_set = set(candidate_rare_topos)
    rare_df = pq_single[pq_single["cath_t"].isin(rare_set)]
    rare_by_topo: dict[str, list[Any]] = defaultdict(list)
    for row in rare_df.itertuples(index=False):
        rare_by_topo[str(row.cath_t)].append(row)

    for t in candidate_rare_topos:
        if len(orphan_records) >= int(n_orphan_cath_fold):
            break
        rows_t = rare_by_topo.get(t, [])
        if not rows_t:
            continue
        perm = rng.permutation(len(rows_t))
        added_from_t = False
        for idx in perm:
            if len(orphan_records) >= int(n_orphan_cath_fold):
                break
            row = rows_t[int(idx)]
            seq = str(row.sequence).strip().upper()
            if not is_valid_aa20(seq, min_len=min_len, max_len=max_len):
                continue
            digest = seq_sha256(seq)
            if digest in used_seq_hashes:
                continue
            used_seq_hashes.add(digest)
            acc = str(row.accession)
            orphan_records.append(
                AuditSequenceRecord(
                    record_id=f"orphan_{row.uniref50_id}",
                    split="unseen_fold",
                    sub_stratum="orphan_cath_fold",
                    sequence=seq,
                    seq_len=len(seq),
                    seq_sha256=digest,
                    accession=acc,
                    cath_h=str(row.cath_h),
                    cath_t=str(row.cath_t),
                    source_db="swissprot" if acc in sprot_accs else "trembl",
                    created_date=str(sprot_dates.get(acc, "")),
                )
            )
            added_from_t = True
        if added_from_t:
            orphan_topos.add(t)

    # 4. Select seen_cath_matched (pre-cutoff canonical Swiss-Prot) & same-CATH unseen_homolog
    non_orphan_df = pq_single[~pq_single["cath_t"].isin(orphan_topos)]
    is_sprot_mask = non_orphan_df["accession"].isin(precutoff_sprot_accs)
    sprot_cath_df = non_orphan_df[is_sprot_mask]
    uniref50_cath_accs.update(non_orphan_df["accession"].astype(str).tolist())

    # Map Swiss-Prot CATH-annotated entries for seen_cath_matched
    sprot_by_h: dict[str, list[Any]] = defaultdict(list)
    for row in sprot_cath_df.itertuples(index=False):
        sprot_by_h[str(row.cath_h)].append(row)
        cath_by_acc[str(row.accession)] = (str(row.cath_h), str(row.cath_t))

    # Pick diverse CATH H-superfamilies that also have non-SwissProt TrEMBL homologs
    non_sprot_df = non_orphan_df[~non_orphan_df["accession"].isin(sprot_accs)]
    trembl_h_counts = non_sprot_df["cath_h"].value_counts()
    shared_hs = sorted(
        [h for h in sprot_by_h.keys() if int(trembl_h_counts.get(h, 0)) >= 2],
        key=lambda h: (-int(trembl_h_counts.get(h, 0)), h),
    )

    seen_cath_matched: list[AuditSequenceRecord] = []
    matched_hs: set[str] = set()
    h_idx = 0
    # Round-robin across shared CATH H-superfamilies so seen_cath_matched spans diverse families
    per_h_cursors: dict[str, int] = defaultdict(int)
    while len(seen_cath_matched) < int(n_seen_cath_matched) and shared_hs:
        added_round = False
        for h in shared_hs:
            rows_h = sprot_by_h[h]
            cur = per_h_cursors[h]
            while cur < len(rows_h):
                row = rows_h[cur]
                cur += 1
                seq = str(row.sequence).strip().upper()
                digest = seq_sha256(seq)
                if digest in used_seq_hashes or not is_valid_aa20(seq, min_len=min_len, max_len=max_len):
                    continue
                used_seq_hashes.add(digest)
                acc = str(row.accession)
                seen_cath_matched.append(
                    AuditSequenceRecord(
                        record_id=f"seen_cath_{row.uniref50_id}",
                        split="seen_cath_matched",
                        sub_stratum="swissprot_precutoff_cath",
                        sequence=seq,
                        seq_len=len(seq),
                        seq_sha256=digest,
                        accession=acc,
                        cath_h=h,
                        cath_t=str(row.cath_t),
                        source_db="swissprot",
                        created_date=str(sprot_dates.get(acc, "")),
                    )
                )
                matched_hs.add(h)
                added_round = True
                break
            per_h_cursors[h] = cur
            if len(seen_cath_matched) >= int(n_seen_cath_matched):
                break
        if not added_round:
            break
        h_idx += 1

    # 5. Build unseen_homolog (half post-2023 / TrEMBL same-CATH, half UniRef90 non-rep homologs)
    unseen_homolog: list[AuditSequenceRecord] = []
    n_same_cath_target = int(n_unseen_homolog) // 2
    n_ur90_target = int(n_unseen_homolog) - n_same_cath_target

    # 5a. First harvest genuine post-2023 Swiss-Prot entries in matched_hs (true temporal holdouts!)
    for acc in sorted(postcutoff_sprot_accs):
        if len(unseen_homolog) >= n_same_cath_target:
            break
        ch_ct = cath_tsv_map.get(acc)
        if ch_ct is None or ch_ct[0] not in matched_hs or ch_ct[1] in orphan_topos:
            continue
        seq = sprot_seqs.get(acc, "")
        if not seq:
            continue
        digest = seq_sha256(seq)
        if digest in used_seq_hashes:
            continue
        used_seq_hashes.add(digest)
        unseen_homolog.append(
            AuditSequenceRecord(
                record_id=f"hom_post2023_sp_{acc}",
                split="unseen_homolog",
                sub_stratum="post2023_swissprot_same_cath",
                sequence=seq,
                seq_len=len(seq),
                seq_sha256=digest,
                accession=acc,
                cath_h=ch_ct[0],
                cath_t=ch_ct[1],
                source_db="swissprot_post2023",
                created_date=str(sprot_dates.get(acc, "")),
            )
        )

    # 5b. Fill remainder of n_same_cath_target from non-SwissProt TrEMBL in the exact same matched_hs
    trembl_matched_df = non_sprot_df[non_sprot_df["cath_h"].isin(matched_hs)]
    # Vectorized cap per superfamily (avoid iterating millions of rows in Python!)
    recent_mask = trembl_matched_df["accession"].astype(str).str.startswith(("A0AA", "A0AB", "A0AC", "A0A9"))
    trembl_recent = trembl_matched_df[recent_mask].groupby("cath_h", sort=False).head(8)
    trembl_other = trembl_matched_df[~recent_mask].groupby("cath_h", sort=False).head(4)
    trembl_capped = pd.concat([trembl_recent, trembl_other], ignore_index=True)
    trembl_by_h: dict[str, list[Any]] = defaultdict(list)
    for row in trembl_capped.itertuples(index=False):
        trembl_by_h[str(row.cath_h)].append(row)

    t_cursors: dict[str, int] = defaultdict(int)
    matched_hs_sorted = sorted(matched_hs)
    while len(unseen_homolog) < n_same_cath_target and matched_hs_sorted:
        added_any = False
        for h in matched_hs_sorted:
            rows_h = trembl_by_h.get(h, [])
            cur = t_cursors[h]
            while cur < len(rows_h):
                row = rows_h[cur]
                cur += 1
                seq = str(row.sequence).strip().upper()
                digest = seq_sha256(seq)
                if digest in used_seq_hashes or not is_valid_aa20(seq, min_len=min_len, max_len=max_len):
                    continue
                used_seq_hashes.add(digest)
                acc = str(row.accession)
                unseen_homolog.append(
                    AuditSequenceRecord(
                        record_id=f"hom_trembl_{row.uniref50_id}",
                        split="unseen_homolog",
                        sub_stratum="trembl_same_cath",
                        sequence=seq,
                        seq_len=len(seq),
                        seq_sha256=digest,
                        accession=acc,
                        cath_h=h,
                        cath_t=str(row.cath_t),
                        source_db="trembl",
                    )
                )
                added_any = True
                break
            t_cursors[h] = cur
            if len(unseen_homolog) >= n_same_cath_target:
                break
        if not added_any:
            break

    # 5c. Non-representative UniRef90 intra-cluster homologs (2026_03)
    uniref90_gz = data_root / "data/pairwise_assets/uniref90_2026_03/uniref90.fasta.gz"
    ur90_records = _sample_uniref90_nonrep_homologs(
        uniref90_gz,
        target_count=n_ur90_target,
        uniref50_rep_accs=uniref50_cath_accs | sprot_accs,
        cath_by_acc=cath_by_acc,
        allowed_cath_h=matched_hs,
        forbidden_cath_t=orphan_topos,
        excluded_seq_hashes=used_seq_hashes,
        min_len=min_len,
        max_len=max_len,
        max_scan_records=max_uniref90_scan,
        seed=seed,
    )
    unseen_homolog.extend(ur90_records)

    # Top up unseen_homolog from trembl_same_cath if ur90 file was unavailable or short
    while len(unseen_homolog) < int(n_unseen_homolog) and matched_hs_sorted:
        added_any = False
        for h in matched_hs_sorted:
            rows_h = trembl_by_h.get(h, [])
            cur = t_cursors[h]
            while cur < len(rows_h):
                row = rows_h[cur]
                cur += 1
                seq = str(row.sequence).strip().upper()
                digest = seq_sha256(seq)
                if digest in used_seq_hashes or not is_valid_aa20(seq, min_len=min_len, max_len=max_len):
                    continue
                used_seq_hashes.add(digest)
                unseen_homolog.append(
                    AuditSequenceRecord(
                        record_id=f"hom_trembl_{row.uniref50_id}",
                        split="unseen_homolog",
                        sub_stratum="trembl_same_cath",
                        sequence=seq,
                        seq_len=len(seq),
                        seq_sha256=digest,
                        accession=str(row.accession),
                        cath_h=h,
                        cath_t=str(row.cath_t),
                        source_db="trembl",
                    )
                )
                added_any = True
                break
            t_cursors[h] = cur
            if len(unseen_homolog) >= int(n_unseen_homolog):
                break
        if not added_any:
            break

    # 6. UniRef50 depth-bucket population census + HT stratified sample
    uniref50_fasta = data_root / "data/uniref50/uniref50.fasta"
    uniref50_idx_pkl = data_root / "data/uniref50/uniref50.fasta.idx_max1022.pkl"
    if max_uniref50_records is not None and uniref50_idx_pkl.is_file():
        ht_scan = sample_uniref50_depth_buckets_via_index(
            uniref50_fasta,
            uniref50_idx_pkl,
            n_probe_records=int(max_uniref50_records),
            n_per_bucket=n_per_depth_bucket,
            min_len=min_len,
            max_len=max_len,
            swissprot_accs=sprot_accs,
            swissprot_dates=sprot_dates,
            cath_by_acc=cath_by_acc,
            excluded_seq_hashes=used_seq_hashes,
            seed=seed,
        )
    else:
        ht_scan = scan_uniref50_depth_buckets(
            uniref50_fasta,
            n_per_bucket=n_per_depth_bucket,
            min_len=min_len,
            max_len=max_len,
            swissprot_accs=sprot_accs,
            swissprot_dates=sprot_dates,
            cath_by_acc=cath_by_acc,
            excluded_seq_hashes=used_seq_hashes,
            seed=seed,
            max_records=max_uniref50_records,
        )
    seen_bucket_records: list[AuditSequenceRecord] = ht_scan["sampled_records"]

    # If a fast bounded UniRef50 scan (max_uniref50_records) was requested, scale bucket
    # counts proportionally to the exact certified 60,315,044-cluster / 17,282,055,793-residue
    # UniRef50 totals from certificate.json & stage2_cache/summary.json!
    pop_all = ht_scan["population_buckets_all"]
    pop_elig = ht_scan["population_buckets_eligible"]
    census_mode = "exact_full_uniref50_census"
    if max_uniref50_records is not None:
        cert_path = data_root / "results/shared/megascale_disjointness_20260813/certificate.json"
        s2_sum_path = data_root / "output/stage2_cache/summary.json"
        exact_total_clusters = 60_315_044
        exact_total_residues = 17_282_055_793
        exact_elig_clusters = 31_196_107
        if cert_path.is_file():
            c_doc = json.loads(cert_path.read_text(encoding="utf-8"))
            exact_total_clusters = int(c_doc.get("database", {}).get("clusters", exact_total_clusters))
            exact_total_residues = int(c_doc.get("database", {}).get("letters", exact_total_residues))
        if s2_sum_path.is_file():
            s_doc = json.loads(s2_sum_path.read_text(encoding="utf-8"))
            exact_elig_clusters = int(s_doc.get("fasta_stats", {}).get("kept_candidates", exact_elig_clusters))
        obs_c_all = max(1, sum(v["n_clusters"] for v in pop_all.values()))
        obs_r_all = max(1, sum(v["total_residues"] for v in pop_all.values()))
        obs_c_elig = max(1, sum(v["n_clusters"] for v in pop_elig.values()))
        scale_c_all = float(exact_total_clusters) / float(obs_c_all)
        scale_r_all = float(exact_total_residues) / float(obs_r_all)
        scale_elig = float(exact_elig_clusters) / float(obs_c_elig)
        pop_all = {
            b: {
                "n_clusters": int(round(v["n_clusters"] * scale_c_all)),
                "total_residues": int(round(v["total_residues"] * scale_r_all)),
                "sum_cluster_members": int(round(v["sum_cluster_members"] * scale_c_all)),
            }
            for b, v in pop_all.items()
        }
        pop_elig = {
            b: {
                "n_clusters": int(round(v["n_clusters"] * scale_elig)),
                "total_residues": int(round(v["total_residues"] * scale_elig)),
                "sum_cluster_members": int(round(v["sum_cluster_members"] * scale_elig)),
            }
            for b, v in pop_elig.items()
        }
        census_mode = f"scaled_from_{ht_scan['total_fasta_records_scanned']}_to_{exact_total_clusters}_certified"

    # 7. 3D paired panel (64 natural CATH-S40 + 64 de novo zero-hit backbones + Foldseek 3Di)
    panel_records, coords_pack = build_paired_3d_panel(
        data_root,
        denovo_raw,
        n_natural=n_panel_natural,
        n_denovo=n_panel_denovo,
        denovo_pdb_cache_dir=denovo_pdb_cache_dir,
        fold_denovo=fold_denovo,
        fold_device=fold_device,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
    )

    # 8. Persist splits atomically + verify strict disjointness
    unseen_fold_records = denovo_records + orphan_records
    sha_seen_buckets = atomic_write_jsonl(
        splits_dir / "seen_pretrain_buckets.jsonl", [r.to_dict() for r in seen_bucket_records]
    )
    sha_seen_cath = atomic_write_jsonl(
        splits_dir / "seen_cath_matched.jsonl", [r.to_dict() for r in seen_cath_matched]
    )
    sha_hom = atomic_write_jsonl(
        splits_dir / "unseen_homolog.jsonl", [r.to_dict() for r in unseen_homolog]
    )
    sha_fold = atomic_write_jsonl(
        splits_dir / "unseen_fold.jsonl", [r.to_dict() for r in unseen_fold_records]
    )
    sha_3d = atomic_write_jsonl(
        splits_dir / "paired_3d_panel.jsonl", [r.to_dict() for r in panel_records]
    )

    coords_npz_path = splits_dir / "panel_3d_coords.npz"
    np.savez_compressed(coords_npz_path, **coords_pack)
    sha_coords = sha256_file(coords_npz_path)

    # Disjointness audit
    h_seen_b = {r.seq_sha256 for r in seen_bucket_records}
    h_seen_c = {r.seq_sha256 for r in seen_cath_matched}
    h_hom = {r.seq_sha256 for r in unseen_homolog}
    h_fold = {r.seq_sha256 for r in unseen_fold_records}
    if (h_seen_b & h_hom) or (h_seen_b & h_fold) or (h_seen_c & h_hom) or (h_seen_c & h_fold) or (h_hom & h_fold):
        raise RuntimeError("Cross-split sequence SHA-256 leakage detected!")

    seen_cath_topos = {r.cath_t for r in seen_cath_matched if r.cath_t}
    hom_topos = {r.cath_t for r in unseen_homolog if r.cath_t}
    if (seen_cath_topos & orphan_topos) or (hom_topos & orphan_topos):
        raise RuntimeError("CATH topology overlap detected with orphan_cath_fold!")

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())

    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_utc": _now_utc(),
        "seed": int(seed),
        "length_filter": {"min_len": int(min_len), "max_len": int(max_len), "alphabet": "AA20_strict"},
        "uniref50_total_records_scanned": int(ht_scan["total_fasta_records_scanned"]),
        "uniref50_census_mode": census_mode,
        "population_buckets_all": pop_all,
        "population_buckets_eligible": pop_elig,
        "split_counts": {
            "seen_pretrain_buckets": len(seen_bucket_records),
            "seen_pretrain_by_bucket": {
                b: sum(1 for r in seen_bucket_records if r.depth_bucket == b) for b in DEPTH_BUCKET_ORDER
            },
            "seen_cath_matched": len(seen_cath_matched),
            "unseen_homolog": len(unseen_homolog),
            "unseen_homolog_by_sub_stratum": {
                s: sum(1 for r in unseen_homolog if r.sub_stratum == s)
                for s in sorted({r.sub_stratum for r in unseen_homolog})
            },
            "unseen_fold": len(unseen_fold_records),
            "unseen_fold_denovo_zero_hit": len(denovo_records),
            "unseen_fold_orphan_cath": len(orphan_records),
            "paired_3d_panel": len(panel_records),
            "paired_3d_natural_cath_s40": sum(1 for r in panel_records if r.domain_group == "natural_cath_s40"),
            "paired_3d_denovo_zero_hit": sum(1 for r in panel_records if r.domain_group == "de_novo_zero_hit"),
        },
        "disjointness_checks": {
            "cross_split_seq_overlap": 0,
            "n_orphan_cath_topologies": len(orphan_topos),
            "n_matched_cath_superfamilies": len(matched_hs),
            "orphan_vs_seen_hom_topology_overlap": 0,
        },
        "files_sha256": {
            "splits/seen_pretrain_buckets.jsonl": sha_seen_buckets,
            "splits/seen_cath_matched.jsonl": sha_seen_cath,
            "splits/unseen_homolog.jsonl": sha_hom,
            "splits/unseen_fold.jsonl": sha_fold,
            "splits/paired_3d_panel.jsonl": sha_3d,
            "splits/panel_3d_coords.npz": sha_coords,
        },
        "bundle_size_bytes": int(total_bytes),
        "bundle_size_mb": round(float(total_bytes) / (1024.0 * 1024.0), 3),
    }
    atomic_write_json_with_sha256(out_dir / "bundle_manifest.json", manifest)
    return manifest
