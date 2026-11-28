"""Deterministic Stage-2 Knob-2 (UniRef50-CATH) corpus, fold-disjoint splits, and benchmark packager."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .objectives import PackedSequenceStore, encode_aa_sequence
from .spec import (
    AA20,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MIN_SEQ_LEN,
    assert_no_forbidden_infra_strings,
)

AA20_SET: frozenset[str] = frozenset(AA20)
BUNDLE_SCHEMA = "stage2_supply_demand_bundle"
SINGLE_MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
CATH_H_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


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
    """Atomically write JSON with a .sha256 sidecar after checking infra guardrails."""
    assert_no_forbidden_infra_strings(payload, label=path.name)
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


def atomic_write_gzip_json(path: Path, payload: Any) -> str:
    """Atomically write compact gzipped JSON."""
    assert_no_forbidden_infra_strings(payload, label=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=6) as fh:
            json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def cath_topology_of(cath_superfamily: str) -> str:
    """Extract C.A.T fold topology from a C.A.T.H superfamily string."""
    parts = cath_superfamily.strip().split(".")
    if len(parts) != 4:
        raise ValueError(f"Invalid CATH H-superfamily code: {cath_superfamily!r}")
    return ".".join(parts[:3])


@dataclass(frozen=True)
class AnnotatedSequence:
    """One canonical natural protein sequence with optional CATH C.A.T.H assignment."""

    seq_id: str
    sequence: str
    cath_h: str  # "" for unannotated UniRef50 singletons
    cath_t: str  # "" for unannotated UniRef50 singletons

    @property
    def length(self) -> int:
        return len(self.sequence)


def save_packed_split_npz(
    path: Path,
    records: Sequence[AnnotatedSequence],
) -> dict[str, Any]:
    """Save sequence split as compact uint8 PackedSequenceStore + metadata arrays."""
    store = PackedSequenceStore.from_sequences([r.sequence for r in records])
    ids = np.asarray([r.seq_id for r in records], dtype=str)
    cath_h = np.asarray([r.cath_h for r in records], dtype=str)
    cath_t = np.asarray([r.cath_t for r in records], dtype=str)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(
            tmp_path,
            tokens_flat=store.tokens_flat,
            offsets=store.offsets,
            lengths=store.lengths,
            ids=ids,
            cath_h=cath_h,
            cath_t=cath_t,
        )
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return {
        "file": path.name,
        "n_sequences": int(store.n_sequences),
        "total_tokens": int(store.total_tokens),
        "mean_length": (
            float(store.total_tokens) / float(store.n_sequences) if store.n_sequences else 0.0
        ),
        "n_distinct_cath_h": len({r.cath_h for r in records if r.cath_h}),
        "n_distinct_cath_t": len({r.cath_t for r in records if r.cath_t}),
        "size_bytes": int(path.stat().st_size),
    }


def load_packed_split_npz(path: Path) -> tuple[PackedSequenceStore, dict[str, np.ndarray]]:
    """Load packed sequence split returning (PackedSequenceStore, metadata_dict)."""
    with np.load(path, allow_pickle=False) as data:
        store = PackedSequenceStore(
            tokens_flat=np.asarray(data["tokens_flat"], dtype=np.uint8),
            offsets=np.asarray(data["offsets"], dtype=np.int64),
            lengths=np.asarray(data["lengths"], dtype=np.int32),
        )
        meta = {
            "ids": np.asarray(data["ids"], dtype=str),
            "cath_h": np.asarray(data["cath_h"], dtype=str),
            "cath_t": np.asarray(data["cath_t"], dtype=str),
        }
    return store, meta


# --------------------------------------------------------------------------- #
# Streaming CATH + UniRef50 Join
# --------------------------------------------------------------------------- #


def load_single_cath_mapping_tsv(path: Path) -> dict[str, str]:
    """Load unambiguous single-C.A.T.H mapping ``accession -> C.A.T.H`` from a TSV or TSV.gz."""
    opener = gzip.open if path.suffix == ".gz" else open
    acc_to_h: dict[str, str | None] = {}
    with opener(path, "rt", encoding="utf-8") as fh:
        first = fh.readline()
        if not first:
            return {}
        fields = first.rstrip("\n").split("\t")
        has_header = len(fields) >= 2 and fields[0].lower() in ("uniprot", "accession")
        if not has_header and len(fields) >= 2 and CATH_H_RE.match(fields[1]):
            acc_to_h[fields[0]] = fields[1]
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            acc, sf = parts[0], parts[1]
            if not CATH_H_RE.match(sf):
                continue
            prev = acc_to_h.get(acc)
            if prev is None and acc not in acc_to_h:
                acc_to_h[acc] = sf
            elif prev != sf:
                # Multi-domain / ambiguous primary CATH superfamily -> mark excluded
                acc_to_h[acc] = None
    return {acc: sf for acc, sf in acc_to_h.items() if sf is not None}


def extract_uniref50_cath_table(
    protein2ipr_gz: Path,
    uniref50_accessions: set[str],
    *,
    output_tsv_gz: Path | None = None,
) -> dict[str, str]:
    """Stream InterPro ``protein2ipr.dat.gz`` filtering ``G3DSA:<C.A.T.H>`` for UniRef50 accessions.

    Returns unambiguous single-CATH-H mapping ``accession -> C.A.T.H``. Uses native
    ``pigz``/``gzip -dc`` + ``LC_ALL=C grep -F G3DSA:`` pipeline when available for
    high throughput on the 16 GB dump, with automatic pure-Python fallback.
    """
    if not protein2ipr_gz.is_file():
        raise FileNotFoundError(f"Missing protein2ipr dump: {protein2ipr_gz}")

    acc_to_h: dict[str, str | None] = {}
    decomp = shutil.which("pigz") or shutil.which("gzip")
    grep_bin = shutil.which("grep")

    def _consume_lines(line_iter: Iterable[str]) -> None:
        for line in line_iter:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            acc = parts[0]
            if acc not in uniref50_accessions:
                continue
            sig = parts[3]
            if not sig.startswith("G3DSA:"):
                continue
            sf = sig[6:]
            if not CATH_H_RE.match(sf):
                continue
            if acc not in acc_to_h:
                acc_to_h[acc] = sf
            elif acc_to_h[acc] != sf:
                acc_to_h[acc] = None

    if decomp and grep_bin:
        env = dict(os.environ, LC_ALL="C")
        p1 = subprocess.Popen(
            [decomp, "-dc", str(protein2ipr_gz)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        p2 = subprocess.Popen(
            [grep_bin, "-F", "G3DSA:"],
            stdin=p1.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert p1.stdout is not None and p2.stdout is not None
        p1.stdout.close()
        try:
            _consume_lines(p2.stdout)
        finally:
            p2.stdout.close()
            p2.wait()
            p1.wait()
    else:
        with gzip.open(protein2ipr_gz, "rt", encoding="utf-8", errors="replace") as fh:
            _consume_lines(line for line in fh if "G3DSA:" in line)

    single_map = {acc: sf for acc, sf in acc_to_h.items() if sf is not None}

    if output_tsv_gz is not None:
        output_tsv_gz.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{output_tsv_gz.name}.", suffix=".tmp", dir=str(output_tsv_gz.parent)
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as out:
                out.write("uniprot\tcath_superfamily\n")
                for acc, sf in sorted(single_map.items()):
                    out.write(f"{acc}\t{sf}\n")
            os.replace(tmp_path, output_tsv_gz)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    return single_map


def iter_uniref50_fasta(
    fasta_path: Path,
    *,
    min_len: int = DEFAULT_MIN_SEQ_LEN,
    max_len: int = DEFAULT_MAX_SEQ_LEN,
) -> Iterable[tuple[str, str, str]]:
    """Stream canonical 20-AA sequences within [min_len, max_len] from UniRef50 FASTA.

    Yields ``(cluster_id, accession, sequence)``.
    """
    opener = gzip.open if fasta_path.suffix == ".gz" else open
    cur_cid = ""
    cur_acc = ""
    chunks: list[str] = []

    def _flush() -> tuple[str, str, str] | None:
        if not cur_cid:
            return None
        seq = "".join(chunks).strip().upper()
        if min_len <= len(seq) <= max_len and set(seq).issubset(AA20_SET):
            return cur_cid, cur_acc, seq
        return None

    with opener(fasta_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                rec = _flush()
                if rec is not None:
                    yield rec
                tok = line[1:].split(None, 1)[0].strip()
                cur_cid = tok
                cur_acc = tok[len("UniRef50_") :] if tok.startswith("UniRef50_") else tok
                chunks = []
            else:
                chunks.append(line.strip())
        rec = _flush()
        if rec is not None:
            yield rec


# --------------------------------------------------------------------------- #
# Core Fold-Disjoint Splitter & Token-Matched (M, d) Arm Builder
# --------------------------------------------------------------------------- #


def _trim_records_to_token_budget(
    selected: list[AnnotatedSequence],
    reserve_pool: Sequence[AnnotatedSequence],
    target_tokens: int,
    *,
    rel_tol: float = 0.002,
) -> list[AnnotatedSequence]:
    """Adjust tail sequences so total tokens matches target_tokens within rel_tol (<0.2%)."""
    records = list(selected)
    total = sum(r.length for r in records)
    max_err = max(1, int(math.floor(float(target_tokens) * float(rel_tol))))

    if abs(total - target_tokens) <= max_err:
        return records

    # If over budget, drop tail records until within or just below target_tokens
    while total > target_tokens + max_err and len(records) > 1:
        dropped = records.pop()
        total -= dropped.length

    if abs(total - target_tokens) <= max_err:
        return records

    # If below budget, greedily pick closest length sequences from reserve_pool
    used_ids = {r.seq_id for r in records}
    avail = [r for r in reserve_pool if r.seq_id not in used_ids]
    avail.sort(key=lambda r: (r.length, r.seq_id))

    while total < target_tokens - max_err and avail:
        need = target_tokens - total
        # Pick record whose length gets closest to `need` without overshooting beyond max_err
        best_idx = None
        best_gap = None
        for idx, cand in enumerate(avail):
            new_total = total + cand.length
            gap = abs(new_total - target_tokens)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_idx = idx
        if best_idx is None:
            break
        chosen = avail.pop(best_idx)
        if total + chosen.length > target_tokens + max_err:
            avail.append(chosen)
            break
        records.append(chosen)
        total += chosen.length

    # Fine-tune via 1-for-1 swap from tail window against reserve_pool to hit exact token budget
    if abs(total - target_tokens) > max_err and avail and records:
        window_start = max(0, len(records) - 128)
        best_gap = abs(total - target_tokens)
        best_pair: tuple[int, int] | None = None
        for r_idx in range(window_start, len(records)):
            r_len = records[r_idx].length
            for a_idx, cand in enumerate(avail):
                gap = abs((total - r_len + cand.length) - target_tokens)
                if gap < best_gap:
                    best_gap = gap
                    best_pair = (r_idx, a_idx)
                    if gap == 0:
                        break
            if best_gap == 0:
                break
        if best_pair is not None:
            r_idx, a_idx = best_pair
            old_rec = records[r_idx]
            new_rec = avail[a_idx]
            records[r_idx] = new_rec
            total = total - old_rec.length + new_rec.length

    return records


def build_controlled_natural_splits(
    cath_records: Sequence[AnnotatedSequence],
    *,
    singleton_uniref_pool: Sequence[AnnotatedSequence] = (),
    priority_unseen_topologies: Sequence[str] = (),
    excluded_eval_seq_hashes: set[str] | frozenset[str] = frozenset(),
    target_unique_tokens: int = 5_000_000,
    arm_family_targets: Sequence[int] = (64, 256, 1024, 4096),
    include_singleton_arm: bool = True,
    n_unseen_fold_eval: int = 2_048,
    min_unseen_topologies: int = 80,
    n_homolog_eval: int = 1_024,
    n_train_subsample_eval: int = 1_024,
    n_global_natural_eval: int = 2_048,
    token_rel_tol: float = 0.002,
    seed: int = 20261128,
) -> dict[str, Any]:
    """Construct strict C.A.T fold-disjoint eval splits + token-matched (M, d) training arms.

    Hard invariants enforced by assertion before returning:
      1. ZERO sequence from any held-out ``C.A.T`` topology appears in ANY training arm,
         ``eval_unseen_homolog``, or ``eval_global_natural``.
      2. ZERO sequence from ``excluded_eval_seq_hashes`` appears in ANY training arm.
      3. For every arm, ``train`` and ``eval_unseen_homolog`` share ZERO sequences, while
         100% of ``eval_unseen_homolog`` sequences belong to the arm's active training
         ``C.A.T.H`` superfamilies.
      4. Every training arm's unique token count is within ``token_rel_tol`` (default 0.2%)
         of ``target_unique_tokens`` (when corpus supply permits; otherwise raises explicitly).
    """
    rng = np.random.default_rng(int(seed))

    # Deduplicate by sequence hash and filter eval WT exact hits from train-eligible pool
    seen_hashes: set[str] = set()
    clean_cath: list[AnnotatedSequence] = []
    for rec in cath_records:
        if not rec.cath_h:
            continue
        sh = seq_sha256(rec.sequence)
        if sh in seen_hashes or sh in excluded_eval_seq_hashes:
            continue
        seen_hashes.add(sh)
        cath_t = rec.cath_t or cath_topology_of(rec.cath_h)
        clean_cath.append(
            AnnotatedSequence(
                seq_id=rec.seq_id,
                sequence=rec.sequence,
                cath_h=rec.cath_h,
                cath_t=cath_t,
            )
        )

    by_topo: dict[str, list[AnnotatedSequence]] = defaultdict(list)
    by_h: dict[str, list[AnnotatedSequence]] = defaultdict(list)
    for rec in clean_cath:
        by_topo[rec.cath_t].append(rec)
        by_h[rec.cath_h].append(rec)

    all_topos = sorted(by_topo.keys())
    if len(all_topos) < min_unseen_topologies + 2:
        raise RuntimeError(
            f"Corpus has only {len(all_topos)} distinct C.A.T topologies; need at least "
            f"{min_unseen_topologies + 2} for fold-disjoint split"
        )

    # Protect top deepest H-superfamilies' topologies so M64 always has deep training families
    top_h_sorted = sorted(by_h.keys(), key=lambda h: (-len(by_h[h]), h))
    n_protect = min(max(arm_family_targets[0] if arm_family_targets else 64, 64), len(top_h_sorted) // 2)
    protected_topos = {cath_topology_of(h) for h in top_h_sorted[:n_protect]}

    # Select held-out C.A.T topologies prioritizing downstream overlap (contacts, MegaScale, PG)
    unseen_topos: list[str] = []
    unseen_topo_set: set[str] = set()
    unseen_seq_count = 0

    prio_candidates = [
        t for t in priority_unseen_topologies
        if t in by_topo and t not in protected_topos and t not in unseen_topo_set
    ]
    for t in prio_candidates:
        unseen_topos.append(t)
        unseen_topo_set.add(t)
        unseen_seq_count += len(by_topo[t])
        if len(unseen_topos) >= min_unseen_topologies and unseen_seq_count >= n_unseen_fold_eval:
            break

    # Fill remaining held-out topologies deterministically from non-protected topologies
    remaining_topos = [t for t in all_topos if t not in protected_topos and t not in unseen_topo_set]
    rng.shuffle(remaining_topos)
    # Sort slightly toward moderate-size topologies so we meet both >=min_unseen_topologies and >=n_unseen_fold_eval
    for t in remaining_topos:
        if len(unseen_topos) >= min_unseen_topologies and unseen_seq_count >= n_unseen_fold_eval:
            break
        unseen_topos.append(t)
        unseen_topo_set.add(t)
        unseen_seq_count += len(by_topo[t])

    # If still short of n_unseen_fold_eval, add more non-protected topologies
    if unseen_seq_count < n_unseen_fold_eval:
        for t in all_topos:
            if t not in unseen_topo_set and t not in protected_topos:
                unseen_topos.append(t)
                unseen_topo_set.add(t)
                unseen_seq_count += len(by_topo[t])
                if unseen_seq_count >= n_unseen_fold_eval:
                    break

    if len(unseen_topos) < min_unseen_topologies or unseen_seq_count < n_unseen_fold_eval:
        raise RuntimeError(
            f"Could not satisfy unseen_fold requirements: got {len(unseen_topos)} topologies "
            f"(need >= {min_unseen_topologies}) and {unseen_seq_count} sequences (need >= {n_unseen_fold_eval})"
        )

    unseen_topos = sorted(unseen_topo_set)

    # Sample balanced eval_unseen_fold across ALL held-out topologies (round-robin after shuffling)
    unseen_by_t_shuffled: dict[str, list[AnnotatedSequence]] = {}
    for t in unseen_topos:
        recs = list(by_topo[t])
        rng.shuffle(recs)
        unseen_by_t_shuffled[t] = recs

    eval_unseen_fold: list[AnnotatedSequence] = []
    ptrs = {t: 0 for t in unseen_topos}
    while len(eval_unseen_fold) < n_unseen_fold_eval:
        progressed = False
        for t in unseen_topos:
            if len(eval_unseen_fold) >= n_unseen_fold_eval:
                break
            p = ptrs[t]
            if p < len(unseen_by_t_shuffled[t]):
                eval_unseen_fold.append(unseen_by_t_shuffled[t][p])
                ptrs[t] = p + 1
                progressed = True
        if not progressed:
            break

    # Partition train_eligible H-superfamilies into disjoint (train_pool, homolog_pool, global_pool)
    train_eligible_h: dict[str, list[AnnotatedSequence]] = {}
    for h, recs in sorted(by_h.items()):
        t = cath_topology_of(h)
        if t in unseen_topo_set:
            continue
        shuf = list(recs)
        rng.shuffle(shuf)
        train_eligible_h[h] = shuf

    # Reserve a disjoint eval_global_natural pool from train_eligible families (keeping >=2 for multi-seq fams)
    global_candidates: list[AnnotatedSequence] = []
    train_pool_by_h: dict[str, list[AnnotatedSequence]] = {}
    homolog_pool_by_h: dict[str, list[AnnotatedSequence]] = {}

    for h, recs in sorted(train_eligible_h.items()):
        n = len(recs)
        if n >= 3:
            # Reserve 1 for global_natural candidate if family has >= 3, then split rest into homolog & train
            global_candidates.append(recs[0])
            rest = recs[1:]
            n_hom = min(max(1, int(round(0.15 * len(rest)))), 64)
            n_hom = min(n_hom, len(rest) - 1)
            homolog_pool_by_h[h] = rest[:n_hom]
            train_pool_by_h[h] = rest[n_hom:]
        elif n == 2:
            homolog_pool_by_h[h] = [recs[0]]
            train_pool_by_h[h] = [recs[1]]
        else:
            train_pool_by_h[h] = list(recs)
            homolog_pool_by_h[h] = []

    rng.shuffle(global_candidates)
    # If global_candidates < n_global_natural_eval, supplement from largest train_pools (keeping them disjoint from train)
    if len(global_candidates) < n_global_natural_eval:
        extra_needed = n_global_natural_eval - len(global_candidates)
        deep_hs = sorted(train_pool_by_h.keys(), key=lambda h: (-len(train_pool_by_h[h]), h))
        for h in deep_hs:
            if extra_needed <= 0:
                break
            pool = train_pool_by_h[h]
            if len(pool) > 10:
                take = min(extra_needed, max(1, len(pool) // 20))
                global_candidates.extend(pool[-take:])
                train_pool_by_h[h] = pool[:-take]
                extra_needed -= take
    eval_global_natural = global_candidates[:n_global_natural_eval]
    global_ids = {r.seq_id for r in eval_global_natural}

    # Build token-matched (M, d) arms
    arms_data: dict[str, dict[str, Any]] = {}
    # Sort eligible H-superfamilies that have BOTH train_pool and homolog_pool by train token supply
    eligible_h_ranked = sorted(
        [h for h in train_pool_by_h if train_pool_by_h[h] and homolog_pool_by_h[h]],
        key=lambda h: (-sum(r.length for r in train_pool_by_h[h]), h),
    )

    for target_m in arm_family_targets:
        eff_m = min(int(target_m), len(eligible_h_ranked))
        if eff_m < 1:
            raise RuntimeError(f"No train-eligible CATH superfamilies available for M={target_m}")
        active_hs = eligible_h_ranked[:eff_m]
        arm_id = f"M{int(target_m)}"

        # Water-filling token allocation across active_hs to match target_unique_tokens
        fam_selected: dict[str, list[AnnotatedSequence]] = {h: [] for h in active_hs}
        fam_tokens: dict[str, int] = {h: 0 for h in active_hs}
        fam_ptrs: dict[str, int] = {h: 0 for h in active_hs}

        total_supply = sum(sum(r.length for r in train_pool_by_h[h]) for h in active_hs)
        if total_supply < int(target_unique_tokens * (1.0 - token_rel_tol)):
            raise RuntimeError(
                f"Arm {arm_id} (realized M={eff_m}) has only {total_supply:,} train tokens, "
                f"below target_unique_tokens={target_unique_tokens:,}"
            )

        # Round-robin balanced fill across all active_hs until target_unique_tokens is reached
        total_tok = 0
        # First pass: take 1 sequence from every active family so all M families are represented
        for h in active_hs:
            rec = train_pool_by_h[h][0]
            fam_selected[h].append(rec)
            fam_tokens[h] += rec.length
            fam_ptrs[h] = 1
            total_tok += rec.length

        # Subsequent passes: fill families with smallest current token count first (exact water-filling)
        active_remaining = [h for h in active_hs if fam_ptrs[h] < len(train_pool_by_h[h])]
        while total_tok < target_unique_tokens and active_remaining:
            next_active: list[str] = []
            # Order families by current token count ascending so depths stay maximally balanced
            active_remaining.sort(key=lambda h: (fam_tokens[h], h))
            for h in active_remaining:
                if total_tok >= target_unique_tokens:
                    next_active.append(h)
                    continue
                ptr = fam_ptrs[h]
                pool = train_pool_by_h[h]
                if ptr < len(pool):
                    rec = pool[ptr]
                    fam_selected[h].append(rec)
                    fam_tokens[h] += rec.length
                    fam_ptrs[h] = ptr + 1
                    total_tok += rec.length
                    if fam_ptrs[h] < len(pool):
                        next_active.append(h)
            active_remaining = next_active

        selected_all = [rec for h in active_hs for rec in fam_selected[h]]
        reserve_all = [
            rec for h in active_hs for rec in train_pool_by_h[h][fam_ptrs[h] :]
        ]
        selected_trimmed = _trim_records_to_token_budget(
            selected_all,
            reserve_all,
            target_unique_tokens,
            rel_tol=token_rel_tol,
        )
        rng.shuffle(selected_trimmed)

        # Count realized family weights in selected_trimmed for proportional homolog sampling
        realized_fam_counts = Counter(r.cath_h for r in selected_trimmed)
        active_realized_hs = sorted(realized_fam_counts.keys())
        eval_homolog = _sample_proportional_homologs(
            active_realized_hs,
            realized_fam_counts,
            homolog_pool_by_h,
            n_target=n_homolog_eval,
            rng=rng,
        )
        train_sub = list(selected_trimmed[: min(n_train_subsample_eval, len(selected_trimmed))])

        realized_tokens = sum(r.length for r in selected_trimmed)
        rel_err = abs(realized_tokens - target_unique_tokens) / float(target_unique_tokens)
        if rel_err > token_rel_tol:
            raise RuntimeError(
                f"Arm {arm_id} token budget mismatch: got {realized_tokens:,} vs "
                f"target {target_unique_tokens:,} (rel_err={rel_err:.5f} > {token_rel_tol})"
            )

        arms_data[arm_id] = {
            "arm_id": arm_id,
            "target_families": int(target_m),
            "realized_families": len(active_realized_hs),
            "is_singleton_limit": False,
            "train": selected_trimmed,
            "eval_train_sub": train_sub,
            "eval_unseen_homolog": eval_homolog,
            "realized_tokens": int(realized_tokens),
            "token_budget_rel_error": float(rel_err),
            "mean_depth": float(len(selected_trimmed)) / float(max(1, len(active_realized_hs))),
        }

    if include_singleton_arm:
        # d=1 maximum-deduplication limit: 1 sequence per train_eligible CATH H-superfamily,
        # topped up to exact target_unique_tokens using disjoint UniRef50 singletons.
        singleton_recs: list[AnnotatedSequence] = []
        singleton_h_used: list[str] = []
        used_seq_hashes: set[str] = set()

        cath_h_all = sorted(train_pool_by_h.keys())
        rng.shuffle(cath_h_all)
        for h in cath_h_all:
            if not train_pool_by_h[h]:
                continue
            rec = train_pool_by_h[h][0]
            sh = seq_sha256(rec.sequence)
            if sh in used_seq_hashes or rec.seq_id in global_ids:
                continue
            singleton_recs.append(rec)
            singleton_h_used.append(h)
            used_seq_hashes.add(sh)
            if sum(r.length for r in singleton_recs) >= target_unique_tokens:
                break

        # Top up with disjoint unannotated UniRef50 cluster representatives
        cur_tok = sum(r.length for r in singleton_recs)
        reserve_singletons: list[AnnotatedSequence] = []
        if cur_tok < target_unique_tokens:
            pool_copy = list(singleton_uniref_pool)
            rng.shuffle(pool_copy)
            for rec in pool_copy:
                if rec.cath_t and rec.cath_t in unseen_topo_set:
                    continue
                sh = seq_sha256(rec.sequence)
                if sh in used_seq_hashes or sh in excluded_eval_seq_hashes or rec.seq_id in global_ids:
                    continue
                used_seq_hashes.add(sh)
                if cur_tok < target_unique_tokens:
                    singleton_recs.append(rec)
                    cur_tok += rec.length
                else:
                    reserve_singletons.append(rec)
                    if len(reserve_singletons) >= 512:
                        break

        if cur_tok < int(target_unique_tokens * (1.0 - token_rel_tol)):
            raise RuntimeError(
                f"M_max_singleton has only {cur_tok:,} tokens (need {target_unique_tokens:,}); "
                "supply more singleton_uniref_pool sequences"
            )

        singleton_trimmed = _trim_records_to_token_budget(
            singleton_recs,
            reserve_singletons,
            target_unique_tokens,
            rel_tol=token_rel_tol,
        )
        rng.shuffle(singleton_trimmed)

        # Eval unseen homologs strictly from CATH H-superfamilies present in singleton_trimmed
        present_hs = sorted({r.cath_h for r in singleton_trimmed if r.cath_h and homolog_pool_by_h.get(r.cath_h)})
        hom_counts = Counter({h: 1 for h in present_hs})
        eval_hom_single = _sample_proportional_homologs(
            present_hs,
            hom_counts,
            homolog_pool_by_h,
            n_target=n_homolog_eval,
            rng=rng,
        )
        train_sub_single = list(
            singleton_trimmed[: min(n_train_subsample_eval, len(singleton_trimmed))]
        )
        realized_tokens = sum(r.length for r in singleton_trimmed)
        rel_err = abs(realized_tokens - target_unique_tokens) / float(target_unique_tokens)
        if rel_err > token_rel_tol:
            raise RuntimeError(
                f"M_max_singleton token budget mismatch: {realized_tokens:,} vs {target_unique_tokens:,}"
            )

        arms_data["M_max_singleton"] = {
            "arm_id": "M_max_singleton",
            "target_families": len(singleton_trimmed),
            "realized_families": len(singleton_trimmed),
            "n_cath_h_represented": len({r.cath_h for r in singleton_trimmed if r.cath_h}),
            "n_uniref50_singletons": sum(1 for r in singleton_trimmed if not r.cath_h),
            "is_singleton_limit": True,
            "train": singleton_trimmed,
            "eval_train_sub": train_sub_single,
            "eval_unseen_homolog": eval_hom_single,
            "realized_tokens": int(realized_tokens),
            "token_budget_rel_error": float(rel_err),
            "mean_depth": 1.0,
        }

    # Verify strict fold-disjointness & sequence-disjointness invariants across ALL arms
    verify_fold_disjointness(
        unseen_topologies=unseen_topo_set,
        eval_unseen_fold=eval_unseen_fold,
        eval_global_natural=eval_global_natural,
        arms_data=arms_data,
        excluded_eval_seq_hashes=excluded_eval_seq_hashes,
    )

    return {
        "unseen_fold_topologies": unseen_topos,
        "eval_unseen_fold": eval_unseen_fold,
        "eval_global_natural": eval_global_natural,
        "arms": arms_data,
    }


def _sample_proportional_homologs(
    active_hs: Sequence[str],
    family_weights: Mapping[str, int],
    homolog_pool_by_h: Mapping[str, Sequence[AnnotatedSequence]],
    *,
    n_target: int,
    rng: np.random.Generator,
) -> list[AnnotatedSequence]:
    """Sample up to n_target unseen homologs from exact active_hs proportional to family_weights."""
    pools: dict[str, list[AnnotatedSequence]] = {
        h: list(homolog_pool_by_h.get(h, ())) for h in active_hs if homolog_pool_by_h.get(h)
    }
    if not pools:
        return []
    for recs in pools.values():
        rng.shuffle(recs)

    total_avail = sum(len(v) for v in pools.values())
    n_goal = min(int(n_target), total_avail)
    total_w = float(sum(family_weights[h] for h in pools))

    # Proportional quota allocation + largest-remainder fill
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for h in sorted(pools.keys()):
        exact = float(n_goal) * float(family_weights[h]) / max(total_w, 1.0)
        q = min(len(pools[h]), int(math.floor(exact)))
        quotas[h] = q
        assigned += q
        remainders.append((exact - q, h))

    # Distribute remaining slots respecting pool caps
    remainders.sort(key=lambda kv: (-kv[0], kv[1]))
    while assigned < n_goal:
        progressed = False
        for _, h in remainders:
            if assigned >= n_goal:
                break
            if quotas[h] < len(pools[h]):
                quotas[h] += 1
                assigned += 1
                progressed = True
        if not progressed:
            break

    selected: list[AnnotatedSequence] = []
    for h in sorted(pools.keys()):
        selected.extend(pools[h][: quotas[h]])
    rng.shuffle(selected)
    return selected


def verify_fold_disjointness(
    *,
    unseen_topologies: set[str] | frozenset[str],
    eval_unseen_fold: Sequence[AnnotatedSequence],
    eval_global_natural: Sequence[AnnotatedSequence],
    arms_data: Mapping[str, Mapping[str, Any]],
    excluded_eval_seq_hashes: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Assert every strict Stage-2 fold-disjointness and homolog-membership invariant."""
    unseen_ids = {r.seq_id for r in eval_unseen_fold}
    unseen_hashes = {seq_sha256(r.sequence) for r in eval_unseen_fold}
    for r in eval_unseen_fold:
        if r.cath_t not in unseen_topologies:
            raise AssertionError(f"eval_unseen_fold record {r.seq_id} has non-held-out fold {r.cath_t}")

    for r in eval_global_natural:
        if r.cath_t in unseen_topologies:
            raise AssertionError(f"eval_global_natural leaked held-out fold {r.cath_t}")

    for arm_id, arm in arms_data.items():
        train_recs: Sequence[AnnotatedSequence] = arm["train"]
        hom_recs: Sequence[AnnotatedSequence] = arm["eval_unseen_homolog"]

        train_topos = {r.cath_t for r in train_recs if r.cath_t}
        leaked_topos = train_topos & set(unseen_topologies)
        if leaked_topos:
            raise AssertionError(
                f"Arm {arm_id} training set leaked {len(leaked_topos)} held-out C.A.T folds: "
                f"{sorted(leaked_topos)[:5]}"
            )

        train_ids = {r.seq_id for r in train_recs}
        train_hashes = {seq_sha256(r.sequence) for r in train_recs}
        train_hs = {r.cath_h for r in train_recs if r.cath_h}

        if train_ids & unseen_ids or train_hashes & unseen_hashes:
            raise AssertionError(f"Arm {arm_id} training set shares sequences with eval_unseen_fold")

        if excluded_eval_seq_hashes and (train_hashes & set(excluded_eval_seq_hashes)):
            raise AssertionError(f"Arm {arm_id} training set leaked excluded downstream WT sequence")

        hom_ids = {r.seq_id for r in hom_recs}
        hom_hashes = {seq_sha256(r.sequence) for r in hom_recs}
        if train_ids & hom_ids or train_hashes & hom_hashes:
            raise AssertionError(f"Arm {arm_id} train and eval_unseen_homolog share sequences")

        for r in hom_recs:
            if r.cath_t in unseen_topologies:
                raise AssertionError(f"Arm {arm_id} eval_unseen_homolog leaked held-out fold {r.cath_t}")
            if r.cath_h not in train_hs:
                raise AssertionError(
                    f"Arm {arm_id} eval_unseen_homolog record {r.seq_id} ({r.cath_h}) "
                    "is not in arm's active training CATH H-superfamilies"
                )


# --------------------------------------------------------------------------- #
# Downstream Benchmark Packagers (MegaScale, ProteinGym, Unseen-Fold Contacts)
# --------------------------------------------------------------------------- #


def package_megascale_benchmark(
    *,
    megascale_parquet_dir: Path,
    disjointness_dir: Path,
    acc_to_cath_h: Mapping[str, str],
    unseen_topologies: set[str] | frozenset[str],
    max_wt_len: int = 120,
    min_variants_per_wt: int = 20,
) -> dict[str, Any]:
    """Package MegaScale ddG single-substitution benchmark with de_novo / unseen_fold / seen_pool strata."""
    import pandas as pd

    parquets = sorted(megascale_parquet_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {megascale_parquet_dir}")

    dfs = [
        pd.read_parquet(p, columns=["WT_name", "mut_type", "aa_seq", "ddG_ML"])
        for p in parquets
    ]
    df = pd.concat(dfs, ignore_index=True)

    # Load disjointness certificate and UniRef50 DIAMOND hits
    qi_list = json.loads((disjointness_dir / "query_index.json").read_text(encoding="utf-8"))
    cert = json.loads((disjointness_dir / "certificate.json").read_text(encoding="utf-8"))

    # Designs that hit UniRef50 at E<=1e-3 are listed in cert["design_hits_at_gate"]
    hit_design_wts = {
        str(item["WT_name"]) for item in cert.get("design_hits_at_gate", []) if "WT_name" in item
    }
    qi_by_wt = {str(row["WT_name"]): row for row in qi_list}

    qid_to_hits: dict[str, list[str]] = defaultdict(list)
    hits_path = disjointness_dir / "hits.tsv"
    if hits_path.is_file():
        with hits_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 10:
                    continue
                qid, subj = parts[0], parts[1]
                try:
                    evalue = float(parts[9])
                except ValueError:
                    continue
                if evalue <= 1e-3:
                    acc = subj[len("UniRef50_") :] if subj.startswith("UniRef50_") else subj
                    qid_to_hits[qid].append(acc)

    # Extract WT sequences
    wt_rows = df[df["mut_type"] == "wt"].groupby("WT_name")["aa_seq"].first().to_dict()

    # Filter canonical single-residue substitutions with finite ddG_ML
    mut_series = df["mut_type"].astype(str)
    is_single = mut_series.str.match(SINGLE_MUT_RE)
    ddg_vals = pd.to_numeric(df["ddG_ML"], errors="coerce")
    valid = df[is_single & ddg_vals.notna()].copy()
    valid["ddG_num"] = ddg_vals[is_single & ddg_vals.notna()].astype(float)

    domains: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()

    for wt_name, grp in valid.groupby("WT_name", sort=True):
        wt_key = str(wt_name)
        qi_row = qi_by_wt.get(wt_key)
        wt_seq = str(qi_row["sequence"] if qi_row is not None else wt_rows.get(wt_key, "")).strip().upper()
        if not wt_seq or len(wt_seq) > max_wt_len or not set(wt_seq).issubset(AA20_SET):
            continue

        # Aggregate duplicate measurements for the same single mutation by mean ddG_ML
        mut_accum: dict[tuple[int, str, str], list[float]] = defaultdict(list)
        for m_str, ddg in zip(grp["mut_type"].astype(str), grp["ddG_num"]):
            match = SINGLE_MUT_RE.match(m_str)
            if match is None:
                continue
            w_aa, pos1_s, m_aa = match.groups()
            pos0 = int(pos1_s) - 1
            if not (0 <= pos0 < len(wt_seq)) or wt_seq[pos0] != w_aa or w_aa == m_aa:
                continue
            mut_accum[(pos0, w_aa, m_aa)].append(float(ddg))

        if len(mut_accum) < min_variants_per_wt:
            continue

        # Determine stratum: de_novo_design (certified zero-hit), unseen_fold_natural, seen_pool_natural
        kind = str(qi_row.get("kind", "unknown")) if qi_row is not None else "unknown"
        qid = str(qi_row.get("query_id", "")) if qi_row is not None else ""
        hit_accs = qid_to_hits.get(qid, [])
        hit_topos = sorted(
            {cath_topology_of(acc_to_cath_h[a]) for a in hit_accs if a in acc_to_cath_h}
        )

        if kind == "design" and wt_key not in hit_design_wts:
            stratum = "de_novo_design"
        elif kind == "natural":
            if hit_topos and all(t in unseen_topologies for t in hit_topos):
                stratum = "unseen_fold_natural"
            else:
                stratum = "seen_pool_natural"
        else:
            stratum = "design_with_uniref_hit"

        variants = [
            [int(pos0), w_aa, m_aa, round(float(np.mean(vals)), 5)]
            for (pos0, w_aa, m_aa), vals in sorted(mut_accum.items())
        ]
        stratum_counts[stratum] += 1
        domains.append(
            {
                "wt_name": wt_key,
                "wildtype": wt_seq,
                "length": len(wt_seq),
                "kind": kind,
                "stratum": stratum,
                "cath_topologies": hit_topos,
                "n_uniref50_hits": len(hit_accs),
                "variants": variants,
            }
        )

    return {
        "schema": "stage2_megascale_ddg",
        "max_wt_len": int(max_wt_len),
        "n_domains": len(domains),
        "n_variants_total": int(sum(len(d["variants"]) for d in domains)),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "domains": domains,
    }


def load_swissprot_entry_to_accession(swissprot_fasta_gz: Path) -> dict[str, str]:
    """Build ``entry_name -> UniProt accession`` map from Swiss-Prot FASTA headers."""
    mapping: dict[str, str] = {}
    if not swissprot_fasta_gz.is_file():
        return mapping
    opener = gzip.open if swissprot_fasta_gz.suffix == ".gz" else open
    with opener(swissprot_fasta_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            token = line[1:].split(None, 1)[0]
            parts = token.split("|")
            if len(parts) >= 3:
                acc, entry_name = parts[1], parts[2]
                mapping[entry_name] = acc
    return mapping


def package_proteingym_benchmark(
    *,
    reference_csv: Path,
    substitutions_dir: Path,
    entry_to_acc: Mapping[str, str],
    acc_to_cath_h: Mapping[str, str],
    unseen_topologies: set[str] | frozenset[str],
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    milestone_subset_size: int = 36,
    max_variants_per_assay_milestone: int = 2_000,
    seed: int = 20261128,
) -> dict[str, Any]:
    """Package all L<=384 ProteinGym substitution assays + stratified 36-assay milestone subset."""
    import pandas as pd

    ref_df = pd.read_csv(reference_csv)
    ref_df = ref_df[ref_df["seq_len"] <= max_seq_len].copy()
    rng = np.random.default_rng(int(seed))

    assays: list[dict[str, Any]] = []
    for _, row in ref_df.sort_values("DMS_id").iterrows():
        dms_id = str(row["DMS_id"])
        csv_path = substitutions_dir / f"{dms_id}.csv"
        if not csv_path.is_file():
            continue
        wt_seq = str(row["target_seq"]).strip().upper()
        if not wt_seq or len(wt_seq) > max_seq_len or not set(wt_seq).issubset(AA20_SET):
            continue

        uniprot_entry = str(row["UniProt_ID"])
        acc = entry_to_acc.get(uniprot_entry, uniprot_entry)
        cath_h = acc_to_cath_h.get(acc, "")
        cath_t = cath_topology_of(cath_h) if cath_h else ""
        if cath_t and cath_t in unseen_topologies:
            fold_stratum = "unseen_fold"
        elif cath_t:
            fold_stratum = "seen_fold"
        else:
            fold_stratum = "unannotated"

        sub_df = pd.read_csv(csv_path, usecols=["mutant", "DMS_score"])
        variants: list[list[Any]] = []
        n_single = 0
        for m_raw, score_raw in zip(sub_df["mutant"].astype(str), sub_df["DMS_score"]):
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            parts = [p.strip().upper() for p in m_raw.split(":") if p.strip()]
            parsed: list[tuple[int, str, str]] = []
            valid_mut = True
            for tok in parts:
                m = SINGLE_MUT_RE.match(tok)
                if m is None:
                    valid_mut = False
                    break
                w_aa, pos1_s, m_aa = m.groups()
                pos0 = int(pos1_s) - 1
                if not (0 <= pos0 < len(wt_seq)) or wt_seq[pos0] != w_aa:
                    valid_mut = False
                    break
                parsed.append((pos0, w_aa, m_aa))
            if not valid_mut or not parsed:
                continue
            if len(parsed) == 1:
                n_single += 1
            variants.append([m_raw.strip().upper(), round(score, 5)])

        if len(variants) < 10:
            continue

        assays.append(
            {
                "dms_id": dms_id,
                "uniprot_id": uniprot_entry,
                "uniprot_acc": acc,
                "cluster_id": uniprot_entry,
                "coarse_selection_type": str(row.get("coarse_selection_type", "Unknown")),
                "wildtype": wt_seq,
                "length": len(wt_seq),
                "cath_superfamily": cath_h,
                "cath_topology": cath_t,
                "fold_stratum": fold_stratum,
                "n_single_variants": int(n_single),
                "n_variants": len(variants),
                "in_milestone_subset": False,
                "variants": variants,
            }
        )

    # Select stratified milestone subset (36 assays covering selection types & fold strata, unique clusters preferred)
    by_type: dict[str, list[int]] = defaultdict(list)
    for idx, a in enumerate(assays):
        by_type[a["coarse_selection_type"]].append(idx)

    chosen_indices: list[int] = []
    chosen_clusters: set[str] = set()
    # First prioritize unseen_fold assays across types, then round-robin across selection types
    for pass_unseen_only in (True, False):
        types_sorted = sorted(by_type.keys())
        ptrs = {t: 0 for t in types_sorted}
        # Order each type's assays deterministically preferring compact length (fast AR & MLM) & single-mut richness
        ordered_by_type = {
            t: sorted(
                by_type[t],
                key=lambda i: (
                    0 if assays[i]["fold_stratum"] == "unseen_fold" else 1,
                    assays[i]["length"],
                    assays[i]["dms_id"],
                ),
            )
            for t in types_sorted
        }
        while len(chosen_indices) < min(milestone_subset_size, len(assays)):
            progressed = False
            for t in types_sorted:
                if len(chosen_indices) >= min(milestone_subset_size, len(assays)):
                    break
                lst = ordered_by_type[t]
                while ptrs[t] < len(lst):
                    cand_i = lst[ptrs[t]]
                    ptrs[t] += 1
                    if cand_i in chosen_indices:
                        continue
                    a = assays[cand_i]
                    if pass_unseen_only and a["fold_stratum"] != "unseen_fold":
                        continue
                    if not pass_unseen_only and a["cluster_id"] in chosen_clusters and len(chosen_indices) < len(assays) - 5:
                        continue
                    chosen_indices.append(cand_i)
                    chosen_clusters.add(a["cluster_id"])
                    progressed = True
                    break
            if not progressed:
                break

    for idx in chosen_indices:
        assays[idx]["in_milestone_subset"] = True
        # Store deterministic subsample indices if an assay has > max_variants_per_assay_milestone
        if len(assays[idx]["variants"]) > max_variants_per_assay_milestone:
            sub_idx = np.sort(
                rng.choice(len(assays[idx]["variants"]), size=max_variants_per_assay_milestone, replace=False)
            ).tolist()
            assays[idx]["milestone_variant_indices"] = sub_idx

    stratum_counts = Counter(a["fold_stratum"] for a in assays)
    return {
        "schema": "stage2_proteingym_substitutions",
        "max_seq_len": int(max_seq_len),
        "n_assays_full": len(assays),
        "n_assays_milestone": len(chosen_indices),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "assays": assays,
    }


def parse_pdb_cb_coords(pdb_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Parse canonical sequence and C-beta (C-alpha for GLY) coordinates from a PDB / PDB.gz."""
    opener = gzip.open if pdb_path.suffix == ".gz" else open
    res_order: list[tuple[str, str, str]] = []
    res_aa: dict[tuple[str, str, str], str] = {}
    res_atoms: dict[tuple[str, str, str], dict[str, tuple[float, float, float]]] = {}
    first_chain: str | None = None

    with opener(pdb_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  "):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C", "CB"):
                continue
            alt = line[16]
            if alt not in (" ", "A", "1"):
                continue
            ch = line[21]
            if first_chain is None:
                first_chain = ch
            elif ch != first_chain:
                continue
            rk = (ch, line[22:26], line[26])
            if rk not in res_atoms:
                res_order.append(rk)
                res_aa[rk] = THREE_TO_ONE.get(line[17:20].strip(), "X")
                res_atoms[rk] = {}
            if atom_name not in res_atoms[rk]:
                res_atoms[rk][atom_name] = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )

    n_res = len(res_order)
    cb_coords = np.full((n_res, 3), np.nan, dtype=np.float32)
    cb_mask = np.zeros(n_res, dtype=np.bool_)
    chars: list[str] = []

    for i, rk in enumerate(res_order):
        aa = res_aa[rk]
        chars.append(aa)
        atoms = res_atoms[rk]
        if aa == "G":
            if "CA" in atoms:
                cb_coords[i] = atoms["CA"]
                cb_mask[i] = True
        elif "CB" in atoms:
            cb_coords[i] = atoms["CB"]
            cb_mask[i] = True
        elif "N" in atoms and "CA" in atoms and "C" in atoms:
            # Ideal virtual CB from (N, CA, C)
            n_v = np.asarray(atoms["N"], dtype=np.float64)
            ca_v = np.asarray(atoms["CA"], dtype=np.float64)
            c_v = np.asarray(atoms["C"], dtype=np.float64)
            b = ca_v - n_v
            c = c_v - ca_v
            a = np.cross(b, c)
            cb_coords[i] = (-0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_v).astype(np.float32)
            cb_mask[i] = True
        elif "CA" in atoms:
            cb_coords[i] = atoms["CA"]
            cb_mask[i] = True

    return "".join(chars), cb_coords, cb_mask


def virtual_cb_from_ncaco(coords_ncaco: np.ndarray, sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute C-beta (C-alpha for GLY) from (L, 4, 3) [N, CA, C, O] backbone coordinates."""
    arr = np.asarray(coords_ncaco, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (L, >=3, 3) coords, got {arr.shape}")
    L = len(sequence)
    if arr.shape[0] != L:
        raise ValueError(f"Sequence len {L} != coords len {arr.shape[0]}")
    n_v, ca_v, c_v = arr[:, 0, :], arr[:, 1, :], arr[:, 2, :]
    b = ca_v - n_v
    c = c_v - ca_v
    a = np.cross(b, c)
    virt_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_v

    cb = np.full((L, 3), np.nan, dtype=np.float32)
    mask = np.zeros(L, dtype=np.bool_)
    for i, ch in enumerate(sequence):
        if ch == "G":
            if np.isfinite(ca_v[i]).all():
                cb[i] = ca_v[i].astype(np.float32)
                mask[i] = True
        else:
            if np.isfinite(virt_cb[i]).all():
                cb[i] = virt_cb[i].astype(np.float32)
                mask[i] = True
            elif np.isfinite(ca_v[i]).all():
                cb[i] = ca_v[i].astype(np.float32)
                mask[i] = True
    return cb, mask


def package_unseen_fold_contacts(
    *,
    cath_3d_kv_path: Path,
    alphafold_dir: Path,
    unseen_topologies: set[str] | frozenset[str],
    output_npz: Path,
    output_meta_json: Path,
    min_len: int = 50,
    max_len: int = 256,
    target_n_domains: int = 50,
    min_plddt: float = 80.0,
) -> dict[str, Any]:
    """Package 40–60 held-out unseen-fold 3D domains with exact CB coordinates for contact eval."""
    kv = json.loads(cath_3d_kv_path.read_text(encoding="utf-8"))
    domains_map = kv.get("domains", {})

    # Group candidate domains strictly in unseen_topologies by topology for maximum fold diversity
    by_topo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for acc, rec in sorted(domains_map.items()):
        topo = str(rec.get("cath_topology", ""))
        if topo not in unseen_topologies:
            continue
        length = int(rec.get("length", 0))
        plddt = float(rec.get("mean_plddt", 0.0))
        seq = str(rec.get("aa_seq", "")).strip().upper()
        if not (min_len <= length <= max_len) or plddt < min_plddt:
            continue
        if not set(seq).issubset(AA20_SET):
            continue
        pdb_path = alphafold_dir / f"AF-{acc}-F1-model_v6.pdb.gz"
        if not pdb_path.is_file():
            pdb_path = alphafold_dir / f"AF-{acc}-F1-model_v4.pdb.gz"
        if not pdb_path.is_file():
            continue
        by_topo[topo].append(
            {
                "domain_id": acc,
                "cath_superfamily": str(rec.get("cath_superfamily", "")),
                "cath_topology": topo,
                "length": length,
                "mean_plddt": plddt,
                "pdb_path": pdb_path,
            }
        )

    for topo in by_topo:
        by_topo[topo].sort(key=lambda d: (-d["mean_plddt"], d["length"], d["domain_id"]))

    # Round-robin across unseen topologies so every domain maximizes fold diversity
    selected: list[dict[str, Any]] = []
    ptrs = {t: 0 for t in sorted(by_topo.keys())}
    topo_keys = sorted(by_topo.keys())
    while len(selected) < target_n_domains:
        progressed = False
        for t in topo_keys:
            if len(selected) >= target_n_domains:
                break
            p = ptrs[t]
            if p < len(by_topo[t]):
                selected.append(by_topo[t][p])
                ptrs[t] = p + 1
                progressed = True
        if not progressed:
            break

    meta_entries: list[dict[str, Any]] = []
    npz_arrays: dict[str, np.ndarray] = {}

    for rec in selected:
        seq, cb_coords, cb_mask = parse_pdb_cb_coords(rec["pdb_path"])
        if not (min_len <= len(seq) <= max_len) or not set(seq).issubset(AA20_SET):
            continue
        if float(cb_mask.mean()) < 0.95:
            continue
        # Verify at least 10 long-range/medium-range contacts (|i-j| >= 6, d_CB < 8.0)
        dmat = np.linalg.norm(cb_coords[:, None, :] - cb_coords[None, :, :], axis=-1)
        idx = np.arange(len(seq))
        sep = np.abs(idx[:, None] - idx[None, :])
        valid_pair = np.outer(cb_mask, cb_mask) & (sep >= 6)
        n_contacts_6 = int(((dmat < 8.0) & valid_pair).sum() // 2)
        if n_contacts_6 < 10:
            continue

        dom_id = rec["domain_id"]
        npz_arrays[f"cb_{dom_id}"] = cb_coords.astype(np.float32)
        npz_arrays[f"mask_{dom_id}"] = cb_mask.astype(np.bool_)
        meta_entries.append(
            {
                "domain_id": dom_id,
                "sequence": seq,
                "length": len(seq),
                "cath_superfamily": rec["cath_superfamily"],
                "cath_topology": rec["cath_topology"],
                "mean_plddt": float(rec["mean_plddt"]),
                "n_contacts_sep6": n_contacts_6,
                "source": "alphafold_cath_unseen_fold",
            }
        )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output_npz.stem}.", suffix=".npz", dir=str(output_npz.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **npz_arrays)
        os.replace(tmp_path, output_npz)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    meta_payload = {
        "schema": "stage2_unseen_fold_contacts",
        "n_domains": len(meta_entries),
        "n_distinct_topologies": len({e["cath_topology"] for e in meta_entries}),
        "npz_file": output_npz.name,
        "domains": meta_entries,
    }
    atomic_write_json_with_sha256(output_meta_json, meta_payload)
    return meta_payload


# --------------------------------------------------------------------------- #
# Bundle Loader
# --------------------------------------------------------------------------- #


class Stage2Bundle:
    """Reader for a portable Stage-2 dataset + benchmark bundle directory."""

    def __init__(self, bundle_dir: Path | str) -> None:
        self.root = Path(bundle_dir)
        manifest_path = self.root / "bundle_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing bundle_manifest.json in {self.root}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(
                f"Unexpected bundle schema {self.manifest.get('schema')!r}; expected {BUNDLE_SCHEMA!r}"
            )

    @property
    def arm_ids(self) -> list[str]:
        return list(self.manifest["arms"].keys())

    def load_split(self, name: str) -> tuple[PackedSequenceStore, dict[str, np.ndarray]]:
        path = self.root / "splits" / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Split {name!r} not found at {path}")
        return load_packed_split_npz(path)

    def load_arm_splits(
        self, arm_id: str
    ) -> dict[str, PackedSequenceStore]:
        if arm_id not in self.manifest["arms"]:
            raise KeyError(f"Unknown arm_id {arm_id!r}; available: {self.arm_ids}")
        train_store, _ = self.load_split(f"train_{arm_id}")
        train_sub_store, _ = self.load_split(f"eval_train_sub_{arm_id}")
        hom_store, _ = self.load_split(f"eval_unseen_homolog_{arm_id}")
        arm_unseen_path = self.root / "splits" / f"eval_unseen_fold_{arm_id}.npz"
        if arm_unseen_path.is_file():
            unseen_store, _ = self.load_split(f"eval_unseen_fold_{arm_id}")
        else:
            unseen_store, _ = self.load_split("eval_unseen_fold")
        out = {
            "train": train_store,
            "eval_train_sub": train_sub_store,
            "eval_unseen_homolog": hom_store,
            "eval_unseen_fold": unseen_store,
        }
        glob_path = self.root / "splits" / "eval_global_natural.npz"
        if glob_path.is_file():
            glob_store, _ = self.load_split("eval_global_natural")
            out["eval_global_natural"] = glob_store
        return out

    def load_megascale(self) -> dict[str, Any] | None:
        path = self.root / "benchmarks" / "megascale_ddg.json.gz"
        if not path.is_file():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def load_proteingym(self) -> dict[str, Any] | None:
        path = self.root / "benchmarks" / "proteingym_substitutions.json.gz"
        if not path.is_file():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def load_contacts(self) -> list[dict[str, Any]]:
        meta_path = self.root / "benchmarks" / "unseen_fold_contacts_meta.json"
        npz_path = self.root / "benchmarks" / "unseen_fold_contacts.npz"
        if not (meta_path.is_file() and npz_path.is_file()):
            return []
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out: list[dict[str, Any]] = []
        with np.load(npz_path, allow_pickle=False) as data:
            for entry in meta.get("domains", []):
                dom_id = str(entry["domain_id"])
                out.append(
                    {
                        **entry,
                        "cb_coords": np.asarray(data[f"cb_{dom_id}"], dtype=np.float32),
                        "cb_mask": np.asarray(data[f"mask_{dom_id}"], dtype=np.bool_),
                    }
                )
        return out
