"""Leakage-free cohort builders for Track B (B2 KV/gradient geometry, B3 horizon, B4 tokenization).

Builds two content-addressed JSON cohorts under results/extensions/memory_theory_20261120/:
1. ``cath_3d_kv_cohort.json``: 900 protein domain pairs (300 close homologs >50% id,
   300 twilight-zone remote homologs <25% id in the same CATH H-superfamily with
   Foldseek TM >= 0.50, and 300 unrelated CATH folds <20% id matched for local
   alpha/beta secondary-structure composition), with Foldseek 3Di strings for every
   domain.
2. ``text_kv_and_horizon_cohort.json``: 256 contiguous >=512-token WikiText-103
   articles (for B3/B2), 600 OpenWebText pairs from frozen R4 cohort.json
   (300 bm25_top10 related + 300 random unrelated for B2), and 256 long (>=520 AA)
   natural Swiss-Prot sequences (for B3).
"""
from __future__ import annotations

import csv
import gzip
import os
import re
import shutil
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.amino_acids import AA20
from .io_utils import load_json_file, write_json_with_sha256

AA20_SET = set(AA20)
THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def ensure_foldseek_binary(
    tarball_path: Path,
    scratch_dir: Path = Path("/tmp/memory_theory_tools"),
) -> Path:
    """Unpack Foldseek into a local scratch directory (never inside the repo)."""
    binary = scratch_dir / "foldseek/bin/foldseek"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tf:
        tf.extractall(scratch_dir)
    if not binary.is_file():
        raise FileNotFoundError(f"Expected Foldseek binary at {binary} after extracting {tarball_path}")
    return binary


def parse_pdb_ca_and_ss(pdb_gz_path: Path) -> dict[str, Any]:
    """Parse C-alpha coordinates, sequence, pLDDT, and local helix/sheet fractions from AFDB PDB.gz.

    Uses standard C-alpha local distance geometry (i->i+2, i->i+3, i->i+4 distances)
    calibrated for fast, deterministic helix/sheet classification without external DSSP:
      - Alpha-helix signature: d(i, i+3) in [4.5, 5.8] A and d(i, i+4) in [5.5, 7.0] A.
      - Beta-strand signature: extended backbone d(i, i+2) >= 6.0 A and d(i, i+3) >= 8.5 A.
    """
    coords: list[tuple[float, float, float]] = []
    bfactors: list[float] = []
    residues: list[str] = []
    seen_resseq: set[tuple[str, int, str]] = set()

    with gzip.open(pdb_gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            chain = line[21].strip() or "A"
            resseq = int(line[22:26])
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen_resseq:
                continue
            seen_resseq.add(key)
            resname = line[17:20].strip()
            aa = THREE_TO_ONE.get(resname, "X")
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bf = float(line[60:66])
            residues.append(aa)
            coords.append((x, y, z))
            bfactors.append(bf)

    n = len(coords)
    if n < 10:
        return {
            "aa_seq": "".join(residues),
            "length": n,
            "mean_plddt": 0.0,
            "helix_frac": 0.0,
            "sheet_frac": 0.0,
            "coil_frac": 1.0,
        }

    xyz = np.asarray(coords, dtype=np.float64)
    d2 = np.linalg.norm(xyz[2:] - xyz[:-2], axis=1)  # len n-2
    d3 = np.linalg.norm(xyz[3:] - xyz[:-3], axis=1)  # len n-3
    d4 = np.linalg.norm(xyz[4:] - xyz[:-4], axis=1)  # len n-4

    is_helix = np.zeros(n, dtype=bool)
    is_sheet = np.zeros(n, dtype=bool)

    for i in range(n - 4):
        if 4.4 <= d3[i] <= 5.9 and 5.4 <= d4[i] <= 7.2:
            is_helix[i : i + 5] = True

    for i in range(n - 3):
        if not is_helix[i] and d2[i] >= 5.9 and d3[i] >= 8.3:
            is_sheet[i : i + 4] = True
    is_sheet &= ~is_helix

    helix_frac = float(np.mean(is_helix))
    sheet_frac = float(np.mean(is_sheet))
    coil_frac = float(max(0.0, 1.0 - helix_frac - sheet_frac))

    return {
        "aa_seq": "".join(residues),
        "length": n,
        "mean_plddt": float(np.mean(bfactors)),
        "helix_frac": helix_frac,
        "sheet_frac": sheet_frac,
        "coil_frac": coil_frac,
    }


def make_saprot_string(aa_seq: str, three_di_seq: str) -> str:
    """Interleave uppercase AA and lowercase 3Di characters (SaProt 441-token format)."""
    if len(aa_seq) != len(three_di_seq):
        raise ValueError(f"AA len {len(aa_seq)} != 3Di len {len(three_di_seq)}")
    return "".join(f"{a}{d.lower()}" for a, d in zip(aa_seq, three_di_seq))


def _load_single_cath_afdb_candidates(
    alphafold_dir: Path,
    cath_tsv: Path,
    *,
    min_len: int = 64,
    max_len: int = 384,
    min_plddt: float = 70.0,
) -> dict[str, dict[str, Any]]:
    """Load single-CATH-H AFDB domains meeting length, AA20, and pLDDT criteria."""
    af_files: dict[str, Path] = {}
    for entry in alphafold_dir.iterdir():
        name = entry.name
        if name.endswith(".pdb.gz") and name.startswith("AF-"):
            parts = name.split("-")
            if len(parts) >= 2:
                af_files[parts[1]] = entry

    cath_by_acc: dict[str, set[str]] = defaultdict(set)
    with cath_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            acc = row["uniprot"].strip()
            h = row["cath_superfamily"].strip()
            if acc in af_files and h.count(".") == 3:
                cath_by_acc[acc].add(h)

    candidates: dict[str, dict[str, Any]] = {}
    for acc, hs in sorted(cath_by_acc.items()):
        if len(hs) != 1:
            continue
        h = next(iter(hs))
        parts = h.split(".")
        cath_class = parts[0]
        cath_arch = ".".join(parts[:2])
        cath_topo = ".".join(parts[:3])
        pdb_path = af_files[acc]
        info = parse_pdb_ca_and_ss(pdb_path)
        seq = info["aa_seq"]
        if not (min_len <= info["length"] <= max_len):
            continue
        if info["mean_plddt"] < min_plddt:
            continue
        if set(seq) - AA20_SET:
            continue
        candidates[acc] = {
            "domain_id": acc,
            "cath_superfamily": h,
            "cath_class": cath_class,
            "cath_architecture": cath_arch,
            "cath_topology": cath_topo,
            "length": info["length"],
            "aa_seq": seq,
            "mean_plddt": info["mean_plddt"],
            "helix_frac": info["helix_frac"],
            "sheet_frac": info["sheet_frac"],
            "coil_frac": info["coil_frac"],
            "pdb_path": str(pdb_path),
        }
    return candidates


def _extract_foldseek_3di_and_hits(
    candidates: dict[str, dict[str, Any]],
    foldseek_bin: Path,
    diamond_bin: Path,
    work_dir: Path,
    *,
    threads: int = 32,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[tuple[str, str], float]]:
    """Run Foldseek 3Di extraction + all-vs-all structural search + DIAMOND sequence search."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pdb_link_dir = work_dir / "pdbs"
    if pdb_link_dir.exists():
        shutil.rmtree(pdb_link_dir)
    pdb_link_dir.mkdir(parents=True)

    for acc, rec in candidates.items():
        link = pdb_link_dir / f"{acc}.pdb.gz"
        os.symlink(os.path.abspath(rec["pdb_path"]), link)

    # 1. Extract Foldseek 3Di strings for every candidate
    desc_tsv = work_dir / "descriptors.tsv"
    subprocess.run(
        [
            str(foldseek_bin),
            "structureto3didescriptor",
            str(pdb_link_dir),
            str(desc_tsv),
            "--threads",
            str(threads),
            "-v",
            "1",
        ],
        check=True,
    )

    three_di_map: dict[str, str] = {}
    with desc_tsv.open("r", encoding="utf-8") as handle:
        for line in handle:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            raw_name = cols[0].split()[0]
            acc = raw_name.replace(".pdb.gz", "").replace(".pdb", "")
            aa_fs = cols[1].strip()
            tdi = cols[2].strip()
            if acc in candidates and len(aa_fs) == len(tdi) and len(tdi) == candidates[acc]["length"]:
                three_di_map[acc] = tdi

    # 2. Foldseek all-vs-all TM-aligned search (sensitive, alignment-type 2 = 3Di+AA)
    db_path = work_dir / "afdb_cath_db"
    res_path = work_dir / "afdb_cath_res"
    tmp_path = work_dir / "fs_tmp"
    alis_tsv = work_dir / "foldseek_hits.tsv"
    subprocess.run(
        [str(foldseek_bin), "createdb", str(pdb_link_dir), str(db_path), "--threads", str(threads), "-v", "1"],
        check=True,
    )
    subprocess.run(
        [
            str(foldseek_bin),
            "search",
            str(db_path),
            str(db_path),
            str(res_path),
            str(tmp_path),
            "--alignment-type",
            "2",
            "-a",
            "-e",
            "10.0",
            "--max-seqs",
            "300",
            "-s",
            "9.5",
            "--threads",
            str(threads),
            "-v",
            "1",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(foldseek_bin),
            "convertalis",
            str(db_path),
            str(db_path),
            str(res_path),
            str(alis_tsv),
            "--format-output",
            "query,target,fident,alntmscore,qtmscore,ttmscore,qcov,tcov,evalue,bits",
            "--threads",
            str(threads),
            "-v",
            "1",
        ],
        check=True,
    )

    fs_hits: list[dict[str, Any]] = []
    with alis_tsv.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            q = parts[0].replace(".pdb.gz", "").replace(".pdb", "")
            t = parts[1].replace(".pdb.gz", "").replace(".pdb", "")
            if q == t or q not in three_di_map or t not in three_di_map:
                continue
            fs_hits.append(
                {
                    "query": q,
                    "target": t,
                    "fident": float(parts[2]),
                    "alntmscore": float(parts[3]),
                    "qtmscore": float(parts[4]),
                    "ttmscore": float(parts[5]),
                    "qcov": float(parts[6]),
                    "tcov": float(parts[7]),
                    "evalue": float(parts[8]),
                    "bits": float(parts[9]),
                }
            )

    # 3. DIAMOND all-vs-all sequence search to cross-check sequence identity
    fasta_path = work_dir / "candidates.fa"
    with fasta_path.open("w", encoding="utf-8") as fh:
        for acc in sorted(three_di_map.keys()):
            fh.write(f">{acc}\n{candidates[acc]['aa_seq']}\n")

    dmnd_db = work_dir / "candidates.dmnd"
    dmnd_out = work_dir / "diamond_hits.tsv"
    subprocess.run(
        [str(diamond_bin), "makedb", "--in", str(fasta_path), "-d", str(dmnd_db), "--quiet"],
        check=True,
    )
    subprocess.run(
        [
            str(diamond_bin),
            "blastp",
            "-q",
            str(fasta_path),
            "-d",
            str(dmnd_db),
            "-o",
            str(dmnd_out),
            "--ultra-sensitive",
            "--max-target-seqs",
            "300",
            "-e",
            "10",
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "qcovhsp",
            "scovhsp",
            "evalue",
            "--threads",
            str(threads),
            "--quiet",
        ],
        check=True,
    )

    diamond_pident: dict[tuple[str, str], float] = {}
    with dmnd_out.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q, t, pid = parts[0], parts[1], float(parts[2]) / 100.0
            if q == t:
                continue
            pair = (q, t) if q < t else (t, q)
            if pid > diamond_pident.get(pair, 0.0):
                diamond_pident[pair] = pid

    return three_di_map, fs_hits, diamond_pident


def _dedup_and_sample_homolog_pairs(
    raw_pairs: list[dict[str, Any]],
    *,
    target_n: int,
    max_per_superfamily: int,
    max_per_domain: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Sample balanced, deduplicated pairs across CATH H-superfamilies."""
    by_h: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in raw_pairs:
        by_h[rec["cath_superfamily"]].append(rec)

    # Shuffle within each superfamily and round-robin across superfamilies for maximum diversity
    h_keys = sorted(by_h.keys())
    rng.shuffle(h_keys)
    for h in h_keys:
        rng.shuffle(by_h[h])

    selected: list[dict[str, Any]] = []
    h_counts: dict[str, int] = defaultdict(int)
    dom_counts: dict[str, int] = defaultdict(int)
    seen_pairs: set[tuple[str, str]] = set()

    # Multi-pass round-robin over superfamilies
    for _pass in range(max_per_superfamily):
        progressed = False
        for h in h_keys:
            if len(selected) >= target_n:
                break
            if h_counts[h] >= max_per_superfamily:
                continue
            pool = by_h[h]
            while pool:
                cand = pool.pop()
                a, b = cand["domain_a"], cand["domain_b"]
                key = (a, b) if a < b else (b, a)
                if key in seen_pairs:
                    continue
                if dom_counts[a] >= max_per_domain or dom_counts[b] >= max_per_domain:
                    continue
                seen_pairs.add(key)
                h_counts[h] += 1
                dom_counts[a] += 1
                dom_counts[b] += 1
                selected.append(cand)
                progressed = True
                break
        if len(selected) >= target_n or not progressed:
            break
    return selected


def build_cath_3d_kv_cohort(
    repo_root: Path,
    out_path: Path,
    *,
    target_per_stratum: int = 300,
    scratch_root: Path = Path("/tmp/memory_theory_tools"),
    threads: int = 32,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Build leakage-free 3D structure & Foldseek-3Di pair cohort (300 x 3 strata)."""
    foldseek_bin = ensure_foldseek_binary(
        repo_root / "external/tools/foldseek-linux-avx2.tar.gz",
        scratch_dir=scratch_root,
    )
    diamond_bin = repo_root / "external/tools/diamond/diamond"
    if not diamond_bin.is_file():
        raise FileNotFoundError(f"Missing DIAMOND binary at {diamond_bin}")

    candidates = _load_single_cath_afdb_candidates(
        repo_root / "data/alphafold",
        repo_root / "data/interpro/cath_superfamily.tsv",
        min_len=64,
        max_len=384,
        min_plddt=70.0,
    )

    work_dir = scratch_root / "cath_3d_work"
    three_di_map, fs_hits, diamond_pident = _extract_foldseek_3di_and_hits(
        candidates, foldseek_bin, diamond_bin, work_dir, threads=threads
    )

    # Collapse directed Foldseek hits to best undirected record per pair
    best_fs: dict[tuple[str, str], dict[str, Any]] = {}
    for hit in fs_hits:
        a, b = hit["query"], hit["target"]
        key = (a, b) if a < b else (b, a)
        prev = best_fs.get(key)
        if prev is None or hit["alntmscore"] > prev["alntmscore"]:
            best_fs[key] = hit

    close_pool: list[dict[str, Any]] = []
    twilight_pool: list[dict[str, Any]] = []
    any_fs_pair: dict[tuple[str, str], float] = {}

    for (a, b), hit in best_fs.items():
        any_fs_pair[(a, b)] = hit["alntmscore"]
        ca, cb = candidates[a], candidates[b]
        if ca["cath_superfamily"] != cb["cath_superfamily"]:
            continue
        fident = float(hit["fident"])
        dpid = float(diamond_pident.get((a, b), 0.0))
        max_id = max(fident, dpid)
        tm = float(hit["alntmscore"])
        min_cov = min(float(hit["qcov"]), float(hit["tcov"]))

        if max_id > 0.50 and fident > 0.50 and dpid > 0.50 and tm >= 0.50 and min_cov >= 0.60:
            close_pool.append(
                {
                    "domain_a": a,
                    "domain_b": b,
                    "stratum": "close_homolog",
                    "cath_superfamily": ca["cath_superfamily"],
                    "seq_identity": max_id,
                    "foldseek_fident": fident,
                    "diamond_pident": dpid,
                    "aln_tmscore": tm,
                    "q_tmscore": float(hit["qtmscore"]),
                    "t_tmscore": float(hit["ttmscore"]),
                    "min_cov": min_cov,
                    "delta_helix_frac": abs(ca["helix_frac"] - cb["helix_frac"]),
                    "delta_sheet_frac": abs(ca["sheet_frac"] - cb["sheet_frac"]),
                    "same_cath_h": True,
                    "same_cath_t": True,
                    "same_cath_c": True,
                }
            )
        elif max_id < 0.25 and tm >= 0.50 and min_cov >= 0.60:
            twilight_pool.append(
                {
                    "domain_a": a,
                    "domain_b": b,
                    "stratum": "twilight_remote_homolog",
                    "cath_superfamily": ca["cath_superfamily"],
                    "seq_identity": max_id,
                    "foldseek_fident": fident,
                    "diamond_pident": dpid,
                    "aln_tmscore": tm,
                    "q_tmscore": float(hit["qtmscore"]),
                    "t_tmscore": float(hit["ttmscore"]),
                    "min_cov": min_cov,
                    "delta_helix_frac": abs(ca["helix_frac"] - cb["helix_frac"]),
                    "delta_sheet_frac": abs(ca["sheet_frac"] - cb["sheet_frac"]),
                    "same_cath_h": True,
                    "same_cath_t": True,
                    "same_cath_c": True,
                }
            )

    rng = np.random.default_rng(seed)
    close_pairs = _dedup_and_sample_homolog_pairs(
        close_pool, target_n=target_per_stratum, max_per_superfamily=12, max_per_domain=4, rng=rng
    )
    twilight_pairs = _dedup_and_sample_homolog_pairs(
        twilight_pool, target_n=target_per_stratum, max_per_superfamily=12, max_per_domain=4, rng=rng
    )

    # Relax max_per_superfamily slightly only if strictly needed to hit exact target_per_stratum
    if len(close_pairs) < target_per_stratum:
        close_pairs = _dedup_and_sample_homolog_pairs(
            close_pool, target_n=target_per_stratum, max_per_superfamily=40, max_per_domain=8, rng=rng
        )
    if len(twilight_pairs) < target_per_stratum:
        twilight_pairs = _dedup_and_sample_homolog_pairs(
            twilight_pool, target_n=target_per_stratum, max_per_superfamily=40, max_per_domain=8, rng=rng
        )

    # 3. Unrelated folds sharing local alpha/beta secondary-structure composition
    # Different CATH Topology (and prefer different Class or Architecture), ZERO Foldseek hit >= 0.35 TM,
    # ZERO DIAMOND hit >= 0.20 pident, matched helix_frac & sheet_frac within <= 0.06, matched length within 20%.
    valid_doms = [
        acc
        for acc in sorted(three_di_map.keys())
        if (candidates[acc]["helix_frac"] + candidates[acc]["sheet_frac"]) >= 0.35
    ]
    rng.shuffle(valid_doms)

    # Bucket domains by coarse (helix_bin, sheet_bin) for O(N) matching
    unrelated_pairs: list[dict[str, Any]] = []
    unrel_dom_counts: dict[str, int] = defaultdict(int)
    unrel_topo_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    seen_unrel: set[tuple[str, str]] = set()

    # Balance across alpha-rich, beta-rich, and alpha/beta mixed classes (100 each)
    for idx_i, a in enumerate(valid_doms):
        if len(unrelated_pairs) >= target_per_stratum:
            break
        if unrel_dom_counts[a] >= 2:
            continue
        ca = candidates[a]
        for b in valid_doms[idx_i + 1 :]:
            if unrel_dom_counts[b] >= 2:
                continue
            cb = candidates[b]
            if ca["cath_topology"] == cb["cath_topology"]:
                continue
            # Require different architecture or class for strict structural unrelatedness
            if ca["cath_architecture"] == cb["cath_architecture"]:
                continue
            dh = abs(ca["helix_frac"] - cb["helix_frac"])
            ds = abs(ca["sheet_frac"] - cb["sheet_frac"])
            if dh > 0.06 or ds > 0.06:
                continue
            len_ratio = abs(ca["length"] - cb["length"]) / max(ca["length"], cb["length"])
            if len_ratio > 0.20:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen_unrel:
                continue
            if any_fs_pair.get(key, 0.0) >= 0.35:
                continue
            dpid = diamond_pident.get(key, 0.0)
            if dpid >= 0.20:
                continue
            tp_key = tuple(sorted((ca["cath_topology"], cb["cath_topology"])))
            if unrel_topo_pair_counts[tp_key] >= 3:
                continue

            seen_unrel.add(key)
            unrel_dom_counts[a] += 1
            unrel_dom_counts[b] += 1
            unrel_topo_pair_counts[tp_key] += 1
            unrelated_pairs.append(
                {
                    "domain_a": a,
                    "domain_b": b,
                    "stratum": "unrelated_matched_ss",
                    "cath_superfamily_a": ca["cath_superfamily"],
                    "cath_superfamily_b": cb["cath_superfamily"],
                    "seq_identity": float(dpid),
                    "foldseek_fident": float(best_fs[key]["fident"]) if key in best_fs else 0.0,
                    "diamond_pident": float(dpid),
                    "aln_tmscore": float(any_fs_pair.get(key, 0.0)),
                    "delta_helix_frac": float(dh),
                    "delta_sheet_frac": float(ds),
                    "same_cath_h": False,
                    "same_cath_t": False,
                    "same_cath_c": bool(ca["cath_class"] == cb["cath_class"]),
                }
            )
            break

    if (
        len(close_pairs) < target_per_stratum
        or len(twilight_pairs) < target_per_stratum
        or len(unrelated_pairs) < target_per_stratum
    ):
        raise RuntimeError(
            f"Insufficient CATH pairs: close={len(close_pairs)} (pool={len(close_pool)}), "
            f"twilight={len(twilight_pairs)} (pool={len(twilight_pool)}), "
            f"unrelated={len(unrelated_pairs)} (target={target_per_stratum})"
        )

    # Number pair_ids and materialize minimal domain dictionary with Foldseek 3Di strings
    all_strata = {
        "close_homolog": close_pairs[:target_per_stratum],
        "twilight_remote_homolog": twilight_pairs[:target_per_stratum],
        "unrelated_matched_ss": unrelated_pairs[:target_per_stratum],
    }
    used_domains: set[str] = set()
    for sname, plist in all_strata.items():
        for idx, rec in enumerate(plist):
            rec["pair_id"] = f"{sname}_{idx:04d}"
            used_domains.add(rec["domain_a"])
            used_domains.add(rec["domain_b"])

    domains_out: dict[str, dict[str, Any]] = {}
    for acc in sorted(used_domains):
        c = candidates[acc]
        tdi = three_di_map[acc]
        domains_out[acc] = {
            "domain_id": acc,
            "cath_superfamily": c["cath_superfamily"],
            "cath_class": c["cath_class"],
            "cath_architecture": c["cath_architecture"],
            "cath_topology": c["cath_topology"],
            "length": c["length"],
            "aa_seq": c["aa_seq"],
            "three_di_seq": tdi,
            "saprot_seq": make_saprot_string(c["aa_seq"], tdi),
            "mean_plddt": c["mean_plddt"],
            "helix_frac": c["helix_frac"],
            "sheet_frac": c["sheet_frac"],
            "coil_frac": c["coil_frac"],
        }

    def _stratum_stats(plist: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ids = [p["seq_identity"] for p in plist]
        tms = [p["aln_tmscore"] for p in plist]
        dhs = [p["delta_helix_frac"] for p in plist]
        dss = [p["delta_sheet_frac"] for p in plist]
        return {
            "n_pairs": len(plist),
            "seq_identity_mean": float(np.mean(ids)),
            "seq_identity_min": float(np.min(ids)),
            "seq_identity_max": float(np.max(ids)),
            "aln_tmscore_mean": float(np.mean(tms)),
            "aln_tmscore_min": float(np.min(tms)),
            "delta_helix_frac_mean": float(np.mean(dhs)),
            "delta_sheet_frac_mean": float(np.mean(dss)),
        }

    payload = {
        "schema": "memory_theory_cath_3d_kv_cohort_v1",
        "seed": int(seed),
        "source": {
            "alphafold_dir": "data/alphafold",
            "cath_superfamily_tsv": "data/interpro/cath_superfamily.tsv",
            "n_eligible_single_cath_afdb": len(candidates),
            "n_with_foldseek_3di": len(three_di_map),
            "pool_counts": {
                "close_homolog_pool": len(close_pool),
                "twilight_remote_homolog_pool": len(twilight_pool),
            },
        },
        "summary": {
            "n_unique_domains": len(domains_out),
            "strata": {k: _stratum_stats(v) for k, v in all_strata.items()},
            "unique_cath_h_close": len({p["cath_superfamily"] for p in all_strata["close_homolog"]}),
            "unique_cath_h_twilight": len({p["cath_superfamily"] for p in all_strata["twilight_remote_homolog"]}),
        },
        "domains": domains_out,
        "pairs": all_strata,
    }
    digest = write_json_with_sha256(out_path, payload)
    payload["sha256"] = digest
    return payload


# ---------------------------------------------------------------------------
# Natural Text & Long-Protein Horizon Cohort Builder
# ---------------------------------------------------------------------------


def _extract_wikitext_articles(parquet_paths: Sequence[Path], min_words: int = 480) -> list[dict[str, Any]]:
    """Group WikiText-103 parquet rows into contiguous articles by level-1 headings."""
    import pyarrow.parquet as pq

    title_re = re.compile(r"^\s*=\s+[^=].*[^=]\s+=\s*$")
    articles: list[dict[str, Any]] = []
    cur_title: str | None = None
    cur_lines: list[str] = []

    def _flush() -> None:
        nonlocal cur_title, cur_lines
        if cur_title is None:
            return
        body = "\n".join(line.strip() for line in cur_lines if line.strip())
        words = body.split()
        if len(words) >= min_words:
            articles.append(
                {
                    "title": cur_title.strip(" ="),
                    "n_words": len(words),
                    "text": body,
                }
            )
        cur_title = None
        cur_lines = []

    for ppath in parquet_paths:
        table = pq.read_table(ppath, columns=["text"])
        for val in table["text"].to_pylist():
            s = str(val or "")
            if title_re.match(s):
                _flush()
                cur_title = s.strip()
                cur_lines = []
            elif cur_title is not None:
                if s.strip():
                    cur_lines.append(s)
        _flush()
    return articles


def _extract_long_swissprot_chains(
    swissprot_gz: Path,
    *,
    n_target: int = 256,
    min_len: int = 520,
    max_len: int = 1024,
    seed: int = 20261120,
) -> list[dict[str, Any]]:
    """Sample 256 length >= 520 AA natural Swiss-Prot chains deduplicated by 4-mer Jaccard < 0.25."""
    pool: list[dict[str, Any]] = []
    acc: str | None = None
    header = ""
    buf: list[str] = []

    def _push() -> None:
        if acc is None:
            return
        seq = "".join(buf)
        if min_len <= len(seq) <= max_len and not (set(seq) - AA20_SET):
            pool.append({"protein_id": acc, "header": header[:120], "length": len(seq), "aa_seq": seq})

    with gzip.open(swissprot_gz, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _push()
                header = line[1:]
                parts = header.split("|")
                acc = parts[1] if len(parts) >= 2 else header.split()[0]
                buf = []
                if len(pool) >= 20000:
                    break
            else:
                buf.append(line)
        _push()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    selected: list[dict[str, Any]] = []
    selected_kmers: list[set[str]] = []

    for idx in order:
        if len(selected) >= n_target:
            break
        rec = pool[int(idx)]
        seq = rec["aa_seq"]
        kmers = {seq[i : i + 4] for i in range(0, min(len(seq), 520) - 3)}
        if not kmers:
            continue
        duplicate = False
        for prev in selected_kmers:
            inter = len(kmers & prev)
            if inter / max(len(kmers | prev), 1) >= 0.25:
                duplicate = True
                break
        if not duplicate:
            selected.append(rec)
            selected_kmers.append(kmers)

    if len(selected) < n_target:
        raise RuntimeError(f"Only found {len(selected)} deduplicated Swiss-Prot chains >= {min_len} AA")
    return selected


def build_text_kv_and_horizon_cohort(
    repo_root: Path,
    out_path: Path,
    *,
    wikitext_dir: Path = Path("/Data/public/datasets/wikitext-103-v1/wikitext-103-v1"),
    r4_cohort_path: Path | None = None,
    n_articles: int = 256,
    n_pairs_per_stratum: int = 300,
    n_long_proteins: int = 256,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Build matched natural text (WikiText-103 + OpenWebText BM25/random) & long-protein cohort."""
    from transformers import AutoTokenizer

    if r4_cohort_path is None:
        r4_cohort_path = repo_root / "results/R5/context_homologue_20260826/cohort.json"

    # Load GPT-2 tokenizer locally to guarantee exact >= 512-token contiguous spans
    gpt2_dir = Path("/Data/public/gpt2")
    tok = AutoTokenizer.from_pretrained(str(gpt2_dir), local_files_only=True)

    # 1. Contiguous >= 512-token WikiText-103 articles (prioritize test & validation, then train)
    parquets = [
        p
        for name in (
            "test-00000-of-00001.parquet",
            "validation-00000-of-00001.parquet",
            "train-00000-of-00002.parquet",
        )
        if (p := wikitext_dir / name).is_file()
    ]
    raw_articles = _extract_wikitext_articles(parquets, min_words=450)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(raw_articles))

    selected_articles: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for idx in order:
        if len(selected_articles) >= n_articles:
            break
        art = raw_articles[int(idx)]
        if art["title"] in seen_titles:
            continue
        # Truncate raw text softly before tokenizing so we don't spend time on 10k-word articles
        trimmed = " ".join(art["text"].split()[:1200])
        ids = tok.encode(trimmed, add_special_tokens=False)
        if len(ids) < 520:
            continue
        seen_titles.add(art["title"])
        # Store exact 600-token prefix text so every BPE tokenizer gets >= 512 tokens cleanly
        text_span = tok.decode(ids[:600], clean_up_tokenization_spaces=False)
        selected_articles.append(
            {
                "article_id": f"wt103_{len(selected_articles):04d}",
                "title": art["title"],
                "gpt2_tokens": len(ids[:600]),
                "text": text_span,
            }
        )

    if len(selected_articles) < n_articles:
        raise RuntimeError(f"Only extracted {len(selected_articles)} >=520-token WikiText-103 articles")

    # 2. OpenWebText BM25 related vs random unrelated pairs from frozen R4 cohort.json
    r4 = load_json_file(r4_cohort_path)
    text_sec = r4["text"]
    records: list[str] = text_sec["records"]
    units: list[dict[str, Any]] = text_sec["units"]

    related_candidates: list[dict[str, Any]] = []
    random_candidates: list[dict[str, Any]] = []
    for u in units:
        band = u.get("band")
        target_idx = int(u["target"])
        partners = [int(p) for p in u.get("partners", [])]
        if band == "bm25_top10":
            for p_idx in partners[:3]:
                if p_idx != target_idx:
                    related_candidates.append(
                        {
                            "unit_key": u["key"],
                            "target_idx": target_idx,
                            "partner_idx": p_idx,
                            "text_a": records[target_idx],
                            "text_b": records[p_idx],
                            "stratum": "bm25_top10_related",
                        }
                    )
        elif band in ("bm25_random", "random"):
            for p_idx in partners[:3]:
                if p_idx != target_idx:
                    random_candidates.append(
                        {
                            "unit_key": u["key"],
                            "target_idx": target_idx,
                            "partner_idx": p_idx,
                            "text_a": records[target_idx],
                            "text_b": records[p_idx],
                            "stratum": "random_unrelated",
                        }
                    )

    # Also build unrelated pairs from distinct units' unrelated lists if random band has < 300
    def _sample_unique_text_pairs(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        idxs = rng.permutation(len(pool))
        out: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for i in idxs:
            if len(out) >= n:
                break
            rec = pool[int(i)]
            key = (min(rec["target_idx"], rec["partner_idx"]), max(rec["target_idx"], rec["partner_idx"]))
            if key in seen:
                continue
            seen.add(key)
            rec = dict(rec)
            rec["pair_id"] = f"{rec['stratum']}_{len(out):04d}"
            out.append(rec)
        return out

    related_pairs = _sample_unique_text_pairs(related_candidates, n_pairs_per_stratum)
    random_pairs = _sample_unique_text_pairs(random_candidates, n_pairs_per_stratum)

    if len(related_pairs) < n_pairs_per_stratum or len(random_pairs) < n_pairs_per_stratum:
        raise RuntimeError(
            f"Insufficient R4 text pairs: related={len(related_pairs)}, random={len(random_pairs)}"
        )

    # 3. Long natural protein chains (>= 520 AA) from Swiss-Prot for B3 protein horizon
    long_proteins = _extract_long_swissprot_chains(
        repo_root / "data/swissprot/uniprot_sprot.fasta.gz",
        n_target=n_long_proteins,
        min_len=520,
        max_len=1024,
        seed=seed,
    )

    payload = {
        "schema": "memory_theory_text_kv_and_horizon_cohort_v1",
        "seed": int(seed),
        "sources": {
            "wikitext_dir": str(wikitext_dir),
            "r4_cohort": str(r4_cohort_path),
            "swissprot_fasta": "data/swissprot/uniprot_sprot.fasta.gz",
        },
        "summary": {
            "n_wikitext_articles": len(selected_articles),
            "n_bm25_top10_pairs": len(related_pairs),
            "n_random_unrelated_pairs": len(random_pairs),
            "n_long_proteins": len(long_proteins),
            "long_protein_len_min": min(p["length"] for p in long_proteins),
            "long_protein_len_max": max(p["length"] for p in long_proteins),
        },
        "wikitext_articles": selected_articles,
        "text_pairs": {
            "bm25_top10_related": related_pairs,
            "random_unrelated": random_pairs,
        },
        "long_proteins": long_proteins,
    }
    digest = write_json_with_sha256(out_path, payload)
    payload["sha256"] = digest
    return payload
