#!/usr/bin/env python3
"""Build a compact, deterministic Stage-2 Knob-2 dataset + downstream benchmark bundle.

Produces a portable output directory (target <400 MB) containing:
  - Strict C.A.T fold-disjoint ``eval_unseen_fold`` (2,048 seqs across >=80 held-out folds)
  - Token-matched (M, d) training arms (default 5.0M AA tokens each, <0.2% budget match)
    + per-arm ``eval_train_sub`` (1,024 seqs) + exact same-family ``eval_unseen_homolog`` (1,024 seqs)
  - Fixed ``eval_global_natural`` (2,048 seqs)
  - Downstream benchmarks: MegaScale ddG (with de_novo / unseen_fold / seen_pool strata),
    ProteinGym L<=384 substitutions (with 36-assay milestone subset), and unseen-fold 3D contacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.supply_demand.corpus import (  # noqa: E402
    BUNDLE_SCHEMA,
    AnnotatedSequence,
    _now_utc,
    atomic_write_gzip_json,
    atomic_write_json_with_sha256,
    build_controlled_natural_splits,
    cath_topology_of,
    extract_uniref50_cath_table,
    iter_uniref50_fasta,
    load_single_cath_mapping_tsv,
    load_swissprot_entry_to_accession,
    package_megascale_benchmark,
    package_proteingym_benchmark,
    package_unseen_fold_contacts,
    save_packed_split_npz,
    seq_sha256,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic Stage-2 Knob-2 dataset + benchmark bundle.")
    p.add_argument("--data-root", type=Path, default=Path("/Data/lzp/plm-capability-boundaries"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--prejoined-parquet",
        type=Path,
        default=None,
        help="Optional pre-joined UniRef50-CATH parquet (columns: uniref50_id, accession, cath_h, cath_t, n_cath_t, seq_len, sequence) "
        "to skip streaming uniref50.fasta + protein2ipr.dat.gz for CATH-annotated sequences.",
    )
    p.add_argument(
        "--cath-table",
        type=Path,
        default=None,
        help="Optional pre-extracted TSV(.gz) mapping UniProt acc -> CATH C.A.T.H. "
        "If omitted, uses protein2ipr.dat.gz when --use-full-protein2ipr is set, else cath_superfamily.tsv.",
    )
    p.add_argument(
        "--use-full-protein2ipr",
        action="store_true",
        help="Stream full data/interpro/protein2ipr.dat.gz to annotate all UniRef50 UniProt reps.",
    )
    p.add_argument(
        "--save-extracted-cath",
        type=Path,
        default=None,
        help="Optional path to cache extracted UniRef50-CATH TSV.gz for future rebuilds.",
    )
    p.add_argument("--target-unique-tokens", "--target-tokens", dest="target_unique_tokens", type=int, default=5_000_000)
    p.add_argument("--arm-families", type=str, default="64,256,1024,4096")
    p.add_argument(
        "--allow-family-cap",
        action="store_true",
        help="Allow capping largest requested M to available train-eligible CATH H-superfamilies "
        "(explicitly named M<eff>_all_cath; rejects duplicate realized M).",
    )
    p.add_argument("--min-seq-len", type=int, default=50)
    p.add_argument("--max-seq-len", type=int, default=384)
    p.add_argument("--max-seqs-per-superfamily", type=int, default=2048)
    p.add_argument("--singleton-pool-cap", type=int, default=50_000)
    p.add_argument("--n-unseen-fold", type=int, default=2048)
    p.add_argument("--min-unseen-topologies", type=int, default=80)
    p.add_argument("--n-homolog-eval", type=int, default=1024)
    p.add_argument("--n-global-natural", type=int, default=2048)
    p.add_argument("--n-contact-domains", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--skip-benchmarks", action="store_true")
    return p.parse_args()


def _collect_eval_wt_hashes_and_priority_topos(
    data_root: Path,
    acc_to_cath_h: dict[str, str],
    entry_to_acc: dict[str, str],
) -> tuple[set[str], list[str]]:
    """Collect exact eval WT hashes to exclude from training and priority held-out C.A.T folds."""
    import pandas as pd

    excluded_hashes: set[str] = set()
    topo_scores: dict[str, int] = defaultdict(int)

    # 1. cath_3d_kv_cohort.json -> prioritize folds with 50<=L<=256 high-pLDDT AFDB domains
    kv_path = data_root / "results/extensions/memory_theory_20261120/cath_3d_kv_cohort.json"
    if kv_path.is_file():
        kv = json.loads(kv_path.read_text(encoding="utf-8"))
        for rec in kv.get("domains", {}).values():
            seq = str(rec.get("aa_seq", "")).strip().upper()
            topo = str(rec.get("cath_topology", ""))
            length = int(rec.get("length", 0))
            plddt = float(rec.get("mean_plddt", 0.0))
            if topo and 50 <= length <= 256 and plddt >= 80.0:
                topo_scores[topo] += 3
            if seq:
                excluded_hashes.add(seq_sha256(seq))

    # 2. MegaScale WTs & natural CATH hits
    ms_dir = data_root / "results/shared/megascale_disjointness_20260813"
    qi_path = ms_dir / "query_index.json"
    hits_path = ms_dir / "hits.tsv"
    if qi_path.is_file():
        qi = json.loads(qi_path.read_text(encoding="utf-8"))
        qid_kind = {}
        for row in qi:
            seq = str(row.get("sequence", "")).strip().upper()
            if seq:
                excluded_hashes.add(seq_sha256(seq))
            qid_kind[str(row["query_id"])] = str(row.get("kind", ""))
        if hits_path.is_file():
            with hits_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 10:
                        continue
                    qid, subj = parts[0], parts[1]
                    if qid_kind.get(qid) != "natural":
                        continue
                    try:
                        ev = float(parts[9])
                    except ValueError:
                        continue
                    if ev <= 1e-3:
                        acc = subj[len("UniRef50_") :] if subj.startswith("UniRef50_") else subj
                        sf = acc_to_cath_h.get(acc)
                        if sf:
                            topo_scores[cath_topology_of(sf)] += 2

    # 3. ProteinGym L<=384 WTs
    pg_ref = data_root / "data/proteingym_raw/DMS_substitutions.csv"
    if pg_ref.is_file():
        df = pd.read_csv(pg_ref)
        for _, row in df.iterrows():
            seq = str(row["target_seq"]).strip().upper()
            if seq:
                excluded_hashes.add(seq_sha256(seq))
            if int(row["seq_len"]) <= 384:
                entry = str(row["UniProt_ID"])
                acc = entry_to_acc.get(entry, entry)
                sf = acc_to_cath_h.get(acc)
                if sf:
                    topo_scores[cath_topology_of(sf)] += 2

    priority_topos = [t for t, _ in sorted(topo_scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    return excluded_hashes, priority_topos


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    splits_dir = out_dir / "splits"
    bench_dir = out_dir / "benchmarks"
    splits_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    uniref_fasta = data_root / "data/uniref50/uniref50.fasta"
    protein2ipr_gz = data_root / "data/interpro/protein2ipr.dat.gz"
    default_cath_tsv = data_root / "data/interpro/cath_superfamily.tsv"
    swissprot_gz = data_root / "data/swissprot/uniprot_sprot.fasta.gz"

    print(f"[stage2-bundle] Loading Swiss-Prot entry->acc map from {swissprot_gz}", flush=True)
    entry_to_acc = load_swissprot_entry_to_accession(swissprot_gz)

    all_cath_annotated_accs: set[str] = set()
    if args.cath_table is not None:
        print(f"[stage2-bundle] Loading CATH table from {args.cath_table}", flush=True)
        acc_to_cath_h, all_cath_annotated_accs = load_single_cath_mapping_tsv(
            args.cath_table, return_all_annotated=True
        )
    elif args.use_full_protein2ipr and args.prejoined_parquet is None:
        print(f"[stage2-bundle] Pass 1: collecting UniRef50 UniProt accessions ({args.min_seq_len}<=L<={args.max_seq_len})", flush=True)
        uniref_accs: set[str] = set()
        for _cid, acc, _seq in iter_uniref50_fasta(
            uniref_fasta, min_len=args.min_seq_len, max_len=args.max_seq_len
        ):
            if not acc.startswith("UPI"):
                uniref_accs.add(acc)
        print(f"[stage2-bundle] Collected {len(uniref_accs):,} non-UniParc UniRef50 accessions; streaming {protein2ipr_gz}", flush=True)
        acc_to_cath_h, all_cath_annotated_accs = extract_uniref50_cath_table(
            protein2ipr_gz,
            uniref_accs,
            output_tsv_gz=args.save_extracted_cath,
            return_all_annotated=True,
        )
    else:
        print(f"[stage2-bundle] Loading Swiss-Prot CATH table from {default_cath_tsv}", flush=True)
        acc_to_cath_h, all_cath_annotated_accs = load_single_cath_mapping_tsv(
            default_cath_tsv, return_all_annotated=True
        )

    # Load pre-joined parquet BEFORE computing priority_topos so TrEMBL CATH topologies
    # inform held-out selection AND multi-topology accessions are excluded from singletons.
    rng = np.random.default_rng(int(args.seed))
    fam_reservoirs: dict[str, list[AnnotatedSequence]] = defaultdict(list)
    fam_seen: dict[str, int] = defaultdict(int)
    singleton_reservoir: list[AnnotatedSequence] = []
    singleton_seen = 0
    total_scanned = 0

    if args.prejoined_parquet is not None:
        import pandas as pd

        print(f"[stage2-bundle] Loading pre-joined UniRef50-CATH parquet from {args.prejoined_parquet}", flush=True)
        pq_df = pd.read_parquet(args.prejoined_parquet)
        # Track ALL parquet-annotated accessions (including multi-topology n_cath_t > 1)
        # so multi-domain proteins containing held-out folds NEVER enter singleton_reservoir.
        all_cath_annotated_accs.update(str(a) for a in pq_df["accession"].astype(str))
        if "n_cath_t" in pq_df.columns:
            pq_single = pq_df[pq_df["n_cath_t"] == 1]
        else:
            pq_single = pq_df
        pq_single = pq_single[
            (pq_single["seq_len"] >= args.min_seq_len) & (pq_single["seq_len"] <= args.max_seq_len)
        ]
        for row in pq_single.itertuples(index=False):
            cid = str(row.uniref50_id)
            acc = str(row.accession)
            sf = str(row.cath_h)
            topo = str(row.cath_t) if getattr(row, "cath_t", "") else cath_topology_of(sf)
            seq = str(row.sequence).strip().upper()
            acc_to_cath_h[acc] = sf
            fam_seen[sf] += 1
            res = fam_reservoirs[sf]
            rec = AnnotatedSequence(seq_id=cid, sequence=seq, cath_h=sf, cath_t=topo)
            if len(res) < args.max_seqs_per_superfamily:
                res.append(rec)
            else:
                j = int(rng.integers(0, fam_seen[sf]))
                if j < args.max_seqs_per_superfamily:
                    res[j] = rec
            total_scanned += 1

    print(
        f"[stage2-bundle] Loaded {len(acc_to_cath_h):,} unambiguous single-CATH-H accessions "
        f"({len(all_cath_annotated_accs):,} total CATH-annotated excluded from singletons)",
        flush=True,
    )

    excluded_hashes, priority_topos = _collect_eval_wt_hashes_and_priority_topos(
        data_root, acc_to_cath_h, entry_to_acc
    )
    print(
        f"[stage2-bundle] Excluding {len(excluded_hashes):,} downstream WT hashes; "
        f"{len(priority_topos):,} priority unseen-fold candidates",
        flush=True,
    )

    if args.prejoined_parquet is not None:
        # Collect disjoint unannotated UniRef50 singleton reps for M_max_singleton (fast early-exit scan)
        target_sing_scan = max(args.singleton_pool_cap * 3, 100_000)
        if uniref_fasta.is_file():
            print(f"[stage2-bundle] Sampling {args.singleton_pool_cap:,} UniRef50 singletons from {uniref_fasta} ...", flush=True)
            for cid, acc, seq in iter_uniref50_fasta(
                uniref_fasta, min_len=args.min_seq_len, max_len=args.max_seq_len
            ):
                if acc in all_cath_annotated_accs:
                    continue
                singleton_seen += 1
                if len(singleton_reservoir) < args.singleton_pool_cap:
                    singleton_reservoir.append(
                        AnnotatedSequence(seq_id=cid, sequence=seq, cath_h="", cath_t="")
                    )
                else:
                    j = int(rng.integers(0, singleton_seen))
                    if j < args.singleton_pool_cap:
                        singleton_reservoir[j] = AnnotatedSequence(
                            seq_id=cid, sequence=seq, cath_h="", cath_t=""
                        )
                if singleton_seen >= target_sing_scan:
                    break
    else:
        print(f"[stage2-bundle] Streaming {uniref_fasta} ...", flush=True)
        for cid, acc, seq in iter_uniref50_fasta(
            uniref_fasta, min_len=args.min_seq_len, max_len=args.max_seq_len
        ):
            total_scanned += 1
            sf = acc_to_cath_h.get(acc)
            if sf is not None:
                fam_seen[sf] += 1
                res = fam_reservoirs[sf]
                rec = AnnotatedSequence(
                    seq_id=cid,
                    sequence=seq,
                    cath_h=sf,
                    cath_t=cath_topology_of(sf),
                )
                if len(res) < args.max_seqs_per_superfamily:
                    res.append(rec)
                else:
                    j = int(rng.integers(0, fam_seen[sf]))
                    if j < args.max_seqs_per_superfamily:
                        res[j] = rec
            elif acc not in all_cath_annotated_accs:
                singleton_seen += 1
                if len(singleton_reservoir) < args.singleton_pool_cap:
                    singleton_reservoir.append(
                        AnnotatedSequence(seq_id=cid, sequence=seq, cath_h="", cath_t="")
                    )
                else:
                    j = int(rng.integers(0, singleton_seen))
                    if j < args.singleton_pool_cap:
                        singleton_reservoir[j] = AnnotatedSequence(
                            seq_id=cid, sequence=seq, cath_h="", cath_t=""
                        )

    cath_records = [rec for recs in fam_reservoirs.values() for rec in recs]
    print(
        f"[stage2-bundle] Scanned {total_scanned:,} valid UniRef50 seqs -> "
        f"{len(cath_records):,} CATH-annotated across {len(fam_reservoirs):,} H-superfamilies "
        f"+ {len(singleton_reservoir):,} singleton reps",
        flush=True,
    )

    arm_targets = tuple(int(x.strip()) for x in args.arm_families.split(",") if x.strip())
    splits = build_controlled_natural_splits(
        cath_records,
        singleton_uniref_pool=singleton_reservoir,
        priority_unseen_topologies=priority_topos,
        excluded_eval_seq_hashes=excluded_hashes,
        target_unique_tokens=int(args.target_unique_tokens),
        arm_family_targets=arm_targets,
        allow_family_cap=bool(args.allow_family_cap),
        include_singleton_arm=True,
        n_unseen_fold_eval=int(args.n_unseen_fold),
        min_unseen_topologies=int(args.min_unseen_topologies),
        n_homolog_eval=int(args.n_homolog_eval),
        n_global_natural_eval=int(args.n_global_natural),
        seed=int(args.seed),
    )

    unseen_topos_set = set(splits["unseen_fold_topologies"])
    split_summaries: dict[str, Any] = {
        "eval_unseen_fold": save_packed_split_npz(
            splits_dir / "eval_unseen_fold.npz", splits["eval_unseen_fold"]
        ),
        "eval_global_natural": save_packed_split_npz(
            splits_dir / "eval_global_natural.npz", splits["eval_global_natural"]
        ),
    }

    arms_manifest: dict[str, Any] = {}
    for arm_id, arm in splits["arms"].items():
        tr_meta = save_packed_split_npz(splits_dir / f"train_{arm_id}.npz", arm["train"])
        sub_meta = save_packed_split_npz(
            splits_dir / f"eval_train_sub_{arm_id}.npz", arm["eval_train_sub"]
        )
        hom_meta = save_packed_split_npz(
            splits_dir / f"eval_unseen_homolog_{arm_id}.npz", arm["eval_unseen_homolog"]
        )
        arms_manifest[arm_id] = {
            "arm_id": arm_id,
            "target_families": int(arm["target_families"]),
            "realized_families": int(arm["realized_families"]),
            "is_singleton_limit": bool(arm["is_singleton_limit"]),
            "realized_tokens": int(arm["realized_tokens"]),
            "token_budget_rel_error": float(arm["token_budget_rel_error"]),
            "mean_depth": float(arm["mean_depth"]),
            "train": tr_meta,
            "eval_train_sub": sub_meta,
            "eval_unseen_homolog": hom_meta,
        }

    benchmarks_manifest: dict[str, Any] = {}
    if not args.skip_benchmarks:
        print("[stage2-bundle] Packaging MegaScale ddG benchmark ...", flush=True)
        ms_payload = package_megascale_benchmark(
            megascale_parquet_dir=data_root / "data/megascale_tsuboyama2023/dataset2/data",
            disjointness_dir=data_root / "results/shared/megascale_disjointness_20260813",
            acc_to_cath_h=acc_to_cath_h,
            unseen_topologies=unseen_topos_set,
        )
        ms_sha = atomic_write_gzip_json(bench_dir / "megascale_ddg.json.gz", ms_payload)
        benchmarks_manifest["megascale"] = {
            "file": "benchmarks/megascale_ddg.json.gz",
            "sha256": ms_sha,
            "n_domains": ms_payload["n_domains"],
            "n_variants_total": ms_payload["n_variants_total"],
            "stratum_counts": ms_payload["stratum_counts"],
        }

        print("[stage2-bundle] Packaging ProteinGym L<=384 benchmark ...", flush=True)
        pg_payload = package_proteingym_benchmark(
            reference_csv=data_root / "data/proteingym_raw/DMS_substitutions.csv",
            substitutions_dir=data_root / "data/proteingym/DMS_ProteinGym_substitutions",
            entry_to_acc=entry_to_acc,
            acc_to_cath_h=acc_to_cath_h,
            unseen_topologies=unseen_topos_set,
            max_seq_len=int(args.max_seq_len),
            seed=int(args.seed),
        )
        pg_sha = atomic_write_gzip_json(bench_dir / "proteingym_substitutions.json.gz", pg_payload)
        benchmarks_manifest["proteingym"] = {
            "file": "benchmarks/proteingym_substitutions.json.gz",
            "sha256": pg_sha,
            "n_assays_full": pg_payload["n_assays_full"],
            "n_assays_milestone": pg_payload["n_assays_milestone"],
            "stratum_counts": pg_payload["stratum_counts"],
        }

        print("[stage2-bundle] Packaging unseen-fold 3D contacts ...", flush=True)
        ct_meta = package_unseen_fold_contacts(
            cath_3d_kv_path=data_root / "results/extensions/memory_theory_20261120/cath_3d_kv_cohort.json",
            alphafold_dir=data_root / "data/alphafold",
            unseen_topologies=unseen_topos_set,
            output_npz=bench_dir / "unseen_fold_contacts.npz",
            output_meta_json=bench_dir / "unseen_fold_contacts_meta.json",
            target_n_domains=int(args.n_contact_domains),
        )
        benchmarks_manifest["contacts"] = {
            "meta_file": "benchmarks/unseen_fold_contacts_meta.json",
            "npz_file": "benchmarks/unseen_fold_contacts.npz",
            "n_domains": ct_meta["n_domains"],
            "n_distinct_topologies": ct_meta["n_distinct_topologies"],
        }

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "knob": "knob2_natural_uniref50_cath",
        "created_utc": _now_utc(),
        "seed": int(args.seed),
        "min_seq_len": int(args.min_seq_len),
        "max_seq_len": int(args.max_seq_len),
        "target_unique_tokens": int(args.target_unique_tokens),
        "n_unseen_fold_topologies": len(splits["unseen_fold_topologies"]),
        "unseen_fold_topologies": splits["unseen_fold_topologies"],
        "fold_disjointness_verified": True,
        "splits": split_summaries,
        "arms": arms_manifest,
        "benchmarks": benchmarks_manifest,
        "bundle_size_mb": round(float(total_bytes) / (1024.0 * 1024.0), 3),
        "build_wall_sec": round(time.perf_counter() - t0, 2),
    }
    atomic_write_json_with_sha256(out_dir / "bundle_manifest.json", manifest)
    print(
        f"[stage2-bundle] Complete! Bundle at {out_dir} ({manifest['bundle_size_mb']} MB, "
        f"{manifest['build_wall_sec']}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
