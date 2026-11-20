#!/usr/bin/env python3
"""ESM3-1.4B Sequence-Only R1/R2/R3 Scoring, p=0 Readouts & Track B4 VQ-VAE Probe.

Strictly sequence-only mode (all structure/SS8/SASA/function tracks masked) for
fair comparison against the frozen 33-arm causal roster and pMLM exploratory panel:
  - r1_ladder       : 201 ProteinGym assays (25,728 variants; amortized single-site
                      masked marginal + additive or optional exact O(L) PLL for multis)
  - r1_conditionals : 1-pass native 20-AA masked conditional on 4,075 contexts
  - r2_stability    : MegaScale 101 WTs / 25,856 single mutants
  - r2_abundance    : Domainome 428 assays / 109,568 single mutants
  - r2_phenotypes   : Added phenotype breadth (312 WTs / 14,487 rows)
  - r3_cycles       : Exact deterministic O(L) PLL on all 12,977 MegaScale states
                      yielding non-degenerate epsilon_PLL = PLL(AB)-PLL(A)-PLL(B)+PLL(WT)
  - readouts        : p=0 unmasked hidden-state summaries from transformer.blocks[23, 47]
  - b4_vqvae        : Track B4 VQ-VAE tokenization geometry probe (seq_only vs 3d_conditioned
                      residual gradient effective rank r_eff/d_model + 4096-codebook entropy)
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
    AA20,
    AA20_SET,
    exact_pll,
    extract_hidden_states,
    load_backbone_from_structure_file,
    load_esm3_offline,
    native_site_conditionals,
    probe_vqvae_tokenization_geometry,
    single_site_marginals,
)

AA20_INDEX = {aa: i for i, aa in enumerate(AA20)}
ALL_TASKS: tuple[str, ...] = (
    "r1_ladder",
    "r1_conditionals",
    "r2_stability",
    "r2_abundance",
    "r2_phenotypes",
    "r3_cycles",
    "readouts",
    "b4_vqvae",
)


def _read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur_id: str | None = None
    buf: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(buf).upper()
                cur_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(buf).upper()
    return seqs


def _write_json_with_sha256(path: Path, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _apply_substitutions(wt: str, subs: Sequence[Sequence[Any]]) -> str:
    chars = list(wt)
    for wt_aa, pos1, mt_aa in subs:
        idx = int(pos1) - 1
        if not (0 <= idx < len(chars)) or chars[idx] != str(wt_aa):
            raise ValueError(f"Substitution {wt_aa}{pos1}{mt_aa} mismatches WT residue {chars[idx] if 0 <= idx < len(chars) else 'OOB'}")
        chars[idx] = str(mt_aa)
    return "".join(chars)


def compute_cycle_epistasis(pll_wt: float, pll_a: float, pll_b: float, pll_ab: float) -> float:
    """Non-degenerate 4-state epistasis cycle scalar ``PLL(AB) - PLL(A) - PLL(B) + PLL(WT)``."""

    return float(pll_ab - pll_a - pll_b + pll_wt)


def run_r1_ladder(
    handle: Any,
    *,
    variant_table_path: Path,
    wildtypes_fasta: Path,
    multi_exact_pll: bool,
    batch_size: int,
    limit: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Score the frozen 201-assay R1 ProteinGym mutation ladder."""

    wt_map = _read_fasta(wildtypes_fasta)
    table = json.loads(variant_table_path.read_text(encoding="utf-8"))
    samples = table["samples"]
    if limit > 0:
        samples = samples[:limit]

    # Group distinct mutated sites per wildtype_id to amortize single-site forwards
    by_wt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_wt[str(s["wildtype_id"])].append(s)

    scored_rows: list[dict[str, Any]] = []
    wt_pll_cache: dict[str, float] = {}

    for wt_id, group in sorted(by_wt.items()):
        wt_seq = wt_map[wt_id]
        sites = sorted(
            {
                int(pos1) - 1
                for s in group
                for (_w, pos1, _m) in s["substitutions"]
            }
        )
        marginals = single_site_marginals(
            handle, wt_seq, sites, batch_size=batch_size, restrict_aa20=False
        )
        for s in group:
            subs = s["substitutions"]
            additive = 0.0
            for wt_aa, pos1, mt_aa in subs:
                pos0 = int(pos1) - 1
                lp = marginals[pos0]
                additive += float(lp[AA20_INDEX[str(mt_aa)]] - lp[AA20_INDEX[str(wt_aa)]])

            score_exact: float | None = None
            if multi_exact_pll and len(subs) > 1:
                if wt_id not in wt_pll_cache:
                    wt_pll_cache[wt_id] = float(
                        exact_pll(handle, wt_seq, batch_size=batch_size)["pll_sum_nats"]
                    )
                mut_seq = _apply_substitutions(wt_seq, subs)
                mut_pll = float(exact_pll(handle, mut_seq, batch_size=batch_size)["pll_sum_nats"])
                score_exact = float(mut_pll - wt_pll_cache[wt_id])

            scored_rows.append(
                {
                    "sample_id": s["sample_id"],
                    "assay": s["assay"],
                    "wildtype_id": wt_id,
                    "mutation": s["mutation"],
                    "n_substitutions": int(s["n_substitutions"]),
                    "score_masked_marginal_additive": additive,
                    "score_exact_pll_delta": score_exact,
                }
            )

    out_path = out_dir / "r1_ladder_scores.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in scored_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    return {
        "n_samples": len(scored_rows),
        "n_wildtypes": len(by_wt),
        "multi_exact_pll": multi_exact_pll,
        "mean_score": float(np.mean([r["score_masked_marginal_additive"] for r in scored_rows])) if scored_rows else float("nan"),
        "scores_jsonl": out_path.name,
    }


def run_r1_conditionals(
    handle: Any,
    *,
    cond_inputs_path: Path,
    limit: int,
    out_dir: Path,
) -> dict[str, Any]:
    """1-pass native 20-AA masked conditional on frozen ``family_conditionals_20261107``."""

    data = json.loads(cond_inputs_path.read_text(encoding="utf-8"))
    units = data["units"]
    if limit > 0:
        units = units[:limit]

    rows: list[dict[str, Any]] = []
    unit_means: list[float] = []
    for unit in units:
        site_means: list[float] = []
        for site in unit.get("sites", []):
            ctx_vals: list[float] = []
            for ctx in site.get("contexts", []):
                seq = str(ctx["native"])
                idx0 = int(ctx["native_index"])
                obs = str(ctx["observed"])
                res = native_site_conditionals(handle, seq, idx0, obs)
                ctx_vals.append(res["observed_logp_nats"])
                rows.append(
                    {
                        "unit_id": unit["unit_id"],
                        "partition": unit.get("partition"),
                        "query_position": int(site["query_position"]),
                        "row_id": ctx["row_id"],
                        "native_index": idx0,
                        "observed": obs,
                        **res,
                    }
                )
            if ctx_vals:
                site_means.append(float(np.mean(ctx_vals)))
        if site_means:
            unit_means.append(float(np.mean(site_means)))

    out_path = out_dir / "r1_family_conditionals.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    return {
        "n_units": len(units),
        "n_contexts": len(rows),
        "mean_observed_logp_nats": float(np.mean(unit_means)) if unit_means else float("nan"),
        "scores_jsonl": out_path.name,
    }


def _score_single_site_cohort(
    handle: Any,
    items: Sequence[tuple[str, str, Sequence[tuple[str, int, str, str]]]],
    *,
    batch_size: int,
    out_path: Path,
) -> dict[str, Any]:
    """Generic amortized single-site scorer for R2 stability, abundance, and phenotypes.

    ``items`` yields ``(group_name, wildtype_seq, [(variant_id, pos0, wt_aa, mt_aa), ...])``.
    """

    total_vars = 0
    scores_all: list[float] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for group_name, wt_seq, variants in items:
            sites = sorted({pos0 for (_vid, pos0, _w, _m) in variants})
            if not sites:
                continue
            marginals = single_site_marginals(
                handle, wt_seq, sites, batch_size=batch_size, restrict_aa20=False
            )
            for vid, pos0, wt_aa, mt_aa in variants:
                lp = marginals[pos0]
                delta = float(lp[AA20_INDEX[mt_aa]] - lp[AA20_INDEX[wt_aa]])
                scores_all.append(delta)
                total_vars += 1
                fh.write(
                    json.dumps(
                        {
                            "group": group_name,
                            "variant_id": vid,
                            "position_0based": pos0,
                            "wt_aa": wt_aa,
                            "mt_aa": mt_aa,
                            "score_masked_marginal": delta,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return {
        "n_groups": len(items),
        "n_variants": total_vars,
        "mean_score": float(np.mean(scores_all)) if scores_all else float("nan"),
        "scores_jsonl": out_path.name,
    }


def run_r2_stability(
    handle: Any, *, cohort_path: Path, batch_size: int, limit: int, out_dir: Path
) -> dict[str, Any]:
    raw = json.loads(cohort_path.read_text(encoding="utf-8"))
    bgs = raw["backgrounds"][:limit] if limit > 0 else raw["backgrounds"]
    items = []
    for bg in bgs:
        wt = str(bg["wildtype"])
        variants = []
        for idx, v in enumerate(bg.get("variants", [])):
            # Detect single substitution from mutant sequence directly (immune to 0/1-based ambiguity)
            mseq = str(v["sequence"]) if isinstance(v, dict) and "sequence" in v else None
            if mseq and len(mseq) == len(wt):
                diffs = [i for i, (a, b) in enumerate(zip(wt, mseq)) if a != b]
                if len(diffs) == 1 and mseq[diffs[0]] in AA20_SET:
                    p0 = diffs[0]
                    variants.append((f"{bg['name']}:{idx}", p0, wt[p0], mseq[p0]))
        items.append((str(bg["name"]), wt, variants))
    return _score_single_site_cohort(
        handle, items, batch_size=batch_size, out_path=out_dir / "r2_stability_scores.jsonl"
    )


def run_r2_abundance(
    handle: Any, *, cohort_path: Path, batch_size: int, limit: int, out_dir: Path
) -> dict[str, Any]:
    raw = json.loads(cohort_path.read_text(encoding="utf-8"))
    doms = raw["domains"][:limit] if limit > 0 else raw["domains"]
    items = []
    for dom in doms:
        wt = str(dom["wildtype"])
        variants = []
        for idx, v in enumerate(dom.get("variants", [])):
            mseq = str(v["sequence"])
            if len(mseq) == len(wt):
                diffs = [i for i, (a, b) in enumerate(zip(wt, mseq)) if a != b]
                if len(diffs) == 1 and mseq[diffs[0]] in AA20_SET:
                    p0 = diffs[0]
                    variants.append((f"{dom['name']}:{idx}", p0, wt[p0], mseq[p0]))
        items.append((str(dom["name"]), wt, variants))
    return _score_single_site_cohort(
        handle, items, batch_size=batch_size, out_path=out_dir / "r2_abundance_scores.jsonl"
    )


def run_r2_phenotypes(
    handle: Any, *, rows_dir: Path, query_fasta: Path, batch_size: int, limit: int, out_dir: Path
) -> dict[str, Any]:
    wt_by_sha = {
        hashlib.sha256(seq.encode("utf-8")).hexdigest(): seq
        for seq in _read_fasta(query_fasta).values()
    }
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for jf in sorted(rows_dir.glob("*.jsonl")):
        with jf.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_unit[str(row["unit"])].append(row)

    items = []
    unit_keys = sorted(by_unit.keys())
    if limit > 0:
        unit_keys = unit_keys[:limit]
    for u_sha in unit_keys:
        wt = wt_by_sha.get(u_sha)
        if not wt or set(wt) - AA20_SET:
            continue
        variants = []
        for row in by_unit[u_sha]:
            p0 = int(row["position"]) - 1
            wt_aa, mt_aa = str(row["wt_aa"]), str(row["mt_aa"])
            if 0 <= p0 < len(wt) and wt[p0] == wt_aa and mt_aa in AA20_SET:
                variants.append((str(row["row_id"]), p0, wt_aa, mt_aa))
        items.append((u_sha, wt, variants))
    return _score_single_site_cohort(
        handle, items, batch_size=batch_size, out_path=out_dir / "r2_phenotypes_scores.jsonl"
    )


def run_r3_cycles(
    handle: Any,
    *,
    plan_path: Path,
    batch_size: int,
    shard_index: int,
    num_shards: int,
    limit: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Exact deterministic O(L) PLL on MegaScale double-mutant states & epistasis cycles."""

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    bgs = [b for i, b in enumerate(plan["backgrounds"]) if i % num_shards == shard_index]
    if limit > 0:
        bgs = bgs[:limit]

    state_rows_path = out_dir / "r3_state_plls.jsonl"
    cycle_rows_path = out_dir / "r3_cycle_epistasis.jsonl"
    n_states = 0
    n_cycles = 0
    abs_eps: list[float] = []

    with state_rows_path.open("w", encoding="utf-8") as sfh, cycle_rows_path.open(
        "w", encoding="utf-8"
    ) as cfh:
        for bg in bgs:
            seqs: list[str] = list(bg["sequences"])
            seq_to_pll: dict[str, float] = {}
            for seq in seqs:
                if seq in seq_to_pll:
                    continue
                res = exact_pll(handle, seq, batch_size=batch_size)
                seq_to_pll[seq] = float(res["pll_sum_nats"])
                n_states += 1
                sfh.write(
                    json.dumps(
                        {
                            "background": bg["name"],
                            "group": bg["group"],
                            "sequence": seq,
                            "length": len(seq),
                            "pll_sum_nats": res["pll_sum_nats"],
                            "mean_nll_per_residue_nats": res["mean_nll_per_residue_nats"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            for cyc in bg.get("cycles", []):
                # Canonical extraction_plan.json layout: {"positions": [i, j], "separation": ..., "states": [i_wt, i_a, i_b, i_ab]}
                s_wt = s_a = s_b = s_ab = None
                if "states" in cyc and isinstance(cyc["states"], list) and len(cyc["states"]) == 4:
                    i_wt, i_a, i_b, i_ab = (int(x) for x in cyc["states"])
                    s_wt, s_a, s_b, s_ab = seqs[i_wt], seqs[i_a], seqs[i_b], seqs[i_ab]
                else:
                    def _get_seq(key_seq: str, key_idx: str) -> str | None:
                        if key_seq in cyc and isinstance(cyc[key_seq], str):
                            return cyc[key_seq]
                        if key_idx in cyc and isinstance(cyc[key_idx], int):
                            return seqs[cyc[key_idx]]
                        return None

                    s_wt = _get_seq("wt", "wt_index") or str(bg["wildtype"])
                    s_a = _get_seq("a", "a_index") or _get_seq("single_a", "single_a_index")
                    s_b = _get_seq("b", "b_index") or _get_seq("single_b", "single_b_index")
                    s_ab = _get_seq("ab", "ab_index") or _get_seq("double", "double_index")

                if all(s is not None and s in seq_to_pll for s in (s_wt, s_a, s_b, s_ab)):
                    eps = compute_cycle_epistasis(
                        seq_to_pll[s_wt], seq_to_pll[s_a], seq_to_pll[s_b], seq_to_pll[s_ab]  # type: ignore[index]
                    )
                    abs_eps.append(abs(eps))
                    n_cycles += 1
                    cfh.write(
                        json.dumps(
                            {
                                "background": bg["name"],
                                "group": bg["group"],
                                "positions": cyc.get("positions"),
                                "separation": cyc.get("separation"),
                                "epsilon_pll_nats": eps,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            sfh.flush()
            cfh.flush()

    return {
        "n_backgrounds": len(bgs),
        "n_states_scored": n_states,
        "n_cycles_computed": n_cycles,
        "mean_abs_epsilon_pll_nats": float(np.mean(abs_eps)) if abs_eps else None,
        "states_jsonl": state_rows_path.name,
        "cycles_jsonl": cycle_rows_path.name,
    }


def run_readouts_p0(
    handle: Any,
    *,
    stability_cohort: Path,
    batch_size: int,
    limit: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Extract p=0 unmasked hidden-state readouts from transformer.blocks[23, 47]."""

    raw = json.loads(stability_cohort.read_text(encoding="utf-8"))
    bgs = raw["backgrounds"][:limit] if limit > 0 else raw["backgrounds"]
    seqs: list[str] = []
    labels: list[str] = []
    for bg in bgs:
        wt = str(bg["wildtype"])
        if not set(wt) - AA20_SET:
            seqs.append(wt)
            labels.append(str(bg["name"]))
    feats = extract_hidden_states(handle, seqs, batch_size=min(batch_size, 16))
    npz_path = out_dir / "readouts_p0.npz"
    np.savez_compressed(npz_path, labels=np.asarray(labels), **feats)
    return {
        "n_sequences": len(seqs),
        "block_indices": [(handle.n_layers - 1) // 2, handle.n_layers - 1],
        "feature_shapes": {k: list(v.shape) for k, v in feats.items()},
        "npz": npz_path.name,
    }


def run_b4_vqvae_probe(
    handle: Any,
    *,
    alphafold_dir: Path,
    n_proteins: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Track B4 VQ-VAE tokenization geometry probe on ground-truth AFDB structures."""

    seqs: list[str] = []
    coords: list[np.ndarray] = []
    ids: list[str] = []
    for cand in sorted(alphafold_dir.glob("AF-*-F1-model_v6.pdb.gz")):
        if len(seqs) >= n_proteins:
            break
        try:
            rec = load_backbone_from_structure_file(cand)
        except Exception:
            continue
        seq = rec["sequence"]
        if 50 <= len(seq) <= 256 and not (set(seq) - AA20_SET):
            seqs.append(seq)
            coords.append(rec["coords_atom37"])
            ids.append(cand.name)
    if not seqs:
        return {"n_proteins": 0, "skipped": "no_local_afdb_structures_found"}
    probe = probe_vqvae_tokenization_geometry(handle, seqs, coords)
    probe["structure_files"] = ids
    _write_json_with_sha256(out_dir / "b4_vqvae_probe.json", probe)
    return probe


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--tasks", default=",".join(ALL_TASKS))
    p.add_argument("--multi-exact-pll", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--b4-n-proteins", type=int, default=32)
    p.add_argument("--alphafold-dir", type=Path, default=DATA_ROOT / "data/alphafold")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in ALL_TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}; valid tasks: {ALL_TASKS}")

    handle = load_esm3_offline(args.model, device=args.device, dtype=args.dtype)
    results: dict[str, Any] = {}

    if "r1_ladder" in tasks:
        results["r1_ladder"] = run_r1_ladder(
            handle,
            variant_table_path=DATA_ROOT / "results/extensions/statistical_ladder_20261106/inputs/variant_table.json",
            wildtypes_fasta=DATA_ROOT / "results/R5/retrieval_bound_20260807/wildtypes.faa",
            multi_exact_pll=args.multi_exact_pll,
            batch_size=args.batch_size,
            limit=args.limit,
            out_dir=args.out,
        )
    if "r1_conditionals" in tasks:
        results["r1_conditionals"] = run_r1_conditionals(
            handle,
            cond_inputs_path=DATA_ROOT / "results/extensions/family_conditionals_20261107/conditional-inputs.json",
            limit=args.limit,
            out_dir=args.out,
        )
    if "r2_stability" in tasks:
        results["r2_stability"] = run_r2_stability(
            handle,
            cohort_path=DATA_ROOT / "results/R4/gate_stability_20260924/cohort/cohort.json",
            batch_size=args.batch_size,
            limit=args.limit,
            out_dir=args.out,
        )
    if "r2_abundance" in tasks:
        results["r2_abundance"] = run_r2_abundance(
            handle,
            cohort_path=DATA_ROOT / "results/R1/external_confirmation_20260924/cohort.json",
            batch_size=args.batch_size,
            limit=args.limit,
            out_dir=args.out,
        )
    if "r2_phenotypes" in tasks:
        results["r2_phenotypes"] = run_r2_phenotypes(
            handle,
            rows_dir=DATA_ROOT / "results/extensions/phenotype_breadth_20261008/qual-3/rows",
            query_fasta=DATA_ROOT / "results/extensions/phenotype_breadth_20261008/profiles/search/query.faa",
            batch_size=args.batch_size,
            limit=args.limit,
            out_dir=args.out,
        )
    if "r3_cycles" in tasks:
        results["r3_cycles"] = run_r3_cycles(
            handle,
            plan_path=DATA_ROOT / "archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json",
            batch_size=args.batch_size,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            limit=args.limit,
            out_dir=args.out,
        )
    if "readouts" in tasks:
        results["readouts"] = run_readouts_p0(
            handle,
            stability_cohort=DATA_ROOT / "results/R4/gate_stability_20260924/cohort/cohort.json",
            batch_size=args.batch_size,
            limit=args.limit,
            out_dir=args.out,
        )
    if "b4_vqvae" in tasks:
        n_b4 = min(args.b4_n_proteins, args.limit) if args.limit > 0 else args.b4_n_proteins
        results["b4_vqvae"] = run_b4_vqvae_probe(
            handle,
            alphafold_dir=args.alphafold_dir,
            n_proteins=n_b4,
            out_dir=args.out,
        )

    payload = {
        "schema": "esm3_r123_seqonly_v1",
        "model_facts": handle.facts,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_seconds": round(time.time() - t0, 2),
        "results": results,
    }
    _write_json_with_sha256(args.out / "esm3_r123_seqonly.json", payload)


if __name__ == "__main__":
    main()
