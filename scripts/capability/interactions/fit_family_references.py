#!/usr/bin/env python3
"""Fit and qualify one plmc Potts reference per domain, with phylogeny-aware regularization.

Differences from the archived baseline fit (fit_pairwise_sequence_baseline.py), which
this reuses for the objective it declared:
  * the held-out split holds out whole identity clusters, then purges every training row
    that is >= CLUSTER_IDENTITY identical to a held-out row, so no train/validation pair
    exceeds the threshold (the archived split hashed aligned rows);
  * the lambda_E grid is extended on both sides and an edge selection is a failure;
  * every fit records plmc's terminal state, and a fit that did not reach plmc's own
    gradient criterion (||g|| / max(1, ||x||) <= 1e-3) is reported as not converged.

No phenotype, mutation panel or model output is read. The split is used for the choice
of lambda_E only.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from scripts.capability.interactions.fit_pairwise_sequence_baseline import (
    independent_site, plmc_conditional, profile_conditional, scored_support)
from src.capability.context.family_potts import (
    FamilyModel, dense_couplings, load_family_model, log_profile, potts_score, save_family_model, sha256_file)
from src.capability.context.profiles import GAP_CODE, REWEIGHT_IDENTITY_FLOOR, sequence_weights
from src.capability.core.amino_acids import AA20
from src.capability.stability.pairwise_stability import read_plmc

LAMBDA_E_GRID = (0.0064, 0.032, 0.16, 0.8, 4.0, 16.0, 80.0, 400.0, 2000.0)
LAMBDA_H = 0.01
#: Identity at which rows are clustered, held out whole, and purged from training. A 50% purge
#: leaves 8-206 training rows in ten of twelve domains (see split_feasibility in the report),
#: so the strictest threshold that keeps every training fold usable is declared instead.
CLUSTER_IDENTITY = 0.7
FEASIBILITY_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
#: Two rows are linkable only if they share at least this fraction of the domain's columns.
MIN_OVERLAP_FRACTION = 0.5
HELD_OUT_FRACTION = 0.2
#: A cluster holding more than this fraction of all rows is never held out whole.
MAX_HELD_OUT_CLUSTER_FRACTION = 0.25
SPLIT_SEED = 20261013
NEFF_IDENTITY = 30.0
N_SHUFFLES = 200
CONVERGED_STATUS = "Minimization success"
ASSETS = Path("data/pairwise_assets")
MSA_DIR = ASSETS / "same12_provenance_20260924" / "msa"
QUERIES = ASSETS / "same12_provenance_20260924" / "query_12.faa"
PROFILE_NPZ = ASSETS / "same12_provenance_20260924" / "rebuild" / "profile_features.npz"
PROFILE_NPZ_CACHED = Path("results/R4/gate_stability_20260924/cohort/profile_features.npz")
PLMC = ASSETS / "plmc" / "debbiemarkslab-plmc-18c9e55" / "bin" / "plmc"
PRIOR = ASSETS / "same12_provenance_20260924" / "pairwise_provenance.json"
CODE = np.full(256, GAP_CODE, dtype=np.uint8)
for _i, _r in enumerate(AA20):
    CODE[ord(_r)] = _i


def read_fasta(path: Path) -> list[tuple[str, str]]:
    out, name, parts = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(parts)))
            name, parts = line[1:].strip(), []
        elif line.strip():
            parts.append(line.strip())
    if name is not None:
        out.append((name, "".join(parts)))
    return out


def encode_rows(sequences: list[str]) -> np.ndarray:
    return CODE[np.frombuffer("".join(sequences).encode(), dtype=np.uint8)].reshape(len(sequences), -1)


def load_domain(domain: str, wildtype: str):
    """Rows as codes, first row the focus; fails unless the MSA focus is the WT and every row is full-width."""
    records = read_fasta(MSA_DIR / f"{domain}.pdb.fasta")
    if records[0][0] != "focus" or records[0][1] != wildtype:
        raise ValueError(f"{domain}: MSA focus row is not the wild-type query")
    if {len(s) for _, s in records} != {len(wildtype)}:
        raise ValueError(f"{domain}: MSA rows are not all {len(wildtype)} columns wide")
    if set("".join(s for _, s in records)) - set(AA20 + "-"):
        raise ValueError(f"{domain}: MSA carries symbols outside the 20 residues and gap")
    return [s for _, s in records], encode_rows([s for _, s in records])


def adjacency(rows: np.ndarray, threshold: float) -> np.ndarray:
    """Bool (N, N): identity over co-present columns >= threshold with enough overlap."""
    n, length = rows.shape
    onehot = np.zeros((n, length * 20), dtype=np.float32)
    r, c = np.nonzero(rows != GAP_CODE)
    onehot[r, c * 20 + rows[r, c]] = 1.0
    present = (rows != GAP_CODE).astype(np.float32)
    out = np.zeros((n, n), dtype=bool)
    for start in range(0, n, 1024):
        stop = min(start + 1024, n)
        same = onehot[start:stop] @ onehot.T
        both = present[start:stop] @ present.T
        out[start:stop] = (both >= MIN_OVERLAP_FRACTION * length) & (same >= threshold * both)
    out[np.arange(n), np.arange(n)] = True
    return out


def leader_clusters(adj: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Greedy leader clustering: rows by (residues present desc, index asc); each unassigned row
    leads a cluster of all unassigned rows linked to it. Deterministic; ties go to the lower index."""
    order = np.lexsort((np.arange(len(rows)), -(rows != GAP_CODE).sum(axis=1)))
    cluster = np.full(len(rows), -1, dtype=np.int64)
    for leader in order:
        if cluster[leader] < 0:
            cluster[adj[leader] & (cluster < 0)] = leader
    return cluster


def split_rows(domain: str, rows: np.ndarray, threshold: float) -> dict:
    """Row 0 is the focus. Returns held-out and train row indices (focus is train unless purged)."""
    adj = adjacency(rows, threshold)
    cluster = leader_clusters(adj, rows)
    leaders, sizes = np.unique(cluster, return_counts=True)
    n_hits = len(rows) - 1
    eligible = [(int(lead), int(s)) for lead, s in zip(leaders, sizes)
                if lead != cluster[0] and s <= MAX_HELD_OUT_CLUSTER_FRACTION * n_hits]
    eligible.sort(key=lambda e: hashlib.sha256(f"{SPLIT_SEED}:{domain}:{e[0]}".encode()).digest())
    held_leaders, held = [], 0
    for leader, size in eligible:
        if held >= HELD_OUT_FRACTION * n_hits:
            break
        held_leaders.append(leader)
        held += size
    is_held = np.isin(cluster, held_leaders)
    purged = adj[:, is_held].any(axis=1) & ~is_held
    return {"cluster": cluster, "held": is_held, "purged": purged, "n_clusters": int(len(leaders)),
            "n_held_clusters": len(held_leaders), "largest_cluster": int(sizes.max())}


def write_alignment(path: Path, sequences: list[str], index: np.ndarray, weights: np.ndarray, wpath: Path):
    path.write_text("".join(f">{'focus' if i == 0 else f'r{i}'}\n{sequences[i]}\n" for i in index))
    wpath.write_text("".join(f"{w:.8f}\n" for w in weights))


def run_plmc(alignment: Path, weights: Path, out: Path, log: Path, lambda_e: float, max_iter: int, threads: int) -> dict:
    done = subprocess.run([str(PLMC), "-o", str(out), "-f", "focus", "-le", str(lambda_e), "-lh", str(LAMBDA_H),
                           "-m", str(max_iter), "-g", "-n", str(threads), "-w", str(weights), str(alignment)],
                          capture_output=True, text=True)
    log.write_text(done.stdout + done.stderr)
    if done.returncode != 0 or not out.exists():
        raise RuntimeError(f"plmc failed ({done.returncode}): {done.stderr.strip().splitlines()[-3:]}")
    text = done.stdout + done.stderr
    status = re.findall(r"Gradient optimization: (.*)", text)[-1]
    lines = [x.split("\t") for x in text.splitlines() if re.fullmatch(r"\d+\t.*\t.*", x) and x.count("\t") == 6]
    last = lines[-1]
    return {"status": status, "converged": status == CONVERGED_STATUS, "cap_hit": "MAXIMUMITERATION" in status,
            "iterations": int(last[0]), "max_iter": max_iter, "terminal_gradient_criterion_printed": float(last[2]),
            "gradient_criterion_threshold": 1e-3, "final_objective": float(last[3]),
            "seconds": float(last[1])}


def sweep_job(job: dict) -> dict:
    """One held-out fit at one lambda_E; result cached by configuration."""
    work = Path(job["work"])
    cache = work / f"sweep_le{job['lambda_e']}.json"
    if cache.exists() and json.loads(cache.read_text())["config"] == job["config"]:
        return json.loads(cache.read_text())
    split = np.load(work / "split.npz")
    sequences, rows = job["sequences"], job["rows"]
    keep = np.flatnonzero(~split["held"] & ~split["purged"])
    weights = sequence_weights(rows[keep], identity=REWEIGHT_IDENTITY_FLOOR)
    if keep[0] != 0:  # plmc needs the focus first; a purged focus stays as its column anchor at weight zero
        keep, weights = np.concatenate([[0], keep]), np.concatenate([[0.0], weights])
    tag = f"train_le{job['lambda_e']}"
    write_alignment(work / f"{tag}.fa", sequences, keep, weights, work / f"{tag}.w")
    fit = run_plmc(work / f"{tag}.fa", work / f"{tag}.w", work / f"{tag}.params", work / f"{tag}.log",
                   job["lambda_e"], job["max_iter"], job["threads"])
    params = read_plmc(work / f"{tag}.params")
    test = rows[np.flatnonzero(split["held"])]
    fit["held_out_conditional"] = plmc_conditional(params.fields, params.couplings, test, scored_support(test))
    fit["lambda_e"], fit["config"] = job["lambda_e"], job["config"]
    for suffix in (".params", ".fa", ".w"):
        (work / f"{tag}{suffix}").unlink()
    cache.write_text(json.dumps(fit, indent=1))
    return fit


def sign_check(model: FamilyModel, rng: np.random.Generator) -> dict:
    """The wild type must outscore composition-matched shuffles under the exported convention."""
    wt = potts_score(model, model.wildtype)
    shuffles = np.array([potts_score(model, "".join(rng.permutation(list(model.wildtype)))) for _ in range(N_SHUFFLES)])
    out = {"wildtype_score": wt, "shuffle_mean": float(shuffles.mean()), "shuffle_max": float(shuffles.max()),
           "fraction_shuffles_below_wildtype": float((shuffles < wt).mean())}
    if out["fraction_shuffles_below_wildtype"] < 0.99:
        raise ValueError(f"{model.domain}: wild type does not outscore shuffled sequences; sign convention suspect")
    return out


def final_job(job: dict) -> dict:
    """Refit on all rows at the selected lambda_E, qualify, verify the convention and export."""
    work, domain, wildtype = Path(job["work"]), job["domain"], job["wildtype"]
    sequences, rows, length = job["sequences"], job["rows"], len(job["wildtype"])
    weights = sequence_weights(rows, identity=REWEIGHT_IDENTITY_FLOOR)
    write_alignment(work / "full.fa", sequences, np.arange(len(rows)), weights, work / "full.w")
    fit = run_plmc(work / "full.fa", work / "full.w", work / "full.params", work / "full.log",
                   job["lambda_e"], job["max_iter"], job["threads"])
    params = read_plmc(work / "full.params")
    columns = params.fields.shape[0]
    verdict = qualify(job["stats"], job["sweep"], fit, length, columns)
    if params.target != wildtype or not np.array_equal(params.positions, np.arange(1, columns + 1)):
        raise ValueError(f"{domain}: plmc focus columns are not the wild-type residues one-to-one")
    prof = np.load(job["profile_npz"])
    freq, cw = prof[f"{domain}.pdb|frequencies"], prof[f"{domain}.pdb|column_weight"]
    model = FamilyModel(domain, wildtype, AA20, params.fields.astype(np.float64), dense_couplings(params.couplings.astype(np.float64), length),
                        freq, cw, log_profile(freq, cw), np.arange(length), verdict["qualified"], tuple(verdict["failures"]), {})
    rng = np.random.default_rng(SPLIT_SEED)
    for _ in range(3):  # the dense tensors reproduce the pinned reader's score
        probe = "".join(rng.choice(list(AA20), size=length))
        if not np.isclose(potts_score(model, probe), params.score(probe), rtol=0, atol=1e-3):
            raise ValueError(f"{domain}: dense tensor score differs from the plmc reader")
    sign = sign_check(model, rng)
    directory = Path(job["export"])
    save_family_model(directory, domain, wildtype=wildtype, h=params.fields, pairs=params.couplings, profile_freq=freq,
                      profile_column_weight=cw, qualified=verdict["qualified"], qualification_failures=verdict["failures"],
                      extra={"lambda_e": job["lambda_e"], "lambda_h": LAMBDA_H, "fit": fit, "sign_check": sign,
                             "plmc_params_sha256": sha256_file(work / "full.params"),
                             "msa_sha256": sha256_file(MSA_DIR / f"{domain}.pdb.fasta"), "qualification": verdict})
    loaded = load_family_model(domain, directory)
    if not (np.array_equal(loaded.h, model.h) and np.array_equal(loaded.J, model.J)):
        raise ValueError(f"{domain}: exported parameters do not round-trip")
    for suffix in (".fa", ".w"):
        (work / f"full{suffix}").unlink()
    return {"final_fit": fit, "sign_check": sign, "qualification": verdict}


def prepare(domain: str, wildtype: str, work: Path) -> dict:
    sequences, rows = load_domain(domain, wildtype)
    work.mkdir(parents=True, exist_ok=True)
    hits = rows[1:]
    weights80 = sequence_weights(hits, identity=REWEIGHT_IDENTITY_FLOOR)
    # Neff at 30%: 80%-reweighted effective sequences among hits >= 30% identical to the query.
    neff30 = float(weights80[(hits == rows[0]).mean(axis=1) >= NEFF_IDENTITY / 100.0].sum())
    feasibility = {}
    for threshold in FEASIBILITY_THRESHOLDS:
        split = split_rows(domain, rows, threshold)
        keep = np.flatnonzero(~split["held"] & ~split["purged"])
        feasibility[str(threshold)] = {
            "clusters": split["n_clusters"], "held_out_rows": int(split["held"].sum()), "purged_rows": int(split["purged"].sum()),
            "train_rows": int(len(keep)), "train_neff_at_80": float(sequence_weights(rows[keep], identity=REWEIGHT_IDENTITY_FLOOR).sum())}
        if threshold == CLUSTER_IDENTITY:
            chosen = split
    held, purged = chosen["held"], chosen["purged"]
    np.savez_compressed(work / "split.npz", cluster=chosen["cluster"], held=held, purged=purged)
    train = np.flatnonzero(~held & ~purged)
    test = rows[np.flatnonzero(held)]
    mask = scored_support(test)
    weights = sequence_weights(rows[train], identity=REWEIGHT_IDENTITY_FLOOR)
    site = independent_site(rows[train], weights)
    stats = {"rows": int(len(hits)), "neff_at_30_percent_identity": neff30,
             "clusters": chosen["n_clusters"], "held_out_clusters": chosen["n_held_clusters"],
             "largest_cluster": chosen["largest_cluster"], "held_out_rows": int(held.sum()),
             "purged_train_rows": int(purged.sum()), "train_rows": int(len(train)),
             "train_neff_at_80": float(weights.sum()), "held_out_positions_scored": int(mask.sum()),
             "independent_site_held_out_conditional": profile_conditional(site, test, mask),
             "split_feasibility": feasibility}
    return {"sequences": sequences, "rows": rows, "stats": stats}


def qualify(stats: dict, sweep: list[dict], final: dict, length: int, columns: int) -> dict:
    scores = [e["held_out_conditional"] for e in sweep]
    best = int(np.argmax(scores))
    gain = scores[best] - stats["independent_site_held_out_conditional"]
    interior = 0 < best < len(sweep) - 1
    failures = []
    if not final["converged"]:
        failures.append(f"final fit not converged: {final['status']} after {final['iterations']} of {final['max_iter']} iterations")
    if not interior:
        failures.append(f"selected lambda_E={sweep[best]['lambda_e']} is at the grid edge")
    if not gain > 0:
        failures.append(f"held-out gain over independent-site is {gain:.6f}")
    if columns != length:
        failures.append(f"column count {columns} != WT length {length}")
    return {"selected_lambda_e": sweep[best]["lambda_e"], "interior": interior, "held_out_gain_over_independent_site": gain,
            "selected_held_out_conditional": scores[best], "selected_cv_fit_converged": sweep[best]["converged"],
            "sweep_fits_converged": sum(e["converged"] for e in sweep), "qualified": not failures, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--final-threads", type=int, default=16)
    parser.add_argument("--domains", nargs="*")
    args = parser.parse_args()
    if sha256_file(PROFILE_NPZ) != sha256_file(PROFILE_NPZ_CACHED):
        raise ValueError("rebuilt and cached frozen profiles differ")
    queries = {n.removesuffix(".pdb"): s for n, s in read_fasta(QUERIES)}
    domains = args.domains or sorted(queries)
    code_sha = sha256_file(Path(__file__))
    prior = {b["name"].removesuffix(".pdb"): b for b in json.loads(PRIOR.read_text())["prior_fit"]["backgrounds"]}
    config = {"max_iter": args.max_iter, "grid": LAMBDA_E_GRID, "seed": SPLIT_SEED, "identity": CLUSTER_IDENTITY, "code": code_sha}
    state = {}
    for domain in domains:
        state[domain] = prepare(domain, queries[domain], args.out_dir / "fits" / domain)
        print(domain, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in state[domain]["stats"].items()}, flush=True)
    sweeps = {d: {} for d in domains}
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for domain in domains:
            for le in LAMBDA_E_GRID:
                job = {"work": str(args.out_dir / "fits" / domain), "lambda_e": le, "max_iter": args.max_iter, "threads": args.threads,
                       "sequences": state[domain]["sequences"], "rows": state[domain]["rows"], "config": config}
                pending[pool.submit(sweep_job, job)] = (domain, le)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                domain, le = pending.pop(future)
                result = future.result()
                if le is None:
                    results[domain] = result
                    print(domain, "final", result["qualification"]["qualified"], flush=True)
                    continue
                sweeps[domain][le] = result
                print(domain, "sweep", le, round(result["held_out_conditional"], 5), result["status"], result["iterations"], flush=True)
                if len(sweeps[domain]) == len(LAMBDA_E_GRID):
                    sweep = [sweeps[domain][g] for g in LAMBDA_E_GRID]
                    best = LAMBDA_E_GRID[int(np.argmax([e["held_out_conditional"] for e in sweep]))]
                    job = {"work": str(args.out_dir / "fits" / domain), "domain": domain, "wildtype": queries[domain],
                           "lambda_e": best, "max_iter": args.max_iter, "threads": args.final_threads,
                           "sequences": state[domain]["sequences"], "rows": state[domain]["rows"], "profile_npz": str(PROFILE_NPZ),
                           "export": str(args.out_dir), "stats": state[domain]["stats"], "sweep": sweep}
                    pending[pool.submit(final_job, job)] = (domain, None)
    table = {}
    for domain in domains:
        table[domain + ".pdb"] = {
            "wildtype_length": len(queries[domain]), **state[domain]["stats"], "lambda_e_grid": list(LAMBDA_E_GRID),
            "lambda_e_held_out_scores": {str(g): sweeps[domain][g]["held_out_conditional"] for g in LAMBDA_E_GRID},
            "sweep_fits": [{k: sweeps[domain][g][k] for k in ("lambda_e", "status", "converged", "iterations", "cap_hit")} for g in LAMBDA_E_GRID],
            "prior_archived": {k: prior[domain].get(k) for k in ("selected_lambda_e", "selected_at_grid_edge", "retained_rows",
                                                                  "neff_at_30_percent_identity", "retrieval_ceiling_reached")},
            **results[domain]}
    report = {"schema": "family_reference_qualification_v1",
              "split": {"cluster_identity": CLUSTER_IDENTITY, "min_overlap_fraction": MIN_OVERLAP_FRACTION,
                        "clustering": "greedy leader: rows ordered by residues present desc then index asc; each unassigned row leads all unassigned linked rows",
                        "identity": "matches / co-present columns; rows linked if co-present >= 0.5 L and identity >= cluster_identity",
                        "held_out_fraction": HELD_OUT_FRACTION, "max_held_out_cluster_fraction": MAX_HELD_OUT_CLUSTER_FRACTION, "seed": SPLIT_SEED,
                        "held_out_choice": "clusters in sha256(seed:domain:leader) order until >= 20% of hit rows; the focus cluster is never held out",
                        "purge": "training rows linked to any held-out row are dropped; a purged focus stays at weight zero as plmc's column anchor",
                        "criterion": "mean gap-ignoring per-site conditional log-likelihood on held-out rows; baseline is the train-fold independent-site profile, pseudocount 0.5",
                        "reads_phenotype": False},
              "plmc": {"lambda_h": LAMBDA_H, "max_iter": args.max_iter, "gap_ignoring": True, "weights": "80% identity reweighting",
                       "binary_sha256": sha256_file(PLMC),
                       "convergence": "plmc L-BFGS epsilon 1e-3 on ||g||/max(1,||x||); only the success status counts as converged"},
              "fit_script_sha256": code_sha, "msa_dir": str(MSA_DIR), "profile_npz_sha256": sha256_file(PROFILE_NPZ),
              "qualified_count": sum(t["qualification"]["qualified"] for t in table.values()), "domains": table}
    (args.out_dir / "qualification.json").write_text(json.dumps(report, indent=1) + "\n")
    manifest = {p.name: sha256_file(p) for p in sorted(args.out_dir.glob("*")) if p.is_file() and p.name != "manifest.json"}
    (args.out_dir / "manifest.json").write_text(json.dumps({"sha256": manifest}, indent=1) + "\n")
    print("qualified", report["qualified_count"], "of", len(domains))


if __name__ == "__main__":
    main()
