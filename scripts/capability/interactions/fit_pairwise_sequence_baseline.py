#!/usr/bin/env python3
"""Fit the pairwise sequence baseline Q and its independent-site control.

One regularized plmc pseudolikelihood model and one independently fitted
independent-site model per cohort background, both on identical homolog support.
The coupling regularization is chosen by held-out homolog sequence prediction
only: no stability measurement, no DMS score and no model output is read
anywhere in this file, and the selection criterion is the same gap-ignoring
conditional likelihood that plmc itself optimizes.

An independent-site log profile has a zero double contrast by construction for
aligned substitutions. This script measures that zero rather than replacing it
with a manufactured pairwise comparator.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from src.capability.core.sequence_alignment import alignment_rows, verify_rows_against_profile
from src.capability.context.homology import ALIGNMENT_FIELDS, parse_hits
from src.capability.stability.pairwise_stability import read_plmc, validate_cycle
from src.capability.context.profiles import (
    AA20, GAP_CODE, NEFF_IDENTITY_FLOOR, PROFILE_COVERAGE_FLOOR,
    REWEIGHT_IDENTITY_FLOOR, build_profile, sequence_weights)

#: Coupling L2 grid, spanning the upstream example's strong setting and two
#: decades either side of it. The field regularization stays at the upstream
#: example's value so one axis is selected and the comparison stays readable.
LAMBDA_E_GRID = (0.16, 0.8, 4.0, 16.0, 80.0)
LAMBDA_H = 0.01
MAX_ITER = 200
HELD_OUT_FRACTION = 0.2
SPLIT_SEED = 20260924
#: Independent-site pseudocount, shared with the repository's existing pair
#: estimator so the two independent-site declarations cannot drift.
PSEUDOCOUNT = 0.5
#: Support floors. Below these a background is reported as unsupported rather
#: than fitted with a number that cannot be defended.
MIN_TRAIN_ROWS = 20
MIN_HELD_OUT_ROWS = 5
#: A scored homolog row needs one residue to predict and one to condition on.
MIN_PRESENT_PER_ROW = 2


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
    return sha.hexdigest()


def held_out_mask(rows: np.ndarray, seed: int) -> np.ndarray:
    """Held-out mask from a stable hash of each aligned row's own residues.

    Hashing the row rather than its retrieval rank keeps the split independent of
    hit ordering, puts identical aligned rows on the same side, and cannot be
    influenced by any phenotype or by the order of the cohort.
    """

    boundary = int(HELD_OUT_FRACTION * (1 << 32))
    return np.array([
        int.from_bytes(hashlib.sha256(f"{seed}:".encode() + row.tobytes()).digest()[:4],
                       "big") < boundary
        for row in rows], dtype=bool)


def write_alignment(path: Path, wildtype: str, rows: np.ndarray) -> None:
    """FASTA with the wild type as the focus sequence, then the homolog rows."""

    letters = np.array(list(AA20) + ["-"])
    with path.open("w") as handle:
        handle.write(f">focus\n{wildtype}\n")
        for index, row in enumerate(rows):
            codes = np.where(row == GAP_CODE, len(AA20), row)
            handle.write(f">h{index:06d}\n{''.join(letters[codes])}\n")


def scored_support(rows: np.ndarray) -> np.ndarray:
    """Positions both models are scored on: non-gap, in rows with two or more."""

    present = rows != GAP_CODE
    return present & (present.sum(axis=1) >= MIN_PRESENT_PER_ROW)[:, None]


def independent_site(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted per-column amino-acid log probabilities over non-gap observations."""

    counts = np.full((rows.shape[1], len(AA20)), PSEUDOCOUNT, dtype=np.float64)
    for code in range(len(AA20)):
        counts[:, code] += (weights[:, None] * (rows == code)).sum(axis=0)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def profile_conditional(log_profile: np.ndarray, rows: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-site log likelihood of ``mask`` positions under a site profile.

    For an independent-site model the conditional and the marginal coincide, so
    this is directly comparable with :func:`plmc_conditional` on the same sites.
    """

    if not mask.any():
        raise ValueError("no held-out position to score")
    columns = np.broadcast_to(np.arange(rows.shape[1]), rows.shape)
    return float(log_profile[columns[mask], rows[mask]].mean())


def coupling_tensor(couplings: np.ndarray, length: int) -> np.ndarray:
    """Upper-triangle pair blocks as a symmetric ``(L*q, L*q)`` matrix.

    ``[i*q + a, j*q + b]`` carries J_ij(a, b) with the first residue axis on the
    lower-index position, the orientation verified against real plmc output. The
    diagonal blocks are zero so a site never conditions on itself.
    """

    q = couplings.shape[1]
    table = np.zeros((length, q, length, q), dtype=np.float64)
    k = 0
    for i in range(length):
        for j in range(i + 1, length):
            block = couplings[k].astype(np.float64)
            table[i, :, j, :] = block
            table[j, :, i, :] = block.T
            k += 1
    return table.reshape(length * q, length * q)


def plmc_conditional(fields: np.ndarray, couplings: np.ndarray, rows: np.ndarray,
                     mask: np.ndarray, *, block: int = 512) -> float:
    """Mean gap-ignoring per-site conditional log likelihood under h and J.

    Mirrors the objective plmc optimizes under ``-g``: a gapped position is
    neither scored nor allowed to contribute a coupling term, and the partition
    function runs over the twenty amino-acid states the model carries.
    """

    if not mask.any():
        raise ValueError("no held-out position to score")
    length, q = fields.shape
    table = coupling_tensor(couplings, length)
    present = (rows != GAP_CODE)
    total, count = 0.0, 0
    for start in range(0, len(rows), block):
        stop = min(start + block, len(rows))
        chunk, chunk_mask = rows[start:stop], mask[start:stop]
        if not chunk_mask.any():
            continue
        onehot = np.zeros((len(chunk), length, q), dtype=np.float64)
        rowsel, colsel = np.nonzero(present[start:stop])
        onehot[rowsel, colsel, chunk[rowsel, colsel]] = 1.0
        energy = (onehot.reshape(len(chunk), -1) @ table.T).reshape(len(chunk), length, q)
        energy += fields.astype(np.float64)[None]
        partition = np.logaddexp.reduce(energy, axis=2)
        # A gapped position carries no state in the model's alphabet; it is
        # clamped only so the gather stays in bounds and is then masked out.
        gathered = np.where(chunk == GAP_CODE, 0, chunk).astype(np.int64)
        observed = np.take_along_axis(energy, gathered[:, :, None], axis=2)[:, :, 0]
        total += float((observed - partition)[chunk_mask].sum())
        count += int(chunk_mask.sum())
    return total / count


def fit_one(job: dict) -> dict:
    """One background: reconstruct rows, sweep lambda_e, select, refit, contrast."""

    name, wildtype = job["name"], job["wildtype"]
    executable, work = job["plmc"], Path(job["work"])
    hits = job["hits"]
    record = {"name": name, "length": len(wildtype), "hits_retrieved": len(hits),
              "retrieval_ceiling": job["retrieval_ceiling"],
              "retrieval_ceiling_reached": len(hits) >= job["retrieval_ceiling"]}
    if not hits:
        record["status"] = "no_homolog_hits"
        record["rows"] = 0
        return record
    work.mkdir(parents=True, exist_ok=True)

    rows, weights, neff = alignment_rows(
        wildtype, name, hits, max_sequences=job["max_sequences"],
        coverage_floor=PROFILE_COVERAGE_FLOOR, reweight_identity=REWEIGHT_IDENTITY_FLOOR)
    profile = build_profile(
        wildtype, name, hits, max_sequences=job["max_sequences"],
        coverage_floor=PROFILE_COVERAGE_FLOOR, reweight_identity=REWEIGHT_IDENTITY_FLOOR,
        neff_identity_floor=NEFF_IDENTITY_FLOOR)
    verify_rows_against_profile(rows, weights, profile)
    record.update({"rows": int(len(rows)),
                   "neff_at_30_percent_identity": round(float(neff), 3),
                   "effective_depth_per_site": round(float(neff) / len(wildtype), 4),
                   "coordinate_weight_validation": "passed"})

    held = held_out_mask(rows, SPLIT_SEED)
    train, test = rows[~held], rows[held]
    record["train_rows"], record["held_out_rows"] = int(len(train)), int(len(test))
    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_HELD_OUT_ROWS:
        record["status"] = "insufficient_homolog_support"
        return record
    test_mask = scored_support(test)
    record["held_out_rows_scored"] = int((test_mask.any(axis=1)).sum())
    record["held_out_positions_scored"] = int(test_mask.sum())
    if not test_mask.any():
        record["status"] = "no_scorable_held_out_position"
        return record

    train_weights = sequence_weights(train, identity=REWEIGHT_IDENTITY_FLOOR)
    record["train_neff"] = round(float(train_weights.sum()), 3)
    site = independent_site(train, train_weights)
    record["independent_site_held_out_conditional"] = round(
        profile_conditional(site, test, test_mask), 6)

    alignment = work / "train.fa"
    write_alignment(alignment, wildtype, train)
    supplied = work / "train.weights"
    supplied.write_text("\n".join(["1.000000"] + [f"{w:.6f}" for w in train_weights]) + "\n")

    sweep = []
    for lambda_e in LAMBDA_E_GRID:
        params = work / f"train_le{lambda_e}.params"
        completed = subprocess.run(
            [executable, "-o", str(params), "-f", "focus", "-le", str(lambda_e),
             "-lh", str(LAMBDA_H), "-m", str(MAX_ITER), "-g", "-n", str(job["threads"]),
             "-w", str(supplied), str(alignment)], capture_output=True, text=True)
        entry = {"lambda_e": lambda_e, "returncode": completed.returncode}
        if completed.returncode != 0 or not params.exists():
            entry["status"] = "plmc_failed"
            entry["log_tail"] = completed.stderr.strip().splitlines()[-3:]
            sweep.append(entry)
            continue
        fitted = read_plmc(params)
        if fitted.target != wildtype:
            raise RuntimeError(f"{name}: plmc focus {fitted.target!r} is not the wild type")
        entry["converged"] = "Minimization success" in completed.stderr
        entry["held_out_conditional"] = round(
            plmc_conditional(fitted.fields, fitted.couplings, test, test_mask), 6)
        entry["coupling_frobenius_mean"] = round(
            float(np.sqrt((fitted.couplings.astype(np.float64) ** 2).sum(axis=(1, 2))).mean()), 6)
        entry["status"] = "fitted"
        sweep.append(entry)
        params.unlink()
    record["sweep"] = sweep
    usable = [entry for entry in sweep if entry.get("status") == "fitted"]
    if not usable:
        record["status"] = "all_regularizations_failed"
        return record
    best = max(usable, key=lambda entry: entry["held_out_conditional"])
    record["selected_lambda_e"] = best["lambda_e"]
    record["selected_held_out_conditional"] = best["held_out_conditional"]
    record["held_out_gain_over_independent_site"] = round(
        best["held_out_conditional"] - record["independent_site_held_out_conditional"], 6)
    record["selected_at_grid_edge"] = best["lambda_e"] in (LAMBDA_E_GRID[0], LAMBDA_E_GRID[-1])
    record["converged_in_sweep"] = sum(bool(entry.get("converged")) for entry in usable)

    # Refit on the full support at the selected value; this is the delivered model.
    full = work / "full.fa"
    write_alignment(full, wildtype, rows)
    full_weights = work / "full.weights"
    full_weights.write_text("\n".join(["1.000000"] + [f"{w:.6f}" for w in weights]) + "\n")
    params = work / "selected.params"
    completed = subprocess.run(
        [executable, "-o", str(params), "-f", "focus", "-le", str(best["lambda_e"]),
         "-lh", str(LAMBDA_H), "-m", str(MAX_ITER), "-g", "-n", str(job["threads"]),
         "-w", str(full_weights), str(full)], capture_output=True, text=True)
    if completed.returncode != 0 or not params.exists():
        record["status"] = "final_refit_failed"
        record["final_log_tail"] = completed.stderr.strip().splitlines()[-3:]
        return record
    record["final_converged"] = "Minimization success" in completed.stderr
    record["params"] = {"path": str(params), "sha256": digest(params),
                        "bytes": params.stat().st_size}
    fitted = read_plmc(params)
    site_full = independent_site(rows, weights)

    contrasts = []
    for cycle in job["cycles"]:
        wild, single_a, single_b, double = cycle["sequences"]
        first, second = validate_cycle(wild, single_a, single_b, double)
        codes = {state: np.array([AA20.index(residue) for residue in state])
                 for state in (wild, single_a, single_b, double)}
        flat = {state: float(site_full[np.arange(len(state)), code].sum())
                for state, code in codes.items()}
        contrasts.append({
            "positions": sorted([first + 1, second + 1]),
            "separation": cycle["separation"],
            "pairwise_contrast": fitted.cycle(wild, single_a, single_b, double),
            "independent_site_contrast": (flat[double] - flat[single_a]
                                          - flat[single_b] + flat[wild]),
        })
    record["contrasts"] = contrasts
    record["status"] = "fitted"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--plmc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--retrieval-ceiling", type=int, required=True,
                        help="the search's --max-target-seqs, so truncation is reportable")
    parser.add_argument("--max-sequences", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--plmc-threads", type=int, default=1)
    args = parser.parse_args()
    report = args.out_dir / "baseline_q.json"
    if report.exists():
        raise FileExistsError(report)

    cohort = json.loads(args.cohort.read_text())
    wanted = {row["name"]: row for row in cohort["backgrounds"]}
    grouped: dict[str, list] = {name: [] for name in wanted}
    for hit in parse_hits(args.hits, fields=ALIGNMENT_FIELDS):
        if hit.query in grouped:
            grouped[hit.query].append(hit)

    work_root = args.out_dir / "work"
    jobs = [{"name": name, "wildtype": row["cycles"][0]["sequences"][0],
             "hits": grouped[name], "plmc": str(args.plmc),
             "work": str(work_root / name.replace("/", "_").replace("|", "_")),
             "max_sequences": args.max_sequences, "threads": args.plmc_threads,
             "retrieval_ceiling": args.retrieval_ceiling,
             "cycles": row["cycles"]}
            for name, row in sorted(wanted.items())]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(fit_one, jobs):
            records.append(record)
            print(record["name"], record["status"], record.get("rows"),
                  record.get("selected_lambda_e"),
                  record.get("held_out_gain_over_independent_site"), flush=True)

    fitted = [r for r in records if r["status"] == "fitted"]
    gains = [r["held_out_gain_over_independent_site"] for r in fitted]
    zero = [abs(c["independent_site_contrast"]) for r in fitted for c in r["contrasts"]]
    pairwise = [abs(c["pairwise_contrast"]) for r in fitted for c in r["contrasts"]]
    summary = {
        "backgrounds": len(records), "fitted": len(fitted),
        "failures": {r["name"]: r["status"] for r in records if r["status"] != "fitted"},
        "retrieval_ceiling_reached": sum(r["retrieval_ceiling_reached"] for r in records),
        "selected_lambda_e_counts": {
            str(value): sum(1 for r in fitted if r["selected_lambda_e"] == value)
            for value in LAMBDA_E_GRID},
        "selected_at_grid_edge": sum(r["selected_at_grid_edge"] for r in fitted),
        "converged_final_refits": sum(bool(r.get("final_converged")) for r in fitted),
        "sweep_fits_converged": sum(r.get("converged_in_sweep", 0) for r in fitted),
        "neff_quantiles": (np.quantile([r["neff_at_30_percent_identity"] for r in fitted],
                                       [0, .25, .5, .75, 1]).round(2).tolist() if fitted else None),
        "effective_depth_per_site_quantiles": (
            np.quantile([r["effective_depth_per_site"] for r in fitted],
                        [0, .25, .5, .75, 1]).round(4).tolist() if fitted else None),
        "rows_quantiles": (np.quantile([r["rows"] for r in fitted],
                                       [0, .25, .5, .75, 1]).tolist() if fitted else None),
        "held_out_gain_over_independent_site": (
            {"min": round(min(gains), 4), "median": round(float(np.median(gains)), 4),
             "max": round(max(gains), 4),
             "backgrounds_with_positive_gain": sum(g > 0 for g in gains)} if gains else None),
        "independent_site_double_contrast_max_absolute": max(zero) if zero else None,
        "pairwise_double_contrast_absolute_quantiles": (
            np.quantile(pairwise, [0, .5, .9, 1]).round(6).tolist() if pairwise else None),
        "grid": {"lambda_e": list(LAMBDA_E_GRID), "lambda_h": LAMBDA_H,
                 "max_iter": MAX_ITER, "gap_ignoring": True,
                 "split_seed": SPLIT_SEED, "held_out_fraction": HELD_OUT_FRACTION,
                 "min_train_rows": MIN_TRAIN_ROWS, "min_held_out_rows": MIN_HELD_OUT_ROWS,
                 "pseudocount": PSEUDOCOUNT, "plmc_threads": args.plmc_threads,
                 "selection_criterion": "mean gap-ignoring per-site conditional log likelihood "
                                        "of held-out homolog rows; no phenotype is read"},
        "plmc": {"path": str(args.plmc), "sha256": digest(args.plmc)},
        "hits": {"path": str(args.hits), "sha256": digest(args.hits)},
        "cohort": {"path": str(args.cohort), "sha256": digest(args.cohort)},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema": "pairwise_sequence_baseline_v1",
                                  "summary": summary, "backgrounds": records}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
