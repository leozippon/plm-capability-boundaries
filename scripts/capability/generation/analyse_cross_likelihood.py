#!/usr/bin/env python3
"""Cross-likelihood matrix of the objective-tuned pairs, read within length and cap.

Input: the ``ot_*`` generation ledgers and the ``<model>__on__<source>.jsonl`` score
files written by ``score_cohort_likelihood.py`` for every (model, source) of a pair.
Attempts are the denominator; a score is joined by sequence digest, and a non-empty
attempt without a score refuses the run.

Mean per-token NLL is only comparable at matched length, and the tuned arms are less
budget-censored than their bases, so every cell is split by ledger length stratum and
by stop status (``capped`` = budget_censored, ``native`` = native_terminal).

Decomposition. With T the tuned and B the base model, the mixture contrast
``mixture`` (tuned sequences under B minus base sequences under B) equals, exactly in
the sample means, ``forward_divergence`` (mean over tuned sequences of NLL_B - NLL_T)
plus ``entropy_change`` (tuned sequences under T minus base sequences under B, the
own-model NLLs). Narrowing shows as a lower tuned own-NLL; mass genuinely moved shows
as a large divergence. ``forward_divergence``: preference optimisation optimises this
ratio, so its SIGN is expected a priori and is not evidence; only its magnitude says
how far the distribution moved. ``reverse_divergence`` (mean over base sequences of
NLL_T - NLL_B) is how strongly the tuned model disfavours its parent's samples.
Sequences were sampled at T=0.85 and are scored at T=1, so entropy-plus-divergence is
an approximation (the samples are not draws from the scored distribution); the
cross-model comparison is fair because both arms were sampled alike.

Length: every unpaired contrast is given raw and standardised over 32-residue length
bins to the first-named arm's length distribution (bins with fewer than ``MIN_BIN``
attempts in either arm dropped, coverage reported). Intervals are 95% percentile
bootstraps over attempts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation import objective_tuned_arms as ot  # noqa: E402

BOOT = 2000
BIN = 32
MIN_BIN = 5
MIN_N = 30
SEED = 20261110


def read_attempts(run_dir: Path, arm: str) -> dict[str, np.ndarray]:
    rows = []
    for cell in sorted(run_dir.glob("ot_*")):
        if json.loads((cell / "unconditional_generation.json").read_text())["arm"] != arm:
            continue
        for line in (cell / "attempts.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["sequence"]:
                rows.append((r["sequence_sha256"], len(r["sequence"]), r["stop_status"], r["stratum"] or "outside"))
    if not rows:
        raise SystemExit(f"no non-empty attempts for {arm}")
    digest, length, stop, stratum = (np.array(c) for c in zip(*rows))
    cap = np.where(stop == "budget_censored", "capped", np.where(stop == "native_terminal", "native", stop))
    return {"digest": digest, "length": length.astype(int), "cap": cap, "stratum": stratum}


def attach(scores_dir: Path, model: str, source: str, attempts: dict[str, np.ndarray]) -> np.ndarray:
    table = {}
    for line in (scores_dir / f"{model}__on__{source}.jsonl").read_text().splitlines():
        r = json.loads(line)
        table[r["sequence_sha256"]] = r["mean_nll"]
    missing = [d for d in attempts["digest"] if d not in table]
    if missing:
        raise SystemExit(f"{model} on {source}: {len(missing)} attempts have no score")
    return np.array([table[d] for d in attempts["digest"]])


def _std_diff(x, lx, y, ly):
    """x-mean minus y-mean, standardised over length bins to x's distribution."""
    bx, by = lx // BIN, ly // BIN
    num = wsum = 0.0
    for b in np.unique(bx):
        ix, iy = bx == b, by == b
        if ix.sum() >= MIN_BIN and iy.sum() >= MIN_BIN:
            w = ix.sum()
            num += w * (x[ix].mean() - y[iy].mean())
            wsum += w
    return (num / wsum if wsum else float("nan")), wsum / len(x)


def _ci(samples):
    s = np.asarray(samples)
    s = s[~np.isnan(s)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))] if len(s) else [None, None]


def contrast(x, lx, y, ly, rng):
    raw = float(x.mean() - y.mean())
    adj, cover = _std_diff(x, lx, y, ly)
    raws, adjs = [], []
    for _ in range(BOOT):
        i, j = rng.integers(0, len(x), len(x)), rng.integers(0, len(y), len(y))
        raws.append(x[i].mean() - y[j].mean())
        adjs.append(_std_diff(x[i], lx[i], y[j], ly[j])[0])
    return {"n": [len(x), len(y)], "raw": raw, "raw_ci": _ci(raws),
            "length_standardised": None if np.isnan(adj) else float(adj),
            "length_standardised_ci": _ci(adjs), "length_coverage": float(cover)}


def paired(a, b, rng):
    d = a - b
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    return {"n": len(d), "mean": float(d.mean()), "ci": _ci(boots)}


def analyse_pair(run_dir: Path, scores_dir: Path, base: str, tuned: str) -> dict:
    rng = np.random.default_rng(SEED)
    att = {a: read_attempts(run_dir, a) for a in (base, tuned)}
    nll = {(m, s): attach(scores_dir, m, s, att[s]) for m in (base, tuned) for s in (base, tuned)}
    out = {"base": base, "tuned": tuned, "groups": {}}
    keys = sorted({(st, c) for a in att.values() for st, c in zip(a["stratum"], a["cap"])})
    for stratum, cap in keys:
        sel = {a: (att[a]["stratum"] == stratum) & (att[a]["cap"] == cap) for a in (base, tuned)}
        g = {"attempts": {a: int(sel[a].sum()) for a in sel},
             "length_mean": {a: (float(att[a]["length"][sel[a]].mean()) if sel[a].any() else None) for a in sel},
             "matrix": {}}
        for (m, s), v in nll.items():
            x = v[sel[s]]
            g["matrix"][f"{s}_sequences_under_{m}_model"] = (
                {"n": len(x), "mean_nll": float(x.mean()),
                 "ci": _ci([x[rng.integers(0, len(x), len(x))].mean() for _ in range(BOOT)])} if len(x) else None)
        if min(g["attempts"].values()) >= MIN_N:
            sl = {a: att[a]["length"][sel[a]] for a in sel}
            f = {k: v[sel[k[1]]] for k, v in nll.items()}
            g["mixture"] = contrast(f[(base, tuned)], sl[tuned], f[(base, base)], sl[base], rng)
            g["entropy_change"] = contrast(f[(tuned, tuned)], sl[tuned], f[(base, base)], sl[base], rng)
            g["forward_divergence"] = paired(f[(base, tuned)], f[(tuned, tuned)], rng)
            g["reverse_divergence"] = paired(f[(tuned, base)], f[(base, base)], rng)
            g["identity_residual"] = g["mixture"]["raw"] - g["forward_divergence"]["mean"] - g["entropy_change"]["raw"]
        out["groups"][f"{stratum}|{cap}"] = g
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", action="append", help="restrict to the pair with this base arm (smoke)")
    args = p.parse_args(argv)
    pairs = [(x.base, x.tuned) for x in ot.PAIRS
             if x.lane == ot.LANE_FREE_LENGTH and (not args.only or x.base in args.only)]
    result = {"bootstrap": BOOT, "length_bin": BIN, "min_bin": MIN_BIN, "min_n": MIN_N, "seed": SEED,
              "pairs": [analyse_pair(args.run_dir, args.scores, b, t) for b, t in pairs]}
    args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
