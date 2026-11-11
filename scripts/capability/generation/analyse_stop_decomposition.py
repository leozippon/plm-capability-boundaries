#!/usr/bin/env python3
"""Residue part against stop part of the cross-likelihood matrix, naturally terminated only.

For an attempt that ended on its close token, the model's total NLL of the sequence
plus its ending is the residue NLL (``sum_nll``) plus the stop NLL (``stop_nll`` =
-log p(close token | prompt + residues)). This reports, per pair and length stratum,
the tuned-minus-base contrasts of each part in nats per sequence, in the same four
forms as ``analyse_cross_likelihood.py`` (``mixture``, ``entropy_change``,
``forward_divergence``, ``reverse_divergence``; same definitions, same signs, same
length standardisation and bootstrap), so that the stop decision and residue choice
are in one currency.

Stop-token convention: the close token of the *scored model* (ProtGPT3 ``<|eos|>``;
ProGen2 the terminal marker ``2``), resolved by that model's tokenizer. An attempt
enters only if its stop status is ``native_terminal`` AND the whitespace-stripped raw
continuation is exactly its extracted residues immediately followed by the close
token; a native attempt where something else stood between the residues and the close
token is excluded and counted. Capped (budget-censored) attempts have no stop event
and are excluded, never imputed, and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse_cross_likelihood as acl  # noqa: E402

REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402

PARTS = ("stop_nll", "sum_nll")


def read_native(run_dir: Path, arm: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    rows, counts = [], {}
    for cell in sorted(run_dir.glob("ot_*")):
        summary = json.loads((cell / "unconditional_generation.json").read_text())
        if summary["arm"] != arm:
            continue
        close = summary["close_token"]
        for line in (cell / "attempts.jsonl").read_text().splitlines():
            r = json.loads(line)
            if not r["sequence"]:
                continue
            stratum = r["stratum"] or "outside"
            c = counts.setdefault(stratum, {"capped_excluded": 0, "native": 0, "native_adjacent_eligible": 0,
                                            "native_not_adjacent_excluded": 0})
            if r["stop_status"] == "budget_censored":
                c["capped_excluded"] += 1
            elif r["stop_status"] == "native_terminal":
                c["native"] += 1
                if "".join(r["raw_continuation"].split()).startswith(r["sequence"] + close):
                    c["native_adjacent_eligible"] += 1
                    rows.append((r["sequence_sha256"], len(r["sequence"]), stratum))
                else:
                    c["native_not_adjacent_excluded"] += 1
    digest, length, stratum = (np.array(c) for c in zip(*rows))
    return {"digest": digest, "length": length.astype(int), "stratum": stratum}, counts


def attach(scores_dir: Path, model: str, source: str, att: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    table = {}
    for line in (scores_dir / f"{model}__on__{source}.jsonl").read_text().splitlines():
        r = json.loads(line)
        table[r["sequence_sha256"]] = r
    missing = [d for d in att["digest"] if d not in table]
    if missing:
        raise SystemExit(f"{model} on {source}: {len(missing)} eligible attempts have no score")
    return {p: np.array([table[d][p] for d in att["digest"]]) for p in PARTS}


def analyse_pair(run_dir: Path, scores_dir: Path, base: str, tuned: str) -> dict:
    rng = np.random.default_rng(acl.SEED)
    nat, counts = {}, {}
    for a in (base, tuned):
        nat[a], counts[a] = read_native(run_dir, a)
    val = {(m, s): attach(scores_dir, m, s, nat[s]) for m in (base, tuned) for s in (base, tuned)}
    out = {"base": base, "tuned": tuned, "exclusion_counts": counts, "strata": {}}
    for stratum in sorted(set(nat[base]["stratum"]) | set(nat[tuned]["stratum"])):
        sel = {a: nat[a]["stratum"] == stratum for a in (base, tuned)}
        n = {a: int(sel[a].sum()) for a in sel}
        g = {"eligible": n}
        if min(n.values()) >= acl.MIN_N:
            sl = {a: nat[a]["length"][sel[a]] for a in sel}
            g["length_mean"] = {a: float(sl[a].mean()) for a in sl}
            for part in (*PARTS, "total"):
                f = {k: (v["sum_nll"] + v["stop_nll"] if part == "total" else v[part])[sel[k[1]]] for k, v in val.items()}
                g[part] = {
                    "own_means": {f"{a}_sequences_under_{m}_model": float(f[(m, a)].mean()) for (m, a) in f},
                    "mixture": acl.contrast(f[(base, tuned)], sl[tuned], f[(base, base)], sl[base], rng),
                    "entropy_change": acl.contrast(f[(tuned, tuned)], sl[tuned], f[(base, base)], sl[base], rng),
                    "forward_divergence": acl.paired(f[(base, tuned)], f[(tuned, tuned)], rng),
                    "reverse_divergence": acl.paired(f[(tuned, base)], f[(base, base)], rng),
                }
        out["strata"][stratum] = g
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", action="append")
    args = p.parse_args(argv)
    pairs = [(x.base, x.tuned) for x in ot.PAIRS
             if x.lane == ot.LANE_FREE_LENGTH and (not args.only or x.base in args.only)]
    args.out.write_text(json.dumps({"pairs": [analyse_pair(args.run_dir, args.scores, b, t) for b, t in pairs]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
