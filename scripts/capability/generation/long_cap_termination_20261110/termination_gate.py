#!/usr/bin/env python3
"""Termination gate for one or two attempt ledgers: python termination_gate.py LEDGER [LEDGER ...]"""
import json
import sys
from collections import Counter
import numpy as np
CAP = 768
def q(a, ps=(5, 25, 50, 75, 95, 99)):
    return {f"p{p}": float(np.percentile(a, p)) for p in ps} if len(a) else {}
out = {}
for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    n = len(rows); L = np.array([r["length"] for r in rows]); st = np.array([r["stop_status"] for r in rows])
    nat = L[st == "native_terminal"]; cen = L[st == "budget_censored"]; emp = int((st == "empty_or_noncanonical").sum())
    bins = [0, 16, 64, 128, 200, 300, 400, 500, 600, 700, 750, 767, 768, 10**6]
    hist = np.histogram(L, bins=[b for b in bins])[0]
    d = {"n": n, "native_terminal": int(len(nat)), "budget_censored": int(len(cen)), "empty_or_noncanonical": emp,
         "frac_native": len(nat) / n, "frac_censored": len(cen) / n,
         "frac_len_gt_400_all": float((L > 400).mean()), "frac_len_gt_400_native": float((nat > 400).mean()) if len(nat) else None,
         "n_len_gt_400_native": int((nat > 400).sum()),
         "censored_length_counts_top": Counter(cen.tolist()).most_common(5),
         "censored_exactly_cap": int((cen == CAP).sum()), "native_in_700_767": int(((nat >= 700) & (nat < CAP)).sum()),
         "all_lengths_q": q(L), "native_lengths_q": q(nat), "native_mean": float(nat.mean()) if len(nat) else None,
         "hist_edges": bins, "hist_all": hist.tolist(),
         "native_hist_0_800_by_50": np.histogram(nat, bins=list(range(0, 801, 50)))[0].tolist()}
    out[path] = d
    print(json.dumps({"ledger": path, **d}, indent=1))
if len(out) == 2:
    a, b = (np.array([r["length"] for r in map(json.loads, open(p)) if r["stop_status"] == "native_terminal"]) for p in out)
    lo, hi = max(np.percentile(a, 5), np.percentile(b, 5)), min(np.percentile(a, 95), np.percentile(b, 95))
    ed = np.arange(0, 801, 50); ha, hb = (np.histogram(x, bins=ed)[0] / len(x) for x in (a, b))
    print(json.dumps({"overlap_coefficient_50bin": float(np.minimum(ha, hb).sum()), "shared_p5_p95_range": [float(lo), float(hi)],
                      "ks_stat": float(max(abs(np.searchsorted(np.sort(a), v, side="right") / len(a) - np.searchsorted(np.sort(b), v, side="right") / len(b)) for v in np.unique(np.r_[a, b])))}))
