#!/usr/bin/env python3
"""CPU-only lineage check of the four base/tuned pairs: config diff, tensor inventory,
and weight similarity against a same-shape unrelated baseline (other layers of the base)."""
import json
import os
import re
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(os.environ["MODEL_ROOT"])
OUT = Path(__file__).resolve().parents[4] / "results/long_cap_termination_20261110"
PAIRS = {"protgpt3-112m": ("ProtGPT3-112M", "ProtGPT3-112M-dpo"),
         "protgpt3-1.3b": ("ProtGPT3-1.3B", "ProtGPT3-1.3B-dpo"),
         "protgpt3-10b": ("ProtGPT3-10B", "ProtGPT3-10B-dpo"),
         "progen2-small": ("progen2-small-hugohrban", "progen2-small-mix7")}

def index(d):
    files = sorted(d.glob("*.safetensors")); out = {}
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys():
                s = h.get_slice(k); out[k] = (f, tuple(s.get_shape()), s.get_dtype())
    return out

def load(idx, k):
    with safe_open(idx[k][0], "pt") as h:
        return h.get_tensor(k).float().flatten()

def sim(a, b):
    cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
    ac, bc = a - a.mean(), b - b.mean()
    r = float(torch.dot(ac, bc) / (ac.norm() * bc.norm()))
    return {"cosine": cos, "pearson": r, "rel_diff": float((a - b).norm() / a.norm())}

res = {}
for name, (b, t) in PAIRS.items():
    cb, ct = (json.load(open(ROOT / d / "config.json")) for d in (b, t))
    diff = {k: [cb.get(k, "<absent>"), ct.get(k, "<absent>")] for k in sorted(set(cb) | set(ct)) if cb.get(k, "<absent>") != ct.get(k, "<absent>")}
    ib, it = index(ROOT / b), index(ROOT / t)
    only_b, only_t = sorted(set(ib) - set(it)), sorted(set(it) - set(ib))
    shape_mis = [k for k in set(ib) & set(it) if ib[k][1] != it[k][1]]
    def count(i):
        return sum(int(np.prod(v[1])) for v in i.values())
    common = sorted(set(ib) & set(it) - set(shape_mis))
    big = sorted(common, key=lambda k: -int(np.prod(ib[k][1])))[:6]
    # plus a spread of mid-size matrices from different depths
    rng = np.random.default_rng(0)
    mats = [k for k in common if len(ib[k][1]) == 2]
    pick = list(dict.fromkeys(big + [mats[i] for i in rng.choice(len(mats), min(6, len(mats)), replace=False)]))
    tens = {k: sim(load(ib, k), load(it, k)) for k in pick}
    # unrelated baseline: same-shaped tensor of a different layer of the base
    ctrl = {}
    for k in pick:
        m = re.search(r"\.(\d+)\.", k)
        if not m:
            continue
        sib = [j for j in ib if j != k and ib[j][1] == ib[k][1] and re.sub(r"\.\d+\.", ".#.", j) == re.sub(r"\.\d+\.", ".#.", k)]
        if sib:
            ctrl[k] = sim(load(ib, k), load(ib, sib[0])) | {"vs": sib[0]}
    res[name] = {"base_dir": b, "tuned_dir": t, "config_diff": diff,
                 "params": [count(ib), count(it)], "n_tensors": [len(ib), len(it)],
                 "dtypes": [sorted({v[2] for v in ib.values()}), sorted({v[2] for v in it.values()})],
                 "only_in_base": only_b[:10], "only_in_tuned": only_t[:10], "n_only": [len(only_b), len(only_t)],
                 "shape_mismatches": shape_mis[:10], "n_shape_mismatch": len(shape_mis),
                 "tuned_vs_base": tens, "unrelated_control_same_shape_other_layer": ctrl}
    print(name, json.dumps({k: res[name][k] for k in ("params", "n_tensors", "n_only", "n_shape_mismatch", "config_diff")}), flush=True)
    for k, v in tens.items():
        print("  ", k, {a: round(x, 4) for a, x in v.items()}, "| ctrl", {a: round(x, 4) for a, x in ctrl.get(k, {}).items() if a != "vs"}, flush=True)
json.dump(res, open(OUT / "lineage_check.json", "w"), indent=1)
