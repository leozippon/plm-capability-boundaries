#!/usr/bin/env python3
"""Per-tensor relative change between base and tuned, by tensor class (CPU, all common tensors)."""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(os.environ["MODEL_ROOT"])
OUT = Path(__file__).resolve().parents[4] / "results/long_cap_termination_20261110"
PAIRS = {"protgpt3-112m": ("ProtGPT3-112M", "ProtGPT3-112M-dpo"), "protgpt3-1.3b": ("ProtGPT3-1.3B", "ProtGPT3-1.3B-dpo"),
         "protgpt3-10b": ("ProtGPT3-10B", "ProtGPT3-10B-dpo")}
def idx(d):
    o = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys(): o[k] = f
    return o
out = {}
for n, (b, t) in PAIRS.items():
    ib, it = idx(ROOT / b), idx(ROOT / t)
    cls = defaultdict(list)
    for k in sorted(ib):
        with safe_open(ib[k], "pt") as h: a = h.get_tensor(k).float()
        with safe_open(it[k], "pt") as h: c = h.get_tensor(k).float()
        rel = float((a - c).norm() / a.norm().clamp_min(1e-12)); exact = bool(torch.equal(a, c))
        key = re.sub(r"\.\d+\.", ".#.", k)
        cls[key].append((rel, exact))
    out[n] = {k: {"n": len(v), "median_rel": sorted(r for r, _ in v)[len(v) // 2], "max_rel": max(r for r, _ in v), "n_identical_after_f32_cast": sum(e for _, e in v)} for k, v in cls.items()}
    print(n, flush=True)
    for k, v in out[n].items(): print("  %-60s n=%-4d med %.5f max %.5f identical %d" % (k, v["n"], v["median_rel"], v["max_rel"], v["n_identical_after_f32_cast"]), flush=True)
json.dump(out, open(OUT / "lineage_changed_tensors.json", "w"), indent=1)
