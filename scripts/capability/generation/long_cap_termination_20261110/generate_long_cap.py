#!/usr/bin/env python3
"""Companion generation at a 768-token cap, every other decoding setting untouched.

This is a new cohort, not an extension of the objective-tuned R6 cohort. It reuses the
lane's own loader, sampler, extractor and ledger writer, and changes exactly one entry
of ``unconditional_generation.POLICY`` in memory: ``max_new_tokens``. Nothing under
``src`` is modified. It writes the attempt ledger and nothing else (no structure subset,
no fold input).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.capability.generation import unconditional_generation as ug  # noqa: E402

NEW_CAP = 768

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--n", type=int, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

original = ug.POLICY["max_new_tokens"]
assert original == 400, original
ug.POLICY["max_new_tokens"] = NEW_CAP
ckpt = ug.checkpoint_for_arm(a.arm)
model, tok = ug.load_generator(a.arm, device=a.device, checkpoint=ckpt)
rows = ug.generate(a.arm, n=a.n, seed=a.seed, model=model, tokenizer=tok,
                   output_dir=a.out, checkpoint=ckpt)
print(json.dumps({"arm": a.arm, "n": len(rows), "seed": a.seed, "cap": NEW_CAP}))
