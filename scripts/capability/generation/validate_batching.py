#!/usr/bin/env python3
"""Check batched, padded scoring against single-sequence, no-padding recomputation.

For each arm: take the length-spread sample, score it (a) with ``score_batch`` as one
mixed-length right-padded batch, (b) with an independent path -- tokenizer-encoded
``prompt + residues`` string, one sequence per forward, no attention mask, no padding,
log-softmax gathered token by token -- and compare. Two further checks make the first
one meaningful: the tokenizer-string encoding must equal the per-residue-id encoding,
and a deliberately wrong batching (LEFT padding, no attention mask) must be visibly
off, otherwise the comparison could not have caught a masking defect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_cohort_likelihood as sc  # noqa: E402


def single_reference(scorer, sequence):
    """Independent recomputation: tokenizer string encode, batch of one, no mask."""

    import torch

    tok, model = scorer["tokenizer"], scorer["model"]
    spec_prompt_len = len(scorer["prompt_ids"])
    from src.capability.generation import unconditional_generation as ug

    arm_spec = ug.generation_point(scorer["arm"])
    ids = tok(arm_spec.prompt + sequence, add_special_tokens=arm_spec.add_special_tokens)["input_ids"]
    expected = sc.encode(scorer, sequence)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    nll = [-float(logp[t - 1, ids[t]]) for t in range(spec_prompt_len, len(ids))]
    return list(ids) == expected, sum(nll), len(nll)


def wrong_left_pad_no_mask(scorer, sequences):
    """The defect the check must be able to see: left padding, mask ignored."""

    import torch

    model = scorer["model"]
    device = next(model.parameters()).device
    enc = [sc.encode(scorer, s) for s in sequences]
    width = max(map(len, enc))
    ids = torch.tensor([[scorer["pad_id"]] * (width - len(e)) + e for e in enc], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    logp = torch.log_softmax(logits.float(), dim=-1)
    pl = len(scorer["prompt_ids"])
    out = []
    for r, e in enumerate(enc):
        off = width - len(e)
        out.append(sum(-float(logp[r, off + t - 1, e[t]]) for t in range(pl, len(e))))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--sample", type=Path, required=True, help="jsonl of {arm, sequence, ...} rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--n", type=int, default=12, help="sequences per arm drawn evenly by length")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.sample.read_text().splitlines()]
    for arm in args.arm:
        by_length = {len(r["sequence"]): r["sequence"] for r in rows if r["arm"] == arm}
        seqs = [by_length[k] for k in sorted(by_length)]  # one per distinct length: more padding
        pick = [seqs[round(i * (len(seqs) - 1) / (args.n - 1))] for i in range(args.n)]
        scorer = sc.load_scorer(arm, args.device, args.dtype)
        scorer["arm"] = arm
        batched = sc.score_batch(scorer, pick)
        # a second batching of the same sequences in a different order and composition
        shuffled = sc.score_batch(scorer, pick[::-1][:6])
        ref = [single_reference(scorer, s) for s in pick]
        wrong = wrong_left_pad_no_mask(scorer, pick)
        d_mean = [abs(b[1] / b[0] - r[1] / r[2]) for b, r in zip(batched, ref)]
        d_sum = [abs(b[1] - r[1]) for b, r in zip(batched, ref)]
        d_wrong = [abs(w - r[1]) / r[2] for w, r in zip(wrong, ref)]
        d_recomp = [abs(shuffled[i][1] / shuffled[i][0] - batched[len(pick) - 1 - i][1] / batched[len(pick) - 1 - i][0]) for i in range(len(shuffled))]
        report = {
            "arm": arm, "dtype": scorer["dtype"], "n": len(pick),
            "lengths": [len(s) for s in pick],
            "prompt_ids": scorer["prompt_ids"],
            "tokenizer_string_encoding_equals_residue_ids": all(r[0] for r in ref),
            "n_tokens_equal": all(b[0] == r[2] == len(s) for b, r, s in zip(batched, ref, pick)),
            "batched_mean_nll": [b[1] / b[0] for b in batched],
            "single_mean_nll": [r[1] / r[2] for r in ref],
            "max_abs_diff_mean_nll_batched_vs_single": max(d_mean),
            "max_abs_diff_sum_nll_batched_vs_single": max(d_sum),
            "max_abs_diff_mean_nll_other_batch_composition": max(d_recomp),
            "negative_control_left_pad_no_mask_max_abs_diff_mean_nll": max(d_wrong),
            "negative_control_left_pad_no_mask_median_abs_diff_mean_nll": sorted(d_wrong)[len(d_wrong) // 2],
        }
        (args.out / f"{arm}.validation.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k.startswith(("max", "neg", "token", "n_tok", "dtype", "arm"))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
