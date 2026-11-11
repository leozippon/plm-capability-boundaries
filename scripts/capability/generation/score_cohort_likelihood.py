#!/usr/bin/env python3
"""Mean per-token NLL of cohort sequences under a chosen model: the cross-likelihood matrix.

The generation ledgers of the objective-tuned campaign record each sequence but no
likelihood. This pass scores the sequences of one or more *source* arms under one
*model* arm. Scoring every arm's sequences under both members of its base/tuned pair
gives the 2x2 cross-likelihood matrix; ``--source`` equal to ``--model`` is the
own-model cell.

Convention
----------
* The model, tokenizer and declared checkpoint are those of ``--model``, and the
  prompt contract is checked for the *model* (the prompt it was trained behind), not
  for the arm that produced the sequence. Residue ids come from the scored model's
  own tokenizer.
* The prompt is the generation prompt of ``--model`` (ProtGPT3: ``<|bos|>`` then
  direction ``1``; ProGen2: direction marker alone); ``prompt_contract`` refuses one
  that does not encode to its declared marker ids.
* Scored tokens are the extracted residues, one id per residue; the stop decision is
  not scored, so censored and natively terminated sequences are scored alike.
* Temperature 1, no top-k or top-p: the model's own likelihood, not the probability
  under the sampling procedure (T 0.85, top-k 50, top-p 0.95).
* Float32 weights; right padding with an explicit attention mask; residue-position
  mask from true lengths; length-sorted batches.
* Unit of scoring is the distinct (source arm, sequence). Duplicate attempts share a
  score; the ledgers stay the denominator.

Batching gate
-------------
After each model loads, ``GATE_SEQUENCES`` sequences of spread length are scored as
one padded batch and one by one with no padding. If the largest absolute difference
in mean NLL exceeds ``--gate-threshold`` (default 1e-4; float32 agreement measured
on the L20 is at most 1.2e-5) the pass aborts before writing anything for that model.

Output, ``<out>/<model>__on__<source>.jsonl``: ``sequence_sha256, n_tokens, sum_nll,
mean_nll``; plus ``<model>__on__<source>.manifest.json`` and one
``<model>.gate.json`` per model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_scorer(arm: str, device: str, dtype: str = "float32"):
    """Model, tokenizer, prompt ids and residue ids, all under the generation contract.

    Weights load directly in ``dtype`` (default float32, not the bfloat16 the sampler
    used): the likelihood is a ranking statistic and bf16 forward noise (about 5e-3
    in mean NLL between batch compositions) is not needed here.
    """

    from src.capability.core.arms import load_arm_spec
    from src.capability.generation import objective_tuned_arms as ot
    from src.capability.generation import unconditional_generation as ug

    spec = ug.generation_point(arm)
    handle = load_arm_spec(ot.spec_for(arm), device=device, dtype=dtype)
    model, tokenizer = handle.model, handle.tokenizer
    model.eval()
    ot.prompt_contract(arm, tokenizer, spec.prompt)
    prompt_ids = [
        int(i) for i in tokenizer(spec.prompt, add_special_tokens=spec.add_special_tokens)["input_ids"]
    ]
    residue_ids = ot.residue_alphabet_contract(arm, tokenizer)["residue_ids"]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    return {
        "model": model,
        "tokenizer": tokenizer,
        "prompt_ids": prompt_ids,
        "residue_ids": residue_ids,
        "pad_id": int(pad_id),
        "dtype": dtype,
    }


def encode(scorer: dict[str, Any], sequence: str) -> list[int]:
    ids = scorer["residue_ids"]
    stray = sorted(set(sequence) - set(ids))
    if not sequence or stray:
        raise ValueError(f"sequence is empty or carries non-residue characters {stray}")
    return scorer["prompt_ids"] + [ids[c] for c in sequence]


def score_batch(scorer: dict[str, Any], sequences: list[str]) -> list[tuple[int, float]]:
    """(n_tokens, sum_nll) per sequence, right-padded with an attention mask."""

    import torch

    model = scorer["model"]
    device = next(model.parameters()).device
    prompt_len = len(scorer["prompt_ids"])
    encoded = [encode(scorer, s) for s in sequences]
    width = max(len(e) for e in encoded)
    ids = torch.full((len(encoded), width), scorer["pad_id"], dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for row, e in enumerate(encoded):
        ids[row, : len(e)] = torch.tensor(e)
        mask[row, : len(e)] = 1
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits if hasattr(out, "logits") else out[0]
        # logits at position t predict token t+1; residues start at prompt_len.
        logp = torch.log_softmax(logits[:, prompt_len - 1 : width - 1].float(), dim=-1)
        target = ids[:, prompt_len:]
        token_nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        valid = mask[:, prompt_len:].bool()
        sums = (token_nll * valid).sum(dim=1)
    return [(int(valid[i].sum()), float(sums[i])) for i in range(len(encoded))]


def score_all(scorer: dict[str, Any], sequences: list[str], *, max_batch: int, max_tokens: int):
    """Score distinct sequences length-sorted; return {sequence: (n, sum_nll)}."""

    order = sorted(set(sequences), key=len)
    results: dict[str, tuple[int, float]] = {}
    i = 0
    while i < len(order):
        # sorted ascending, so the batch's width is its last member's length
        j = i + 1
        while j < len(order) and j - i < max_batch and (j - i + 1) * len(order[j]) <= max_tokens:
            j += 1
        batch = order[i:j]
        for seq, res in zip(batch, score_batch(scorer, batch)):
            results[seq] = res
        i = j
    return results


def read_arm_ledgers(run_dir: Path, arm: str) -> list[dict[str, Any]]:
    """Attempt rows of every cell of ``arm``; refuse a ledger that names another arm."""

    rows: list[dict[str, Any]] = []
    cells = 0
    for cell in sorted(run_dir.glob("ot_*")):
        summary = json.loads((cell / "unconditional_generation.json").read_text(encoding="utf-8"))
        if summary["arm"] != arm:
            continue
        cells += 1
        for line in (cell / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["arm"] != arm:
                raise SystemExit(f"{cell}: ledger row names arm {row['arm']!r}, expected {arm!r}")
            rows.append(row)
    if not cells:
        raise SystemExit(f"no cell of arm {arm!r} under {run_dir}")
    return rows


GATE_SEQUENCES = 8


def batching_gate(scorer: dict[str, Any], sequences: list[str], threshold: float) -> dict[str, Any]:
    """Batched padded scores against one-by-one unpadded scores; abort past ``threshold``."""

    ordered = sorted(set(sequences), key=len)
    k = min(GATE_SEQUENCES, len(ordered))
    pick = [ordered[round(i * (len(ordered) - 1) / max(k - 1, 1))] for i in range(k)]
    batched = score_batch(scorer, pick)
    worst = 0.0
    for seq, (n, total) in zip(pick, batched):
        n1, total1 = score_batch(scorer, [seq])[0]
        if n1 != n:
            raise SystemExit(f"batching gate: token count {n} batched against {n1} alone")
        worst = max(worst, abs(total / n - total1 / n1))
    record = {"lengths": [len(s) for s in pick], "max_abs_diff_mean_nll": worst,
              "threshold": threshold, "passed": worst <= threshold}
    if not record["passed"]:
        raise SystemExit(f"batching gate FAILED: {worst:.3e} > {threshold:.1e}; nothing written")
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="directory holding the ot_* generation cells")
    p.add_argument("--model", required=True, help="arm whose checkpoint scores")
    p.add_argument("--source", action="append", required=True, help="arm whose sequences are scored")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=8192, help="padded tokens per batch")
    p.add_argument("--gate-threshold", type=float, default=1e-4)
    p.add_argument("--limit", type=int, default=None, help="score only the first N distinct sequences (smoke)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    pools = {}
    for source in args.source:
        rows = read_arm_ledgers(args.run_dir, source)
        non_empty = [r for r in rows if r["sequence"]]
        for r in non_empty:
            if hashlib.sha256(r["sequence"].encode()).hexdigest() != r["sequence_sha256"]:
                raise SystemExit(f"{source}: sequence digest mismatch in row {r['id']}")
        distinct = sorted({r["sequence"] for r in non_empty}, key=len)
        pools[source] = (len(rows), len(non_empty), distinct[: args.limit] if args.limit else distinct)

    scorer = load_scorer(args.model, args.device, args.dtype)
    gate = batching_gate(scorer, pools[args.source[0]][2], args.gate_threshold)
    (args.out / f"{args.model}.gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    print(f"{args.model}: batching gate passed, max diff {gate['max_abs_diff_mean_nll']:.2e}", flush=True)

    for source, (attempts, non_empty, distinct) in pools.items():
        results = score_all(scorer, distinct, max_batch=args.max_batch, max_tokens=args.max_tokens)
        stem = f"{args.model}__on__{source}"
        with (args.out / f"{stem}.jsonl").open("w", encoding="utf-8") as fh:
            for seq in distinct:
                n, total = results[seq]
                fh.write(json.dumps({
                    "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                    "n_tokens": n, "sum_nll": total, "mean_nll": total / n,
                }) + "\n")
        (args.out / f"{stem}.manifest.json").write_text(json.dumps({
            "model": args.model, "source": source, "attempts": attempts,
            "non_empty_attempts": non_empty, "distinct_sequences_scored": len(distinct),
            "limit": args.limit, "prompt_ids": scorer["prompt_ids"], "dtype": scorer["dtype"],
            "temperature": 1.0, "scored_tokens": "extracted residues only; stop token excluded",
        }, indent=1), encoding="utf-8")
        print(f"{stem}: scored {len(distinct)} distinct sequences", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
