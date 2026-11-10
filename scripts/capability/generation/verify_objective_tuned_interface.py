#!/usr/bin/env python3
"""Verify every objective-tuned arm's native interface, and write the record.

What this refuses, and why each refusal is here
===============================================

The campaign is a paired contrast between checkpoints nobody in this project
trained, so three things have to be true before a single attempt counts, and all
three fail silently if they are merely assumed.

**The prompt is the rendering.** ProtGPT3's published recipe builds the tokenizer
with ``add_bos_token=True`` and prompts with the bare direction token; the
free-length lane spells ``<|bos|>`` into the prompt and leaves the published
``add_bos_token: false`` default alone. Both reach ``(bos, direction)``. Doing
both at once reaches ``(bos, bos, direction)`` and dropping the BOS reaches
``(direction,)`` -- one a prompt the checkpoint never saw, the other the rendering
this repository already priced at 3.0495 nats/residue against 1.7957. Neither is
visible in a decoded string, so the check is on token ids.

**The decoded string is residues.** These tokenizers are per-residue and decode
with spaces between tokens, so the lane's extractor has to be exercised on real
continuations rather than trusted: this writes the realised residue strings and
refuses a sample that extracted nothing at all. Empty and budget-censored
continuations stay in the denominator -- that is the lane's rule and this does not
weaken it -- but an arm whose *every* attempt extracted empty has an interface
fault, not a measurement.

**The pair is matched.** A tuned-minus-base difference is only a within-pair
contrast while the two sides share an architecture and a tokenizer, so both are
asserted from the live configs and vocabularies. ``progen2-small-mix7`` adds seven
``<|pfXXXXX|>`` family tokens above its base's shared entries; the record names
them, because a declared extension and an undetected mismatch must not look alike.

Usage
=====

``--arm`` repeats, and defaults to every staged free-length arm. ``--attempts``
is a small smoke draw, not a measurement: the record it writes carries
``is_scientific_measurement: false`` and no attempt ledger. ``--no-generate``
checks the tokenizer contracts and the pair match without loading weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402

SCHEMA = "objective_tuned_interface_v1"
RECORD = "objective_tuned_interface.json"


#: A tokenizer file, in any of the spellings the arms here ship. Checked before
#: the load, because ``AutoTokenizer`` on a directory that has none falls back to
#: a slow sentencepiece tokenizer and raises ``TypeError: not a string``, which
#: names neither the arm nor the missing file. Observed on a half-staged
#: ProtGPT3-10B-dpo whose weights and ``tokenizer.json`` had not yet arrived.
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json")


def load_tokenizer(arm: str) -> Any:
    """The arm's tokenizer, exactly as its loader builds it.

    Built the way ``load_arm_spec`` builds it -- same path, same
    ``trust_remote_code`` decision, same pad fallback -- so that what is verified
    here is the object generation will use, not a second construction of it.
    """

    from transformers import AutoTokenizer

    from src.capability.core.arms import _BUILTIN_CAUSAL_LM_ARCHITECTURES

    spec = ot.spec_for(arm)
    if not spec.path.is_dir():
        raise SystemExit(f"{arm}: no checkpoint directory at {spec.path}")
    present = {item.name for item in spec.path.iterdir() if item.is_file()}
    if not present.intersection(_TOKENIZER_FILES):
        raise SystemExit(
            f"{arm}: {spec.path} holds none of {list(_TOKENIZER_FILES)}, so its "
            "staging is incomplete and the prompt contract cannot be checked"
        )
    trust = spec.architecture not in _BUILTIN_CAUSAL_LM_ARCHITECTURES
    tokenizer = AutoTokenizer.from_pretrained(str(spec.path), trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def verify_tokenizer(arm: str) -> dict[str, Any]:
    tokenizer = load_tokenizer(arm)
    prompt = ug.prompt_for(arm)
    out = {
        "prompt_contract": ot.prompt_contract(arm, tokenizer, prompt),
        "residue_alphabet": ot.residue_alphabet_contract(arm, tokenizer),
        "close_token": ug.close_token_for(arm),
        "close_token_id": int(tokenizer.convert_tokens_to_ids(ug.close_token_for(arm))),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_length": len(tokenizer),
        "special_ids": sorted(int(value) for value in tokenizer.all_special_ids),
    }
    doubled = _doubled_bos_ids(arm, tokenizer, prompt)
    if doubled is not None:
        out["refused_variant_doubled_bos"] = doubled
    return out


def _doubled_bos_ids(arm: str, tokenizer: Any, prompt: str) -> list[int] | None:
    """The id sequence the published-recipe keyword would add to the lane's prompt.

    Recorded as evidence that the contract check has something to catch: it is the
    out-of-distribution prompt, so it is encoded and written down, never sampled.
    """

    if ot.spec_for(arm).input_format != ot.INPUT_FORMAT_BOS_DIRECTION_SEQ:
        return None
    from transformers import AutoTokenizer

    with_bos = AutoTokenizer.from_pretrained(
        str(ot.spec_for(arm).path), add_bos_token=True, padding_side="left"
    )
    return [int(value) for value in with_bos(prompt)["input_ids"]]


def verify_generation(arm: str, *, device: str, attempts: int) -> dict[str, Any]:
    """A smoke draw through the lane's own path, timed so a campaign can be sized.

    The timing is wall-clock on whatever card this ran on and is a sizing input,
    not a benchmark: it includes the load and shares the card with whatever else
    holds it.
    """

    if not ot.weights_complete(arm):
        raise SystemExit(
            f"{arm}: its weights are not wholly staged, so a draw from it would "
            "fail inside the loader rather than here; verify the tokenizer "
            "contracts with --no-generate until the staging completes"
        )
    start = time.monotonic()
    model, tokenizer = ug.load_generator(
        arm, device=device, checkpoint=ug.checkpoint_for_arm(arm)
    )
    loaded = time.monotonic()
    ot.prompt_contract(arm, tokenizer, ug.prompt_for(arm))
    rows = ug.generate(arm, n=attempts, model=model, tokenizer=tokenizer)
    finished = time.monotonic()
    sequences = [row["sequence"] for row in rows]
    out = ot.require_residue_sequences(arm, sequences)
    out["stop_status"] = {
        status: sum(1 for row in rows if row["stop_status"] == status)
        for status in sorted({row["stop_status"] for row in rows})
    }
    out["examples"] = [sequence[:60] for sequence in sequences[:3]]
    out["device"] = device
    out["load_seconds"] = round(loaded - start, 2)
    out["sample_seconds"] = round(finished - loaded, 2)
    out["seconds_per_attempt"] = round((finished - loaded) / max(attempts, 1), 3)
    out["projected_seconds_per_policy_cell"] = round(
        (finished - loaded) / max(attempts, 1) * int(ug.POLICY["attempts"]), 1
    )
    del model
    _release(device)
    return out


def _release(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.empty_cache()


def verify_pairs(arms: list[str]) -> dict[str, Any]:
    """Architecture and tokenizer equality for every pair both of whose arms are checked."""

    from transformers import AutoConfig

    from src.capability.core.arms import _BUILTIN_CAUSAL_LM_ARCHITECTURES

    out: dict[str, Any] = {}
    for pair in ot.pairs_in_lane(ot.LANE_FREE_LENGTH):
        if not all(name in arms for name in pair.arms):
            continue
        loaded = {}
        for name in pair.arms:
            spec = ot.spec_for(name)
            trust = spec.architecture not in _BUILTIN_CAUSAL_LM_ARCHITECTURES
            loaded[name] = SimpleNamespace(
                config=AutoConfig.from_pretrained(str(spec.path), trust_remote_code=trust),
                tokenizer=load_tokenizer(name),
            )
        out[pair.tuned] = ot.require_matched_pair(pair, loaded)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        help="repeatable; defaults to every staged free-length arm",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attempts",
        type=int,
        default=8,
        help="smoke attempts per arm; not a measurement",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="verify the tokenizer contracts and the pair match without loading weights",
    )
    args = parser.parse_args(argv)

    arms = list(args.arm) if args.arm else list(ot.scheduled_free_length_arms())
    if not arms:
        raise SystemExit(
            "no staged free-length arm to verify; "
            f"staging: {json.dumps(ot.staging_report()['arms'], indent=1)}"
        )
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "campaign": ot.CAMPAIGN,
        "declaration": "docs/objective-alignment.md",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_scientific_measurement": False,
        "generated": not args.no_generate,
        "attempts_per_arm": None if args.no_generate else int(args.attempts),
        "policy": ug.POLICY,
        "staging": ot.staging_report(),
        "arms": {},
        "pairs": {},
    }
    for arm in arms:
        entry: dict[str, Any] = {
            "checkpoint": str(ug.checkpoint_for_arm(arm)),
            "input_format": ot.spec_for(arm).input_format,
            "declared_shape": [ot.spec_for(arm).n_layer, ot.spec_for(arm).d_model],
            "dtype": ug.generation_point(arm).dtype,
            "use_cache": ug.generation_point(arm).use_cache,
            "batch_size": ug.arm_batch_size(arm),
        }
        entry.update(verify_tokenizer(arm))
        if not args.no_generate:
            entry["generation"] = verify_generation(
                arm, device=args.device, attempts=int(args.attempts)
            )
        record["arms"][arm] = entry
        print(f"{arm}: prompt ids {entry['prompt_contract']['prompt_ids']} ok", flush=True)
    record["pairs"] = verify_pairs(arms)
    for tuned, matched in record["pairs"].items():
        print(f"{tuned}: matched to {matched['base']} at {matched['depth_width']}, "
              f"added tokens {matched['tokens_added_by_tuning']}")
    write_json(Path(args.out) / RECORD, record)
    print(Path(args.out) / RECORD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
