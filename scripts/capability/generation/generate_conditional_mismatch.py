#!/usr/bin/env python3
"""Generate the conditional cell the replication campaign never sampled: mismatched.

``configs/generation_replication_manifest.json`` declares ``zymctrl__requested``
and ``prollama__requested`` for both new campaign seeds and **no mismatched
cell**. The retained 2026-09-05 stream is therefore the only stream in which the
requested-minus-mismatched difference -- the one quantity that separates a label
that selects the requested class from a decoder that emits one family whatever it
is asked for -- can be formed at all.

This stage samples the missing side. For target class ``c`` it prompts the arm
with ``c``'s **donor** label from the frozen class queue
(``results/R6/conditioned_generation_queue_20260826/class_queue.json``, whose
digest is verified before it is read) and records the attempts against ``c``, so
the cell is scored for "how often does a generation made under some *other*
request land in ``c``". The pairing is a fixed-point-free permutation in which
every class donates exactly once, which is checked here and not assumed.

Everything except the condition follows the manifest cell verbatim: the same
sampling configuration, the same per-class attempt count and the same
``conditioned_generation.cell_seed`` rule, differing only in the condition string
the seed is derived from. ``--temperature``/``--top-p`` exist for the declared
decoding-sensitivity arm and are recorded in the artefact; a run at a
configuration other than the manifest's is labelled as such and is a separate
measurement, never written into the manifest cell's own stream.

No recognition happens here. The oracle is a separate CPU stage, so a GPU is
never held while a 2 GB profile database is scanned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.generation import conditional_contrast as cc  # noqa: E402
from src.capability.generation import conditioned_generation as cg  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "configs/generation_replication_manifest.json"
DEFAULT_QUEUE = REPO_ROOT / "results/R6/conditioned_generation_queue_20260826/class_queue.json"

SCHEMA_VERSION = "d1_conditional_mismatch_cell_v1"
COMPLETION = "conditional_generation_cell.json"


def _load_arm(name: str, *, device: str, dtype: str) -> Any:
    """The declared door this arm's weights come through, and nothing else."""

    spec = cg.arm(name)
    if spec.loader == "panel":
        from src.capability.core.arms import load_arm

        return load_arm(spec.checkpoint, device=device, dtype=dtype)
    if spec.loader == "lineage":
        from src.capability.models.joint_lineage import load_rung

        # The lineage is keyed by rung name, not by directory name, and the only
        # conditioned rung is stage 2: stage 1 has no instruction interface.
        return load_rung(name, device=device, dtype=dtype)
    raise ValueError(f"{name}: unknown loader {spec.loader!r}")


#: The conditioned protein arms this stage may sample. The text positive-control
#: arms of EXP-R2-227 carry a script oracle rather than a family oracle and are
#: not part of this endpoint.
CONDITIONED_PROTEIN_ARMS: tuple[str, ...] = tuple(
    sorted(name for name, spec in cg.ARMS.items() if spec.conditioned and spec.modality == "protein")
)


def _resources(tag: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "tag": tag,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            record["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "free_mib": free // (1 << 20),
                "total_mib": total // (1 << 20),
            }
    except Exception as error:  # pragma: no cover - a receipt must never fail a run
        record["torch_error"] = repr(error)
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    queue = cg.load_queue(args.queue)
    pairing = cc.pairing_from_queue(queue, args.arm)
    balance = cc.verify_pairing_balance(pairing)

    campaigns = {block["id"]: int(block["seed"]) for block in manifest["campaigns"]}
    if args.campaign not in campaigns:
        raise SystemExit(f"unknown campaign {args.campaign!r}; declared: {sorted(campaigns)}")
    campaign_seed = campaigns[args.campaign]
    cell_name = f"{args.arm}__requested"
    declared = next((cell for cell in manifest["cells"] if cell["cell"] == cell_name), None)
    if declared is None:
        raise SystemExit(f"the manifest declares no cell {cell_name!r} to mirror")
    classes = [str(block["class_key"]) for block in declared["classes"]]
    if sorted(classes) != sorted(pairing):
        raise SystemExit(
            f"the manifest cell's classes and the frozen queue disagree for {args.arm}; "
            "a mismatched cell must be paired by the same frozen permutation the "
            "requested cell was drawn under"
        )

    attempts_per_class = args.attempts_per_class or int(declared["attempts_per_class"])
    decoding = {
        "temperature": args.temperature if args.temperature is not None else float(declared["temperature"]),
        "top_p": args.top_p if args.top_p is not None else float(declared["top_p"]),
        "top_k": args.top_k if args.top_k is not None else int(declared["top_k"]),
        "max_new_tokens": int(declared["max_new_tokens"]),
        "batch_size": args.batch_size or int(declared["batch_size"]),
        "dtype": args.dtype or str(declared["dtype"]),
        "use_cache": bool(declared["use_cache"]),
        "add_special_tokens": bool(declared["add_special_tokens"]),
        "repetition_penalty": float(declared["repetition_penalty"]),
    }
    manifest_decoding = all(
        decoding[field] == declared[field] for field in ("temperature", "top_p", "top_k")
    )
    recipe = args.recipe or ("manifest" if manifest_decoding else "sensitivity")
    if recipe == "manifest" and not manifest_decoding:
        raise SystemExit(
            "--recipe manifest was requested but the decoding differs from the "
            f"manifest cell: {decoding}. A run at another configuration is a separate "
            "measurement and must carry its own --recipe label"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    start = _resources("start")
    write_json(args.out / "runtime-start.json", start)

    spec = cg.arm(args.arm)
    handle = _load_arm(args.arm, device=args.device, dtype=decoding["dtype"])
    delimiter = cg.end_delimiter_for(handle, spec)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    labels = {str(block["class_key"]): block["label"] for block in declared["classes"]}
    for class_key in classes:
        # The oracle always scores against ``class_key``. What changes between the two
        # conditions is which label the prompt carries: the class's own under
        # ``requested``, and its frozen donor's under ``mismatched``.
        donor = pairing[class_key]
        prompt_class = class_key if args.condition == "requested" else donor
        label = labels[prompt_class]
        prompt = cg.prompt_for(handle, spec, label)
        seed = cg.cell_seed(
            seed=campaign_seed, arm_name=args.arm, class_key=class_key, condition=args.condition
        )
        texts = cg.sample_continuations(
            handle.model,
            handle.tokenizer,
            prompt,
            n=attempts_per_class,
            seed=seed,
            batch_size=decoding["batch_size"],
            max_new_tokens=decoding["max_new_tokens"],
            temperature=decoding["temperature"],
            top_p=decoding["top_p"],
            top_k=decoding["top_k"],
            use_cache=decoding["use_cache"],
            add_special_tokens=decoding["add_special_tokens"],
        )
        for index, text in enumerate(texts):
            sequence = cg.extract_protein(text, end_delimiter=delimiter)
            material = f"{args.campaign}|{args.arm}|{class_key}|{args.condition}|{recipe}|{index}"
            rows.append(
                {
                    "id": "gm_" + hashlib.blake2b(material.encode(), digest_size=12).hexdigest(),
                    "schema_version": SCHEMA_VERSION,
                    "arm": args.arm,
                    "campaign": args.campaign,
                    "campaign_seed": campaign_seed,
                    "decoding_recipe": recipe,
                    "class_key": class_key,
                    "native_prompt_class": prompt_class,
                    "native_prompt_label": label,
                    "condition": args.condition,
                    "role": "generation",
                    "primary_class": True,
                    "source_label": f"{args.arm}__{args.condition}",
                    "source_key": f"{args.campaign}|{args.arm}__{args.condition}|{class_key}",
                    "source_sample_index": index,
                    "cell_seed": seed,
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "length": len(sequence),
                    "valid_aa20": bool(sequence) and set(sequence) <= set(cg.AA20),
                    "native_delimiter_observed": delimiter in text,
                    "raw_continuation_length_chars": len(text),
                }
            )
        per_class.append(
            {
                "class_key": class_key,
                "donor_class": donor,
                "prompt_class": prompt_class,
                "prompt_label": label,
                "prompt": prompt,
                "cell_seed": seed,
                "n_attempts": attempts_per_class,
                "n_empty": sum(
                    1 for row in rows[-attempts_per_class:] if not row["sequence"]
                ),
            }
        )

    ledger = args.out / "attempts.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    end = _resources("end")
    write_json(args.out / "runtime-end.json", end)

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "arm": args.arm,
        "campaign": args.campaign,
        "campaign_seed": campaign_seed,
        "condition": args.condition,
        "decoding_recipe": recipe,
        "decoding": decoding,
        "manifest_decoding": manifest_decoding,
        "mirrored_cell": cell_name,
        "attempts_per_class": attempts_per_class,
        "n_classes": len(classes),
        "n_attempts": len(rows),
        "n_empty_attempts": sum(1 for row in rows if not row["sequence"]),
        "n_distinct_sequences": len({row["sequence_sha256"] for row in rows}),
        "pairing_balance": balance,
        "queue_digest": queue["digest"],
        "per_class": per_class,
        "attempts_jsonl": str(ledger),
        "attempts_sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "serving_provenance": getattr(handle, "serving_provenance", None),
        "runtime_start": start,
        "runtime_end": end,
        "recognition": (
            "no family recognition is performed here; run "
            "recognise_generated_families.py on attempts.jsonl"
        ),
        "ceiling": dict(cc.CEILING),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=list(CONDITIONED_PROTEIN_ARMS))
    parser.add_argument("--campaign", required=True)
    parser.add_argument(
        "--condition", default="mismatched", choices=list(cc.CONDITIONS),
        help="mismatched is the missing side; requested is available so a sensitivity recipe can sample both",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--attempts-per-class", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", default=None, choices=(None, "bfloat16", "float16", "float32"))
    parser.add_argument(
        "--recipe", default=None,
        help="decoding-recipe label; 'manifest' is refused unless the decoding equals the manifest cell's",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "arm", "campaign", "condition",
                                                   "decoding_recipe", "n_attempts",
                                                   "n_distinct_sequences")}, sort_keys=True))


if __name__ == "__main__":
    main()
