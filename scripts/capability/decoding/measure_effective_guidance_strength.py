#!/usr/bin/env python3
"""E20: how far the guidance bias actually moved each model's sampling distribution.

For every already-generated sequence the token path is teacher-forced through the
model that produced it, and at each step the divergence between the native and the
guided sampling distributions over the admissible tokens is measured
(:mod:`src.capability.decoding.guidance_strength` owns the math; the sampler in
:mod:`src.capability.decoding.guided_decoding` feeds the same accumulator while
decoding).

The comparable statistic across models is ``kl_per_residue``: the summed per-step
KL divided by the sequence length. Per-step statistics depend on how many residues
a token carries (ProGen2 one, ProtGPT2 several) and are within-model only.

Segmentation caveat
===================

A record that carries its drawn ``token_ids`` is replayed on exactly that path
(``segmentation = "recorded_token_ids"``). A record that holds only the residue
string is re-tokenised canonically. For a single-character tokeniser (ProGen2) that
is the sampled path (``"exact"``). For BPE (ProtGPT2) the canonical segmentation can
differ from the sampled one, and the entry is flagged
``"canonical_reapproximated"``: it measures the strength along an equivalent path.

Gauge
=====

``--gauge`` is required. ``raw`` uses the Potts parameters exactly as stored;
``zero-sum`` centres ``J`` and moves the row-mean sum into ``h``
(:func:`src.capability.context.potts_gauge.to_zero_sum`). A strength diagnostic
must use the gauge the sequences were generated in.

The campaign queue injects ``--device`` and ``--out`` ahead of the other
arguments, so the parser is flat. Every asset is an explicit option, validated
before anything is loaded; no path is derived from this file's location.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context.family_potts import FamilyModel  # noqa: E402
from src.capability.context.potts_gauge import to_zero_sum  # noqa: E402
from src.capability.core.arms import output_logit_width  # noqa: E402
from src.capability.decoding.guidance_strength import summarise_records  # noqa: E402
from src.capability.decoding.guided_decoding import (  # noqa: E402
    GuidedDecoder, build_arm_bias, canonical_token_ids, native_layout, replay_strength,
    select_cross_domain_donor, token_span_table)
from src.capability.readouts.readout_extraction import load_readout_arm  # noqa: E402

SCHEMA = "effective_guidance_strength_v1"
OUTPUT_FILENAME = "effective_guidance_strength.json"
GAUGES = ("raw", "zero-sum")
REQUIRED_FIELDS = ("id", "model", "domain", "guidance_arm", "lambda_fields", "lambda_coupling",
                   "strength_index", "stream_index", "sequence")
COMPARABLE = "kl_per_residue"


def _load_script(name: str) -> Any:
    path = ROOT / "scripts/capability" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_assets(args: argparse.Namespace) -> None:
    """Refuse a missing asset by option name before any weight or fit is loaded."""

    for option, kind in (("records", "file"), ("protocol", "file"), ("references", "dir")):
        path = getattr(args, option)
        ok = path.is_file() if kind == "file" else path.is_dir()
        if not ok:
            raise SystemExit(f"--{option} {path} is not an existing {kind}; stage it and pass the path")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise SystemExit(f"--shard {args.shard} must lie in 0..{args.shards - 1} for --shards {args.shards}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl" else json.loads(text))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"--records {path} holds no records")
    for row in rows:
        missing = [key for key in REQUIRED_FIELDS if key not in row]
        if missing:
            raise ValueError(f"record {row.get('id')!r} lacks {missing}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("record ids are not unique")
    return sorted(rows, key=lambda row: row["id"])


def select_shard(records: Sequence[dict], *, limit: int | None, shard: int, shards: int) -> list[dict]:
    """The first ``limit`` records by id, then every ``shards``-th from ``shard``."""

    return list(records[:limit])[shard::shards]


def apply_gauge(model: FamilyModel, gauge: str) -> FamilyModel:
    if gauge == "raw":
        return model
    if gauge != "zero-sum":
        raise ValueError(f"unknown gauge {gauge!r}; expected one of {list(GAUGES)}")
    h, J = to_zero_sum(model.h, model.J)
    return replace(model, h=h, J=J, sidecar={**model.sidecar, "gauge_convention": "zero-sum"})


def measure_record(arm: Any, decoder: GuidedDecoder, record: Mapping[str, Any]) -> dict[str, Any]:
    """One record's entry: its identity, how its path was obtained, and its strength."""

    sequence = record["sequence"]
    if record.get("token_ids"):
        ids, segmentation = [int(value) for value in record["token_ids"]], "recorded_token_ids"
        if arm.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) \
                != decoder.layout.expected_text(sequence):
            raise ValueError(f"{record['id']}: recorded token_ids do not decode to the sequence")
    else:
        ids, segmentation = canonical_token_ids(arm, decoder, sequence)
    return {
        "id": record["id"], "model": record["model"], "domain": record["domain"],
        "arm": record["guidance_arm"], "lambda_fields": record["lambda_fields"],
        "lambda_coupling": record["lambda_coupling"], "length": len(sequence),
        "segmentation": segmentation,
        "strength": replay_strength(arm=arm, decoder=decoder, token_ids=ids),
    }


def aggregate_cells(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per (model, domain, arm, lambda_fields, lambda_coupling) cell, over its sequences."""

    cells: dict[tuple, list[Mapping[str, Any]]] = {}
    for entry in entries:
        key = (entry["model"], entry["domain"], entry["arm"],
               entry["lambda_fields"], entry["lambda_coupling"])
        cells.setdefault(key, []).append(entry)
    return [
        {"model": key[0], "domain": key[1], "arm": key[2], "lambda_fields": key[3],
         "lambda_coupling": key[4], "segmentation": dict(Counter(e["segmentation"] for e in rows)),
         **summarise_records([e["strength"] for e in rows])}
        for key, rows in sorted(cells.items())
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    collect = _load_script("decoding/collect_guided_decoding_pool.py")
    records_bytes, protocol_bytes = args.records.read_bytes(), args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    selected = select_shard(load_records(args.records), limit=args.limit,
                            shard=args.shard, shards=args.shards)
    if not selected:
        raise ValueError("this shard holds no records")

    device = str(args.device)
    if device != "cpu" and not device.startswith("cuda"):
        raise SystemExit(f"--device must be cpu or cuda:N, got {device!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available on this host")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    first = selected[0]
    declared = collect.validate_protocol(
        protocol, model_arm=first["model"], domain=first["domain"], guidance_arm=first["guidance_arm"],
        strength_index=first["strength_index"], stream_index=first["stream_index"])["domains"]
    reference_dir = collect.resolve_reference_dir(args.references, declared)
    models = {name: apply_gauge(model, args.gauge)
              for name, model in collect.load_declared_models(reference_dir, declared).items()}
    qualified = {name: model.length for name, model in models.items() if model.qualified}

    stage46 = _load_script("stages/context_homologue.py")

    by_model: dict[str, dict[tuple, list[dict]]] = {}
    for record in selected:
        key = (record["domain"], record["guidance_arm"], record["strength_index"],
               record["stream_index"], record["lambda_fields"], record["lambda_coupling"])
        by_model.setdefault(record["model"], {}).setdefault(key, []).append(record)

    entries: list[dict[str, Any]] = []
    for model_name, cells in sorted(by_model.items()):
        arm = load_readout_arm(model_name, stage46, device=device, dtype="float32")
        arm.model.eval()
        arm.model.requires_grad_(False)
        width = int(output_logit_width(arm)["size"])
        for (domain, guidance_arm, strength_index, stream_index, lam_f, lam_c), rows in sorted(cells.items()):
            resolved = collect.validate_protocol(
                protocol, model_arm=model_name, domain=domain, guidance_arm=guidance_arm,
                strength_index=strength_index, stream_index=stream_index)
            if (resolved["strength"].fields, resolved["strength"].coupling) != (lam_f, lam_c):
                raise ValueError(f"{domain}/{guidance_arm}: record strengths differ from the protocol grid")
            target = models[domain]
            donor = (models[select_cross_domain_donor(domain, qualified)]
                     if guidance_arm == "cross_domain_coupling" else None)
            layout = native_layout(arm, target.length)
            decoder = GuidedDecoder(
                bias=build_arm_bias(
                    guidance_arm, target, donor=donor,
                    permutation_seed=resolved["permutation_seed"] if guidance_arm == "permuted_coupling" else None),
                layout=layout, table=token_span_table(arm, layout, logit_width=width),
                strength=resolved["strength"])
            entries.extend(measure_record(arm, decoder, row) for row in rows)
        del arm
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "schema": SCHEMA,
        "comparable_statistic": COMPARABLE,
        "normalisation": "kl_per_residue = sum of per-step KL(guided || native) / residues; "
                         "per-step statistics are within-model only",
        "gauge": args.gauge,
        "shard": {"index": args.shard, "of": args.shards, "limit": args.limit},
        "inputs": {
            "records_sha256": sha256(records_bytes).hexdigest(),
            "protocol_sha256": sha256(protocol_bytes).hexdigest(),
            "references_npz_sha256": {name: model.sidecar["npz_sha256"] for name, model in sorted(models.items())},
        },
        "cells": aggregate_cells(entries),
        "records": entries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True, help="cpu or cuda:N; injected by the campaign queue")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory; injected by the queue")
    parser.add_argument("--records", type=Path, required=True, help="merged E20 per-sequence records (.jsonl or .json list)")
    parser.add_argument("--references", type=Path, required=True, help="directory of qualified Potts npz fits and sidecars")
    parser.add_argument("--protocol", type=Path, required=True, help="the frozen generation protocol json")
    parser.add_argument("--gauge", required=True, choices=GAUGES,
                        help="Potts gauge the sequences were generated in; no default")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N records by id")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_assets(args)
    collect = _load_script("decoding/collect_guided_decoding_pool.py")
    out = collect.prepare_output_dir(args.out)
    result = run(args)
    (out / OUTPUT_FILENAME).write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(out / OUTPUT_FILENAME), "records": len(result["records"]),
                      "cells": len(result["cells"]), "gauge": args.gauge}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
