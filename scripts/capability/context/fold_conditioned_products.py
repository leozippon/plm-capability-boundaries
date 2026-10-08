#!/usr/bin/env python3
"""Predict structure for the length-matched E11 products with ESMFold2.

An instrument and nothing else: it folds the products the generation stage
already marked ``structure_selected`` and records what the model reports. It
chooses no products, excludes none and compares none -- the length-matched,
equal-count selection was declared and recorded upstream in
:mod:`src.capability.context.homology_context`, precisely so that this lane cannot
become a place where a selection rule is quietly decided.

**Runtime.** ESMFold2 needs its own staged interpreter, not the measurement
runtime: the venv at ``runtimes/esmfold2`` carries transformers 5.x, in which
``EsmFold2Model`` exists. That venv also sets ``include-system-site-packages``,
so running it under ``python -I`` silently resolves the *system* transformers
instead, where the class does not exist. This module therefore records the
interpreter and the transformers version it actually ran under, and refuses to
run if the class is missing rather than falling back to another model.

**What a confidence number here is and is not.** ``plddt`` is the model's own
confidence, not measured folding, and it rises with length and completeness: a
truncated 64-residue fragment of a real protein scores far below its full-length
form. Every record therefore carries its residue count, and the comparison that
reads these numbers is the one declared in
:data:`~src.capability.context.homology_context.STRUCTURE_COMPARISON_RULE`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

#: Default staged weights, by the names the launch layer already exports.
WEIGHTS_VARIABLES = ("TRANSFER_ESMFOLD2_DIR", "MODEL_ROOT", "TEXT_MODEL_BASE_DIR")
WEIGHTS_BASENAME = "ESMFold2-hf"

EXPECT = "conditioned_structure.json"
RECORDS = "structure_records.jsonl"


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Atomic, NaN-rejecting write, the same contract as the measurement lanes.

    Spelled here rather than imported: ``src.capability.core.io`` pulls in the
    panel's model interfaces, and this stage runs under a different interpreter
    whose transformers is a different major version.
    """

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_weights(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_dir():
            raise SystemExit(f"{explicit} is not a directory of staged ESMFold2 weights")
        return explicit
    for variable in WEIGHTS_VARIABLES:
        root = os.environ.get(variable)
        if not root:
            continue
        candidate = Path(root) / WEIGHTS_BASENAME
        if candidate.is_dir():
            return candidate
        if variable == "TRANSFER_ESMFOLD2_DIR" and Path(root).is_dir():
            return Path(root)
    raise SystemExit(
        f"no staged {WEIGHTS_BASENAME} found; pass --weights or set one of "
        f"{list(WEIGHTS_VARIABLES)}"
    )


def runtime(device: str) -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "device": device,
        "interpreter": sys.executable,
        "peak_rss_bytes": usage.ru_maxrss * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
    }


def run(args: argparse.Namespace) -> None:
    import torch

    try:
        from transformers import EsmFold2Model
        from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features
    except ImportError as error:  # pragma: no cover - environment contract
        import transformers

        raise SystemExit(
            f"this interpreter ({sys.executable}) resolves transformers "
            f"{transformers.__version__}, which has no EsmFold2Model: {error}. Run the "
            "staged esmfold2 interpreter without -I; with -I it silently resolves the "
            "system transformers instead."
        ) from error
    import transformers

    products = [
        json.loads(line)
        for line in args.products.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not products:
        raise SystemExit(f"{args.products} holds no product")
    selected = [row for row in products if row.get("structure_selected")] if args.only_selected else products
    if not selected:
        raise SystemExit(
            "no product carries structure_selected; the generation stage records the "
            "length-matched selection, and this stage does not invent one"
        )
    selected.sort(key=lambda row: row["attempt_id"])
    if args.limit:
        selected = selected[: args.limit]

    weights = resolve_weights(args.weights)
    device = args.device if args.device.startswith("cuda") else "cpu"
    started = time.monotonic()
    model = EsmFold2Model.from_pretrained(str(weights), dtype=torch.bfloat16).to(device).eval()
    load_seconds = time.monotonic() - started

    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / RECORDS
    failures: list[dict] = []
    written = 0
    folded_started = time.monotonic()
    with records_path.open("w", encoding="utf-8") as handle:
        for number, row in enumerate(selected, start=1):
            sequence = row["sequence"]
            cell_started = time.monotonic()
            try:
                with torch.no_grad():
                    output = model.infer_protein(sequence)
            except Exception as error:  # a product the instrument cannot fold is data
                failures.append(
                    {
                        "attempt_id": row["attempt_id"],
                        "residues": len(sequence),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                # An out-of-memory fold leaves the allocator fragmented, and the
                # next product would fail for that reason rather than its own.
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
            record = {
                "attempt_id": row["attempt_id"],
                "arm": row.get("arm"),
                "target_id": row.get("target_id"),
                "condition": row.get("condition"),
                "residues": len(sequence),
                "plddt": float(output["plddt"].float().mean()),
                "plddt_ca": float(output["plddt_ca"].float().mean())
                if "plddt_ca" in output
                else None,
                "ptm": float(output["ptm"].float().mean()) if "ptm" in output else None,
                "pae": float(output["pae"].float().mean()) if "pae" in output else None,
                "seconds": time.monotonic() - cell_started,
            }
            if args.write_pdb:
                pdb = output_to_pdb(output, prepare_protein_features(sequence, device=device))
                target = args.out / "pdb" / f"{row['attempt_id'].replace('|', '__')}.pdb"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(pdb if isinstance(pdb, str) else pdb[0])
                record["pdb"] = str(target.relative_to(args.out))
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            written += 1
            del output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if number % 20 == 0 or number == len(selected):
                print(f"{number}/{len(selected)} folded ({written} records)", flush=True)

    if written < 1:
        raise RuntimeError(
            "the instrument folded nothing; refusing to record a completion. First "
            f"failures: {failures[:3]}"
        )
    write_json(
        args.out / EXPECT,
        {
            "schema_version": "conditioned_structure_v1",
            "stage": "fold_conditioned_products",
            "status": "complete",
            "instrument": {
                "model": "ESMFold2",
                "weights": str(weights),
                "dtype": "bfloat16",
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "interpreter": sys.executable,
                "load_seconds": load_seconds,
                "note": (
                    "predicted confidence, not measured folding; it rises with product "
                    "length and completeness and is only comparable within a length band"
                ),
            },
            "products": str(args.products),
            "products_sha256": digest(args.products),
            "declaration_sha256": products[0].get("declaration_sha256"),
            "selection": {
                "only_selected": bool(args.only_selected),
                "offered": len(products),
                "attempted": len(selected),
                "folded": written,
                "per_condition": {
                    condition: sum(1 for row in selected if row.get("condition") == condition)
                    for condition in sorted({row.get("condition") for row in selected})
                },
            },
            "failures": failures,
            "records": RECORDS,
            "records_sha256": digest(records_path),
            "elapsed_seconds": time.monotonic() - folded_started,
            "runtime": runtime(device),
        },
    )
    print(f"folded {written} of {len(selected)} products; {len(failures)} failures", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write-pdb", action="store_true")
    parser.add_argument(
        "--all-products",
        dest="only_selected",
        action="store_false",
        help="fold every product rather than the declared length-matched selection",
    )
    parser.set_defaults(only_selected=True)
    main_args = parser.parse_args()
    run(main_args)


if __name__ == "__main__":
    main()
