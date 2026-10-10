#!/usr/bin/env python3
"""Draw the two structural sets of E11 from the frozen products, and nothing else.

E11 asks whether conditioning generation on homologous sequences changes what the
model produces and whether the products are structurally better. Those are two
questions and they need two draws:

* the **length-matched** draw -- within each declared length band, the same number
  of products from every condition. It estimates what homologous context does at a
  fixed product length, which is the only way predicted confidence can be compared
  at all, because confidence rises steeply with length and completeness.
* the **unmatched** draw -- the same number of products from each condition's own
  generation distribution, no length stratification. It estimates what the
  conditions deliver as they generate, and it confounds homology with length,
  because under an empty context this arm reaches a native terminator in 4% of
  attempts against 35% under a close homologue. That difference is a finding, not
  a nuisance, and matching it away would hide it.

Both rules were declared before any structure was predicted --
:data:`~src.capability.context.homology_context.STRUCTURE_COMPARISON_RULE` and
:data:`~src.capability.context.homology_context.STRUCTURE_UNMATCHED_RULE` -- and
this stage only applies them. It reads the frozen ``products.jsonl``, never
rewrites it, and re-derives the length-matched draw rather than trusting the
``structure_selected`` flags already in it: the two must agree, and if they do not
this stage refuses rather than folding a set nobody declared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.context import homology_context as H  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402

EXPECT = "structure_draw.json"
SETS: dict[str, str] = {
    "length_matched": "length_matched_products.jsonl",
    "unmatched": "unmatched_products.jsonl",
}


def read_products(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} holds no product")
    return rows


def write_set(path: Path, rows: list[dict], *, name: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps({**row, "structure_selected": True, "structure_set": name}) + "\n"
            )


def run(args: argparse.Namespace) -> None:
    products = read_products(args.products)
    present = sorted({str(row["condition"]) for row in products})
    if present != sorted(H.GENERATION_CONDITIONS):
        raise SystemExit(
            f"{args.products} carries conditions {present}, but this experiment "
            f"declares {sorted(H.GENERATION_CONDITIONS)}; a draw over a different "
            "condition set is a different experiment"
        )
    # The **declared** order, not the alphabetical one. Each (band, condition)
    # cell consumes draws from one generator in the order the conditions are
    # visited, so the order is part of the draw: visiting them alphabetically
    # reproduced only 11 of the 128 attempts the generation stage selected. The
    # reproduction check below is what caught that, and this is the order it
    # checks against.
    conditions = list(H.GENERATION_CONDITIONS)

    matched, matched_record = H.select_structure_products(products, conditions=conditions)
    frozen = {str(row["attempt_id"]) for row in products if row.get("structure_selected")}
    if frozen and frozen != matched:
        raise SystemExit(
            "the length-matched draw re-derived here is not the one frozen into "
            f"{args.products}: {len(matched - frozen)} attempts appear only in the "
            f"re-derivation and {len(frozen - matched)} only in the file. The draw "
            "rule or its seed has changed, and folding either set would produce a "
            "comparison no declaration describes"
        )
    unmatched, unmatched_record = H.select_unmatched_structure_products(
        products, conditions=conditions
    )

    args.out.mkdir(parents=True, exist_ok=True)
    chosen = {"length_matched": matched, "unmatched": unmatched}
    written: dict[str, dict] = {}
    for name, identifiers in chosen.items():
        rows = [row for row in products if str(row["attempt_id"]) in identifiers]
        if len(rows) != len(identifiers):
            raise SystemExit(f"the {name} draw names attempts the product file does not hold")
        rows.sort(key=lambda row: str(row["attempt_id"]))
        path = args.out / SETS[name]
        write_set(path, rows, name=name)
        written[name] = {
            "file": SETS[name],
            "sha256": sha256_file(path),
            "products": len(rows),
            "per_condition": {
                condition: sum(1 for row in rows if str(row["condition"]) == condition)
                for condition in conditions
            },
            "targets": len({str(row["target_id"]) for row in rows}),
            "native_terminal": sum(1 for row in rows if row.get("stop_status") == "native_terminal"),
        }

    write_json(
        args.out / EXPECT,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "draw_structure_sets",
            "status": "complete",
            **H.declaration_digests(),
            "structure_extension_sha256": H.structure_extension_digest(),
            "structure_extension": H.structure_extension(),
            "products": str(args.products),
            "products_sha256": sha256_file(args.products),
            "conditions": conditions,
            "length_matched": matched_record,
            "unmatched": unmatched_record,
            "reproduced_frozen_selection": bool(frozen) and frozen == matched,
            "sets": written,
            "overlap": sorted(matched & unmatched),
            "overlap_note": (
                "an attempt drawn into both sets is folded once per set; the two folds "
                "are separate draws of the instrument's own sampler and are never "
                "averaged together"
            ),
            "runtime": {"interpreter": sys.executable},
        },
    )
    print(
        "drew "
        + "; ".join(
            f"{name}: {record['products']} products, {record['per_condition']}"
            for name, record in written.items()
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--products", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
