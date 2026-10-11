#!/usr/bin/env python3
"""Render E11's frozen products as a pool E17's own scorer and selector can read.

E11 measures an **unselected** generation distribution; E17 measures what
**likelihood selection over** such a distribution concentrates on. Those are
compatible -- generation need not retrieve for selection on likelihood to
retrieve -- and separating them means applying E17's selector to E11's products
rather than building a second one. A second implementation of the same selector
would make any difference between the two results uninterpretable, so nothing
about the selector or the likelihood convention is restated here: this stage only
puts E11's products into the row shape
``scripts/capability/evaluation/score_generated_likelihood.py`` already consumes
(``arm``, ``roles``, ``id``, ``sequence``, ``length``), and that stage and
:mod:`src.capability.evaluation.generated_phenotype` are used unmodified.

**What the pool is, and why.** Every attempt at or above
:data:`~src.capability.context.homology_context.MIN_PRODUCT_RESIDUES` residues
that is not a copy of its own context -- the same denominator the corpus-novelty
census uses, so the identities and the likelihoods line up row for row. The pool
keeps its condition label, because the question is asked within each condition:
whether selection concentrates on corpus near-duplicates is a property of a
condition's own output, and pooling the four would let a difference between
conditions masquerade as a selection effect.

Copies are excluded rather than left in. A copy of a context item is a corpus
entry by construction, so leaving copies in the pool would seed the
near-duplicate share with sequences whose provenance is already known -- the
very thing the endpoint is trying to detect in sequences whose provenance is not.
E11 measured zero copies on all three rules, so the exclusion removes nothing
here; it is declared because a future arm with a nonzero copy rate would need it.
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

EXPECT = "product_pool.json"
POOL = "product_pool.jsonl"


def run(args: argparse.Namespace) -> None:
    products = [
        json.loads(line)
        for line in args.products.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not products:
        raise SystemExit(f"{args.products} holds no product")

    annotations: dict[str, dict] = {}
    if args.annotations is not None:
        record = json.loads(args.annotations.read_text(encoding="utf-8"))
        H.require_declaration(record, scope="context")
        annotations = {
            row["attempt_id"]: row for row in record["annotations"] if row.get("attempt_id")
        }

    members: list[dict] = []
    excluded = {"short": 0, "copy": 0}
    for row in products:
        if int(row["residues"]) < H.MIN_PRODUCT_RESIDUES:
            excluded["short"] += 1
            continue
        annotation = annotations.get(str(row["attempt_id"]))
        verdict = H.copy_verdict(
            row["copy_statistics"],
            identity_percent=None if annotation is None else annotation.get(
                "context_alignment_identity"
            ),
        )
        if verdict["is_copy"]:
            excluded["copy"] += 1
            continue
        members.append(
            {
                # The scorer's contract.
                "id": str(row["attempt_id"]),
                "arm": str(row["arm"]),
                "roles": ["pool"],
                "sequence": str(row["sequence"]),
                "length": int(row["residues"]),
                # Carried for the selection analysis, ignored by the scorer.
                "condition": str(row["condition"]),
                "target_id": str(row["target_id"]),
                "stop_status": str(row["stop_status"]),
                "copy_verdict": verdict,
            }
        )
    if not members:
        raise SystemExit("the pool is empty; refusing to record a completion")
    arms = sorted({row["arm"] for row in members})
    if len(arms) != 1:
        raise SystemExit(
            f"the products carry arms {arms}; a likelihood is comparable within an "
            "arm and not across arms, so one pool holds one arm"
        )
    members.sort(key=lambda row: row["id"])

    args.out.mkdir(parents=True, exist_ok=True)
    pool_path = args.out / POOL
    pool_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in members), encoding="utf-8"
    )
    conditions = sorted({row["condition"] for row in members})
    write_json(
        args.out / EXPECT,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "freeze_product_pool",
            "status": "complete",
            **H.declaration_digests(),
            "products": str(args.products),
            "products_sha256": sha256_file(args.products),
            "annotations": None if args.annotations is None else str(args.annotations),
            "arm": arms[0],
            "pool": POOL,
            "pool_sha256": sha256_file(pool_path),
            "members": len(members),
            "offered": len(products),
            "excluded": excluded,
            "min_product_residues": H.MIN_PRODUCT_RESIDUES,
            "copy_rule_evaluated": bool(annotations),
            "per_condition": {
                condition: sum(1 for row in members if row["condition"] == condition)
                for condition in conditions
            },
            "conditions": conditions,
            "reuses": (
                "scripts/capability/evaluation/score_generated_likelihood.py and "
                "src.capability.evaluation.generated_phenotype, both unmodified: the "
                "selector and the likelihood convention are E17's, so a difference "
                "between the two results is a difference in the products and not in "
                "the method"
            ),
        },
    )
    print(
        f"pool: {len(members)} members of {len(products)} attempts, "
        f"{excluded} excluded, arm {arms[0]}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="generation_annotation.json, so the alignment-identity copy rule is "
        "evaluated rather than abstaining when the pool is built",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
