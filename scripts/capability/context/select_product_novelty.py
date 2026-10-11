#!/usr/bin/env python3
"""Does likelihood selection over E11's products concentrate on corpus near-duplicates?

The experiment two results jointly point at. E11 measures an **unselected**
generation distribution and finds it no closer to the reference corpus under
homologous conditioning than under an empty context -- if anything further. E17
measures what **likelihood selection over** such a distribution concentrates on
and finds an eightfold enrichment of corpus near-duplicates in its top sixty.
Those are compatible: generation need not retrieve for selection on likelihood to
retrieve. If concentration appears here, in a cohort whose generation
distribution shows none, then retrieval is a property of *selection* rather than
of conditioned generation -- which neither experiment can say alone.

**Everything about the method is E17's.** The selector, the selection fractions,
the size-matched random comparator, the near-duplicate threshold and the selected-
set profile are imported from
:mod:`src.capability.evaluation.generated_phenotype`; the likelihoods come from
``scripts/capability/evaluation/score_generated_likelihood.py`` run unmodified on
this pool. A second implementation of the same selector would make any difference
between the two results a difference in implementations, which is the one thing
this comparison cannot afford.

Three things this stage does take care over.

*The convention.* The declared selector is mean negative log-likelihood per
scored **token**, lowest first. Under a subword tokenisation that is a different
ranking from per **residue** -- E17 measured Spearman 0.9764, not 1.0 -- and the
*summed* convention is a length selector at +0.81. So all three are reported with
their rank correlations against each other and against length, and which one the
result is read on is stated rather than assumed.

*The comparator.* A selected set is compared against a **size-matched random draw
from the same pool**, averaged over :data:`generated_phenotype.RANDOM_BASELINE_KEYS`
independent keys with its own interval, not against the pool mean. The pool mean
is random selection's expectation and is reported beside it, but the spread of
finite draws at the same k is what a single selected set has to beat.

*The unit.* The bootstrap resamples wild-type family groups, not products, and
recomputes the selection inside every draw at the same k, so the comparison is of
methods at equal budget rather than of two fixed sequence sets.

Selection is applied **within each condition**, because whether selection
concentrates on corpus near-duplicates is a property of a condition's own output,
and pooling the four conditions would let a difference between them appear as a
selection effect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.context import homology_context as H  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

EXPECT = "product_selection_novelty.json"

#: The selector the result is read on, and the two reported beside it.
PRIMARY_SELECTOR = "mean_nll_per_token_nats"
SELECTORS = (PRIMARY_SELECTOR, "mean_nll_per_residue_nats", "nll_sum_nats")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rho(left, right) -> float | None:
    value = spearmanr(np.asarray(left, float), np.asarray(right, float)).statistic
    return None if not np.isfinite(value) else float(value)


def run(args: argparse.Namespace) -> None:
    pool_record = json.loads(args.pool.read_text(encoding="utf-8"))
    H.require_declaration(pool_record, scope="context")
    members = read_jsonl(args.pool.parent / pool_record["pool"])
    likelihood_record = json.loads(args.likelihood.read_text(encoding="utf-8"))
    likelihood = {
        row["id"]: row for row in read_jsonl(args.likelihood.parent / "generated_likelihood.jsonl")
    }
    annotation_record = json.loads(args.annotations.read_text(encoding="utf-8"))
    H.require_declaration(annotation_record, scope="context")
    corpus = {
        row["attempt_id"]: row
        for row in annotation_record["annotations"]
        if row.get("attempt_id")
    }
    if annotation_record.get("corpus_search") is None:
        raise SystemExit(
            f"{args.annotations} carries no corpus search, so its products have no "
            "corpus identity and this endpoint has nothing to read"
        )

    absent = [row["id"] for row in members if row["id"] not in likelihood]
    if absent:
        raise SystemExit(
            f"{len(absent)} pool members carry no likelihood (first {absent[:3]}); the "
            "selector is incomplete and no selection is reported"
        )
    unscored = [row["id"] for row in members if row["id"] not in corpus]
    if unscored:
        raise SystemExit(
            f"{len(unscored)} pool members carry no corpus annotation (first "
            f"{unscored[:3]}); the pool and the novelty search are from different runs"
        )

    conditions = sorted({row["condition"] for row in members})
    out: dict[str, dict] = {}
    for condition in [*conditions, "all_conditions"]:
        rows = (
            members
            if condition == "all_conditions"
            else [row for row in members if row["condition"] == condition]
        )
        if len(rows) < 2:
            out[condition] = {"status": "no pool", "members": len(rows)}
            continue
        scores = {
            name: np.asarray([float(likelihood[row["id"]][name]) for row in rows], float)
            for name in SELECTORS
        }
        lengths = np.asarray([int(row["length"]) for row in rows], float)
        tokens = np.asarray([int(likelihood[row["id"]]["scored_tokens"]) for row in rows], float)
        identity = np.asarray(
            [float(corpus[row["id"]]["corpus_max_identity"] or 0.0) for row in rows], float
        )
        # The evaluator the selection is read on: is this product a
        # near-duplicate of a corpus entry? A product with no detectable relative
        # counts as zero identity, which is the reading the declared bands assign
        # on. The threshold is E17's, which is the novelty search's own stratum
        # edge, so the two experiments read one number.
        near_duplicate = (identity >= gp.NEAR_DUPLICATE_IDENTITY).astype(float)
        groups = [row["target_id"] for row in rows]

        fractions: dict[str, dict] = {}
        for fraction in gp.SELECTION_FRACTIONS:
            take = gp.selection_count(len(rows), fraction)
            chosen = gp.selection_indices(scores[PRIMARY_SELECTOR], fraction=fraction)
            random = gp.random_baseline(
                near_duplicate, fraction=fraction, seed=args.seed + 991
            )
            contrast = gp.selection_contrast(
                near_duplicate,
                scores[PRIMARY_SELECTOR],
                groups,
                fraction=fraction,
                seed=args.seed,
                n_bootstrap=args.resamples,
            )
            selected_share = float(near_duplicate[chosen].mean())
            fractions[f"top_{fraction:g}"] = {
                "fraction": float(fraction),
                "n_selected": int(take),
                "near_duplicate_share_selected": selected_share,
                "near_duplicate_share_random": random["mean"],
                "near_duplicate_share_random_interval": random["interval"],
                "near_duplicate_share_pool_mean": random["pool_mean"],
                "random_keys": random["n_keys"],
                "enrichment_vs_random": (
                    None if random["mean"] <= 0.0 else selected_share / random["mean"]
                ),
                "enrichment_note": (
                    "the ratio is undefined when no random draw of this size contains a "
                    "near-duplicate, which is a floor on the pool and not an infinite "
                    "enrichment"
                ),
                "difference_vs_random": contrast.get("difference_vs_random"),
                "difference_vs_random_ci95": contrast.get("difference_vs_random_ci95"),
                "excludes_zero": contrast.get("excludes_zero"),
                "bootstrap_groups": contrast.get("n_groups"),
                "unit_floor": contrast["unit_floor"],
                "mean_identity_selected": float(identity[chosen].mean()),
                "max_identity_selected": float(identity[chosen].max()),
                "mean_identity_pool": float(identity.mean()),
                "mean_length_selected": float(lengths[chosen].mean()),
                "mean_length_pool": float(lengths.mean()),
                "profile_selected": gp.selected_set_profile(
                    [rows[index]["sequence"] for index in chosen],
                    [() for _ in chosen],
                    corpus_identity=identity[chosen].tolist(),
                ),
            }
        out[condition] = {
            "status": "estimated",
            "members": len(rows),
            "groups": len(set(groups)),
            "with_detectable_homology": int((identity > 0.0).sum()),
            "near_duplicates_in_pool": int(near_duplicate.sum()),
            "near_duplicate_share_pool": float(near_duplicate.mean()),
            "max_identity_pool": float(identity.max()),
            "selector_convention": {
                "declared": f"{PRIMARY_SELECTOR}, lowest score taken first",
                "per_token_vs_per_residue_spearman": rho(
                    scores["mean_nll_per_token_nats"], scores["mean_nll_per_residue_nats"]
                ),
                "per_token_equals_per_residue": bool(
                    np.allclose(
                        scores["mean_nll_per_token_nats"], scores["mean_nll_per_residue_nats"]
                    )
                ),
                "mean_scored_tokens_per_residue": float((tokens / lengths).mean()),
                "per_token_vs_length_spearman": rho(scores[PRIMARY_SELECTOR], lengths),
                "per_residue_vs_length_spearman": rho(
                    scores["mean_nll_per_residue_nats"], lengths
                ),
                "summed_vs_length_spearman": rho(scores["nll_sum_nats"], lengths),
                "note": (
                    "a selector whose rank correlation with length is large is a length "
                    "selector wearing a likelihood's name; the summed convention is "
                    "reported for exactly that reason and is not the declared one"
                ),
            },
            "fractions": fractions,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / EXPECT,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "select_product_novelty",
            "status": "complete",
            **H.declaration_digests(),
            "pool": str(args.pool),
            "pool_sha256": sha256_file(args.pool),
            "likelihood": str(args.likelihood),
            "likelihood_sha256": sha256_file(args.likelihood),
            "likelihood_arm": likelihood_record.get("arm"),
            "annotations": str(args.annotations),
            "annotations_sha256": sha256_file(args.annotations),
            "corpus_search": annotation_record.get("corpus_search"),
            "corpus_database": annotation_record.get("corpus_database"),
            "near_duplicate_identity_percent": float(gp.NEAR_DUPLICATE_IDENTITY),
            "selection_fractions": list(gp.SELECTION_FRACTIONS),
            "random_baseline_keys": int(gp.RANDOM_BASELINE_KEYS),
            "seed": args.seed,
            "resamples": args.resamples,
            "reuses": (
                "src.capability.evaluation.generated_phenotype (selector, selection "
                "fractions, size-matched random baseline, near-duplicate threshold, "
                "selected-set profile) and scripts/capability/evaluation/"
                "score_generated_likelihood.py, both unmodified"
            ),
            "estimand": (
                "the near-duplicate share of the top k of one condition's own products "
                "by the model's own likelihood, against a size-matched random draw from "
                "the same condition's pool. It separates retrieval by selection from "
                "retrieval by generation; it is not a statement about any condition's "
                "generation distribution, which the census measures"
            ),
            "limitations": [
                "the near-duplicate threshold is an identity over the query against one "
                "corpus snapshot at e <= 1e-3; no detected relative is not absence from "
                "any model's pretraining corpus",
                "a likelihood is comparable within an arm and not across arms, so this "
                "result does not transfer to another generating checkpoint",
                "no product carries a Pfam family call in this cohort, so the selected-"
                "set profile's repertoire fields are structurally empty rather than low",
            ],
            "per_condition": out,
        },
    )
    for condition, cell in out.items():
        if cell.get("status") != "estimated":
            continue
        top = cell["fractions"]["top_0.05"]
        print(
            f"{condition}: pool {cell['members']} near-duplicates {cell['near_duplicates_in_pool']} "
            f"| top-5% selected {top['near_duplicate_share_selected']:.4f} vs random "
            f"{top['near_duplicate_share_random']:.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool", type=Path, required=True, help="product_pool.json")
    parser.add_argument("--likelihood", type=Path, required=True, help="generated_likelihood.json")
    parser.add_argument("--annotations", type=Path, required=True, help="generation_annotation.json")
    parser.add_argument("--seed", type=int, default=H.DRAW_SEED)
    parser.add_argument("--resamples", type=int, default=H.BOOTSTRAP_DRAWS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
