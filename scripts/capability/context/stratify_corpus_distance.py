#!/usr/bin/env python3
"""E10: predictive increment by distance to a traceable pretraining corpus (CPU).

Three products, in the order they constrain each other:

1. **The traceability enumeration.** Every panel arm classified by what
   ``ArmSpec.pretraining_corpus`` declares, and which of those declarations the
   corpus searched here can stand for. An arm it cannot stand for is refused with
   its reason, never stratified anyway.
2. **The re-band of the prior remote-homology gate.** The groups that gate called
   remote were banded against a staged UniRef50 snapshot; the arms that read
   positive on them declare UniRef90+BFD30. This surfaces how many of those
   groups are still remote with respect to the corpus searched here.
3. **The stratified increment.** For every eligible arm, the frozen out-of-fold
   ``BMPL`` minus ``BPL`` ranking increment read separately on the anchor assays
   whose wild types are near-duplicates of, close to, remote from, or undetectable
   in that corpus -- with each cell's power, and no refit of anything.

Nothing here fits a model, loads a checkpoint or touches a GPU. Loading and
hash-verifying the frozen predictions, and the group bootstrap, belong to the
information-progression panel and are imported from it.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.context import corpus_distance as CD  # noqa: E402
from src.capability.context import homology_context as H  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402

EXPECT = "corpus_distance.json"


def runtime() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "device": "cpu",
        "peak_rss_bytes": usage.ru_maxrss * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "threads_requested": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def run(args: argparse.Namespace) -> None:
    from src.capability.extensions.phenotype_strata import load_production

    contexts = json.loads(args.homologs.read_text(encoding="utf-8"))
    H.require_declaration(contexts, scope="retrieval")
    corpus = contexts["corpus_identity"]
    if args.require_named_release and not corpus.get("named_release"):
        raise SystemExit(
            f"corpus {corpus['corpus_id']!r} carries no published release record, and "
            "E10 stratifies only against a corpus that can be named. Pass "
            "--no-require-named-release to record an object-pinned reading instead."
        )

    started = time.monotonic()
    contract, samples, scores, hashes = load_production(ROOT)
    arms = sorted(contract["models"])
    trace = CD.traceability(arms)
    eligibility = CD.arms_for_corpus(trace, corpus["corpus_id"])

    scored_assays = sorted({row["assay"] for row in scores})
    artifact_assays = sorted({row["assay"] for row in contexts["assays"]})
    if scored_assays != artifact_assays:
        raise SystemExit(
            f"the frozen panel scored {len(scored_assays)} assays and the retrieval "
            f"artefact carries {len(artifact_assays)}; they are not the same support"
        )
    bands, band_support = CD.assay_bands(contexts["targets"], contexts["assays"])
    admitted = band_support["admitted_bands"]
    # A distance *contrast* needs two admitted bands. One band is not a weaker
    # answer to E10's question; it is the finding that this cohort cannot be asked
    # it, and it is reported as that rather than as a one-band panel that looks
    # like a result.
    estimable = len(admitted) >= 2
    saturation = {
        "distance_contrast_estimable": estimable,
        "admitted_bands": admitted,
        "reason": None
        if estimable
        else (
            "every wild type of this cohort retrieves a relative in the searched "
            "corpus above the band edge, so the cohort carries no remote stratum to "
            "contrast against; the increment can be read inside the one admitted band "
            "but no distance comparison exists on this support"
        ),
        "clusters_per_band": band_support["clusters_per_band"],
    }

    panels = {}
    for label, selected in (
        ("primary_declared_corpus_component", eligibility["primary_arms"]),
        ("all_eligible_arms", eligibility["eligible_arms"]),
    ):
        if not selected:
            panels[label] = {"status": "no eligible arm", "arms": []}
            continue
        panel = CD.stratified_panel(
            scores,
            bands=bands,
            arms=selected,
            admitted=admitted,
            draws=args.draws,
            seed=args.seed,
        )
        if panel.get("status") == "estimated":
            panel["monotonicity"] = CD.monotonicity(
                panel["cells"], order=tuple(reversed(CD.DISTANCE_BANDS))
            )
        panel["arms"] = list(selected)
        panels[label] = panel

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / EXPECT,
        {
            "schema_version": CD.SCHEMA_VERSION,
            "stage": "stratify_corpus_distance",
            "status": "complete",
            **H.declaration_digests(),
            "declaration": H.declaration(),
            "question": (
                "does a checkpoint's held-out ranking increment survive on proteins "
                "that have no detectable homologue in the corpus the registry declares "
                "it was trained on?"
            ),
            "corpus_identity": corpus,
            "homologs": str(args.homologs),
            "homologs_sha256": sha256_file(args.homologs),
            "traceability": trace,
            "corpus_eligibility": eligibility,
            "corpus_relations": CD.CORPUS_RELATIONS,
            "unsearchable_corpora": {
                key: {"reason_class": value[0], "reason": value[1]}
                for key, value in CD.UNSEARCHABLE.items()
            },
            "band_support": band_support,
            "corpus_saturation": saturation,
            "assay_bands": bands,
            "prior_gate_reband": contexts.get("remote_gate"),
            "panels": panels,
            "frozen_prediction_source": {
                "panel": "results/extensions/information_progression_20261006",
                "designs": ["BPL", "BMPL"],
                "rows": len(samples),
                "assays": len(scored_assays),
                "families": len({row["cluster"] for row in samples}),
                "models": len(arms),
                "seeds": contract["seeds"],
                "verified": "hash-verified by extensions.phenotype_strata.load_production",
            },
            "does_not_license": (
                "a band is a property of one search against one corpus release. It is "
                "not a statement about any checkpoint's actual training set, no refit "
                "was performed, and the intervals condition on the frozen fitted "
                "predictions"
            ),
            "limitations": list(H.LIMITATIONS)
            + [
                "The corpus searched is a present-day release, not the historical "
                "training snapshot; CORPUS_RELATIONS records which way each arm's bias "
                "runs and whether a remote call is conservative for it.",
                "Unsearched components of a declared mixture (BFD30, ColabFoldDB) mean a "
                "remote call bounds exposure to the searched component only.",
            ],
            "input_sha256": hashes,
            "elapsed_seconds": time.monotonic() - started,
            "runtime": runtime(),
        },
    )
    print(json.dumps(saturation, indent=1), flush=True)
    print(
        f"traceability tiers {trace['tier_counts']}; eligible for "
        f"{corpus['corpus_id']}: {eligibility['eligible_arms']}; admitted bands {admitted}; "
        f"assays per band {band_support['assays_per_band']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--homologs", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=H.BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=H.BOOTSTRAP_SEED)
    parser.add_argument(
        "--no-require-named-release",
        dest="require_named_release",
        action="store_false",
        help="record a reading against a corpus pinned only as an object",
    )
    parser.set_defaults(require_named_release=True)
    args = parser.parse_args()
    if args.device != "cpu":
        raise SystemExit(f"this stage is CPU-only and was given --device {args.device!r}")
    run(args)


if __name__ == "__main__":
    main()
