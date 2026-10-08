#!/usr/bin/env python3
"""Build a position-likelihood cohort for a mutation landscape, label-free.

Three cohorts, one shape. The shape is the one
``scripts/capability/position/extract_position_likelihood.py`` already reads, so
the project's single producer of position-resolved likelihood scores all three
without a second extractor and under one convention:

``singles``
    A seeded single-substitution scan of generated products and of their
    one-to-one length-matched natural partners (E15). Length is matched because
    every downstream-propagation quantity is a mean over ``L - i`` receivers and
    the decay half-distance is of the order of these lengths; an unmatched
    comparison would be a comparison of lengths.

``pairs``
    A four-state design on generated products (E16): contacting residue pairs
    from the folded structure, each matched one-to-one on sequence separation to
    a non-contacting pair of the same protein, with one declared substitution per
    position so that a cycle's single-state terms are literally shared.

``cycles``
    The frozen MegaScale double-mutant cohort rewritten in the same shape (E07),
    so a model's own four-state term can be contrasted at the site pairs the R3
    contact annotation already classified and compared against the measured
    non-additivity that exists there.

No measurement reaches any of it. ``singles`` and ``pairs`` read sequences and
predicted geometry; ``cycles`` reads the frozen cohort's sequences and positions
and drops its ``epsilon`` field, which is the same label-free projection
``pairwise_epistasis.extraction_plan`` makes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.interactions import generated_mutation as gm  # noqa: E402

COMPLETION = "mutation_landscape_cohort.json"
COHORT = "cohort.json"
MODES = ("singles", "pairs", "cycles")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _length_summary(values) -> dict:
    array = np.asarray(sorted(int(value) for value in values), dtype=np.int64)
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "quartiles": [float(value) for value in np.percentile(array, [25, 50, 75])],
    }


# ------------------------------------------------------------------- singles


def build_singles(args) -> dict:
    from src.capability.core.arms import eligible_protein_population

    records = gm.read_generated(args.generated)
    chosen = gm.stratified_subsample(records, per_stage=args.per_stage)
    lengths = [int(row["length"]) for row in chosen]
    floor = max(1, min(lengths) - max(gm.LENGTH_CALIPER_RESIDUES,
                                      int(round(gm.LENGTH_CALIPER_FRACTION * min(lengths)))))
    ceiling = max(lengths) + max(gm.LENGTH_CALIPER_RESIDUES,
                                 int(round(gm.LENGTH_CALIPER_FRACTION * max(lengths))))
    pool, _labels, corpus = eligible_protein_population(floor, ceiling)
    matching = gm.match_natural(chosen, pool)
    matched = {row["matched_to"]: row for row in matching["matched"]}
    paired = [row for row in chosen if row["id"] in matched]

    sequences = [row["sequence"] for row in paired] + [
        matched[row["id"]]["sequence"] for row in paired
    ]
    names, grouping = gm.independence_groups(sequences)
    group_of = dict(zip([row["id"] for row in paired] + [matched[row["id"]]["id"] for row in paired],
                        names))
    # A natural partner is an observation about its generated partner's unit, not
    # an independent one: the contrast is paired, so both members carry the
    # generated member's group.
    for row in paired:
        group_of[matched[row["id"]]["id"]] = group_of[row["id"]]

    assays = []
    for row in paired:
        for origin, sequence, identity in (
            ("generated", row["sequence"], row["id"]),
            ("natural", matched[row["id"]]["sequence"], matched[row["id"]]["id"]),
        ):
            scan = gm.scan_mutations(sequence, sites=args.sites, subs=args.subs)
            assays.append(
                gm.cohort_assay(
                    assay=identity,
                    wildtype=sequence,
                    mutants=[item["label"] for item in scan],
                    sequences=[item["sequence"] for item in scan],
                    cluster=group_of[identity],
                    extra={
                        "origin": origin,
                        "group": group_of[identity],
                        "paired_with": matched[row["id"]]["id"] if origin == "generated" else row["id"],
                        "length": len(sequence),
                        "stage": row["stage"] if origin == "generated" else None,
                        "stream": row["stream"] if origin == "generated" else None,
                        "generator_arm": row["arm"] if origin == "generated" else None,
                        "degenerate": gm.degenerate(row) if origin == "generated" else None,
                        "length_delta": (
                            None if origin == "generated" else int(matched[row["id"]]["length_delta"])
                        ),
                        "sites": [int(item["site"]) for item in scan],
                    },
                )
            )
    design = {
        "mode": "singles",
        "selection_seed": gm.SELECTION_SEED,
        "per_stage": args.per_stage,
        "sites_per_sequence": args.sites,
        "substitutions_per_site": args.subs,
        "natural_corpus": corpus,
        "natural_length_band": [floor, ceiling],
        "length_matching": matching["balance"],
        "unmatched_generated": matching["unmatched"],
        "independence": {key: value for key, value in grouping.items() if key != "detail"},
        "independence_detail": grouping["detail"],
        "generated_products": len(paired),
        "natural_partners": len(paired),
        "degenerate_generated": sum(1 for row in paired if gm.degenerate(row)),
        "generated_lengths": _length_summary(int(row["length"]) for row in paired),
        "natural_lengths": _length_summary(
            matched[row["id"]]["length"] for row in paired
        ),
        "stage_counts": {
            stage: sum(1 for row in paired if row["stage"] == stage)
            for stage in sorted({row["stage"] for row in paired})
        },
        "stream_counts": {
            stream: sum(1 for row in paired if row["stream"] == stream)
            for stream in sorted({row["stream"] for row in paired})
        },
    }
    return {"assays": assays, "design": design}


# --------------------------------------------------------------------- pairs


def build_pairs(args) -> dict:
    records = gm.read_generated(args.generated)
    chosen = gm.stratified_subsample(records, per_stage=args.per_stage)
    assays, refused, structures = [], [], []
    for row in chosen:
        path = Path(args.contacts_dir) / args.contact_file.format(id=row["id"])
        if not path.is_file():
            refused.append({"id": row["id"], "reason": f"no folded structure at {path.name}"})
            continue
        structure = gm.read_predicted_structure(
            path,
            sequence=row["sequence"],
            confidence_key=args.confidence_key,
            confidence_floor=args.confidence_floor,
        )
        pairs = gm.eligible_pairs(structure)
        design = gm.matched_pair_design(
            pairs, row["sequence"], per_stratum=args.pairs_per_stratum
        )
        if not design["pairs"]:
            refused.append(
                {
                    "id": row["id"],
                    "reason": "no separation-matched contact and non-contact pair",
                    "eligible_pairs": design["balance"]["eligible_pairs"],
                    "eligible_contacts": design["balance"]["eligible_contacts"],
                    "admitted_positions": structure["admitted_positions"],
                }
            )
            continue
        states = gm.pair_mutations(row["sequence"], design["pairs"])
        assays.append(
            gm.cohort_assay(
                assay=row["id"],
                wildtype=row["sequence"],
                mutants=states["mutants"],
                sequences=states["sequences"],
                cluster=row["id"],
                extra={
                    "origin": "generated",
                    "length": int(row["length"]),
                    "stage": row["stage"],
                    "stream": row["stream"],
                    "generator_arm": row["arm"],
                    "degenerate": gm.degenerate(row),
                    "cycles": states["cycles"],
                    "substitutions": states["substitutions"],
                    "pair_balance": design["balance"],
                },
            )
        )
        structures.append(
            {
                "id": row["id"],
                "distance_key": structure["distance_key"],
                "confidence_key": structure["confidence_key"],
                "confidence_floor": structure["confidence_floor"],
                "admitted_positions": structure["admitted_positions"],
                "length": structure["length"],
                "sha256": sha256_file(path),
                "balance": design["balance"],
                "dropped_contacts": len(design["dropped_contacts"]),
            }
        )
    if not assays:
        raise SystemExit(
            "no generated product carries a separation-matched contact design; refusing to "
            "write a cohort"
        )
    # Independence over the retained products only, so a group count describes
    # the cohort rather than the draw it came from.
    names, grouping = gm.independence_groups([row["wildtype"] for row in assays])
    for row, name in zip(assays, names):
        row["cluster"] = name
        row["group"] = name
    cycles = [cycle for row in assays for cycle in row["cycles"]]
    design = {
        "mode": "pairs",
        "selection_seed": gm.SELECTION_SEED,
        "per_stage": args.per_stage,
        "pairs_per_stratum": args.pairs_per_stratum,
        "contact_definition": gm.CONTACT_ANGSTROM,
        "min_separation": gm.MIN_SEQUENCE_SEPARATION,
        "separation_caliper": gm.SEPARATION_CALIPER,
        "confidence_key": args.confidence_key,
        "confidence_floor": args.confidence_floor,
        "independence": {key: value for key, value in grouping.items() if key != "detail"},
        "independence_detail": grouping["detail"],
        "products": len(assays),
        "refused_products": refused,
        "structures": structures,
        "cycles": len(cycles),
        "contact_cycles": sum(1 for cycle in cycles if cycle["contact"]),
        "control_cycles": sum(1 for cycle in cycles if not cycle["contact"]),
        "strata": {
            stratum: sum(1 for cycle in cycles if cycle["stratum"] == stratum)
            for stratum in sorted({cycle["stratum"] for cycle in cycles})
        },
        "states": sum(len(row["mutants"]) for row in assays),
        "lengths": _length_summary(row["length"] for row in assays),
    }
    return {"assays": assays, "design": design}


# -------------------------------------------------------------------- cycles


def build_cycles(args) -> dict:
    from src.capability.interactions.pairwise_epistasis import cycle_states

    cohort = json.loads(Path(args.pairwise_cohort).read_text())
    assays, refused = [], []
    cycle_total = 0
    for background in cohort["backgrounds"]:
        wildtype = background["cycles"][0]["sequences"][0]
        if set(wildtype) - set(gm.AA20):
            refused.append({"name": background["name"], "reason": "non-AA20 wild type"})
            continue
        labels: dict[str, str] = {}
        cycles = []
        bad = []
        for cycle in background["cycles"]:
            try:
                states = cycle_states(wildtype, cycle["sequences"])
                names = [gm.state_label(wildtype, state) for state in states[1:]]
            except ValueError as error:
                bad.append({"positions": list(cycle["positions"]), "reason": str(error)})
                continue
            for name, state in zip(names, states[1:]):
                labels[name] = state
            low, high = sorted(int(value) for value in cycle["positions"])
            cycles.append(
                {
                    "i": low,
                    "j": high,
                    "separation": high - low,
                    "stratum": gm.separation_stratum(high - low),
                    "site_pair": f"{background['name']}:{low}-{high}",
                    "single_low": names[0],
                    "single_high": names[1],
                    "double": names[2],
                }
            )
        if not cycles:
            refused.append({"name": background["name"], "reason": "no readable cycle", "detail": bad})
            continue
        cycle_total += len(cycles)
        ordered = sorted(labels)
        assays.append(
            gm.cohort_assay(
                assay=str(background["name"]),
                wildtype=wildtype,
                mutants=ordered,
                sequences=[labels[name] for name in ordered],
                cluster=str(background["group"]),
                extra={
                    "origin": "natural_measured",
                    "group": str(background["group"]),
                    "length": int(background["length"]),
                    "kind": background.get("kind"),
                    "cycles": cycles,
                    "unreadable_cycles": bad,
                },
            )
        )
    if not assays:
        raise SystemExit("the frozen pairwise cohort produced no readable background")
    design = {
        "mode": "cycles",
        "source": str(args.pairwise_cohort),
        "source_sha256": sha256_file(args.pairwise_cohort),
        "source_schema": cohort.get("schema"),
        "backgrounds": len(assays),
        "groups": len({row["group"] for row in assays}),
        "cycles": cycle_total,
        "site_pairs": len({cycle["site_pair"] for row in assays for cycle in row["cycles"]}),
        "states": sum(len(row["mutants"]) for row in assays),
        "refused_backgrounds": refused,
        "lengths": _length_summary(row["length"] for row in assays),
        "label_free": (
            "the frozen cohort's measured epsilon is read nowhere by this builder; only "
            "sequences and cycle positions enter the cohort"
        ),
    }
    return {"assays": assays, "design": design}


# ---------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--generated", type=Path,
                        help="the frozen generated-sequence JSONL (singles, pairs)")
    parser.add_argument("--per-stage", type=int, default=300,
                        help="generated products drawn per stage, equally over streams")
    parser.add_argument("--sites", type=int, default=12, help="scanned positions per sequence")
    parser.add_argument("--subs", type=int, default=2, help="substitutions per scanned position")
    parser.add_argument("--contacts-dir", type=Path,
                        help="directory of per-product folded-structure archives (pairs)")
    parser.add_argument("--contact-file", default="{id}.npz",
                        help="archive basename template, '{id}' substituted")
    parser.add_argument("--confidence-key", default="auto",
                        help="'auto', a literal key, or 'none' to declare no confidence filter")
    parser.add_argument("--confidence-floor", type=float, default=gm.CONFIDENCE_FLOOR)
    parser.add_argument("--pairs-per-stratum", type=int, default=2,
                        help="matched contact/control pairs drawn per separation stratum")
    parser.add_argument("--pairwise-cohort", type=Path,
                        help="the frozen MegaScale double-mutant cohort JSON (cycles)")
    parser.add_argument("--device", default="cpu",
                        help="accepted because the campaign queue injects it; unused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.mode in ("singles", "pairs") and args.generated is None:
        parser.error(f"--generated is required for {args.mode}")
    if args.mode == "pairs" and args.contacts_dir is None:
        parser.error("--contacts-dir is required for pairs")
    if args.mode == "cycles" and args.pairwise_cohort is None:
        parser.error("--pairwise-cohort is required for cycles")
    if args.per_stage < 1 or args.sites < 1 or args.subs < 1 or args.pairs_per_stratum < 1:
        parser.error("--per-stage, --sites, --subs and --pairs-per-stratum are positive")

    out = gm.prepare_output_directory(args.out, COMPLETION)
    built = {"singles": build_singles, "pairs": build_pairs, "cycles": build_cycles}[args.mode](args)
    # The producer's half of the identity contract: every mode passes through here,
    # so no cohort this builder writes can be one the extraction stage refuses.
    gm.require_unique_assays(built["assays"])

    sources = {}
    for name, path in (("generated", args.generated), ("pairwise_cohort", args.pairwise_cohort)):
        if path is not None:
            sources[name] = {"path": str(path), "sha256": sha256_file(path)}
    cohort = {
        "schema": gm.COHORT_SCHEMA,
        "mode": args.mode,
        "created_utc": _now(),
        "sources": sources,
        "design": built["design"],
        "assays": built["assays"],
    }
    write_json(out / COHORT, cohort)
    write_json(out / COMPLETION, {
        "status": "complete",
        "schema": gm.COHORT_SCHEMA,
        "mode": args.mode,
        "created_utc": _now(),
        "cohort": COHORT,
        "cohort_sha256": sha256_file(out / COHORT),
        "assays": len(built["assays"]),
        "states": sum(len(row["mutants"]) for row in built["assays"]),
        "design": built["design"],
        "sources": sources,
        "code_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/capability/interactions/build_mutation_landscape_cohort.py",
                "src/capability/interactions/generated_mutation.py",
            )
        },
    })


if __name__ == "__main__":
    main()
