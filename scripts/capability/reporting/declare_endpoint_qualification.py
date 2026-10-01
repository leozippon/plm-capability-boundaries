#!/usr/bin/env python3
"""Freeze the qualification of the three newly staged endpoints.

Acquisition alone leaves an endpoint unusable. Each of these three is declared
here to the standard the admitted endpoints met: a construction and sign
convention written out rather than implied, a noise floor measured from whatever
replicate or channel structure the source carries, support counted in
independent units rather than rows, and an explicit statement of what the
endpoint licenses and what it does not.

The authored half -- construction, unit definition and scope -- lives in this
file. The measured half is read from the two reports that produced it, so no
quantity is retyped: ``census_endpoint_support.py`` supplies the counts and the
replicate, channel and between-reference floors, and
``measure_endpoint_remoteness.py`` supplies the identity bands and group counts.
The written declaration is digested so a later reading can be bound to it.

No gate is built here and no model quantity is read against any endpoint.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.core.io import sha256_file, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The declared construction of each endpoint: which measured column is the
#: target, in what unit, under what sign convention, and what one independent
#: unit is. A gate reading one of these inherits this declaration.
CONSTRUCTION: dict[str, dict[str, str]] = {
    "mgnify_stability_cho2026": {
        "target": "the combined `deltaG` column of the staged table, the folding free energy inferred jointly from both protease channels",
        "unit_of_measurement": "kcal/mol",
        "sign_convention": "larger deltaG is more stable, as in the staged Tsuboyama 2023 config, so a destabilising substitution lowers it; a variant effect is the variant's deltaG minus its background's deltaG and is negative when destabilising",
        "admission_rule": "every one of the three reported 95% interval widths -- trypsin, chymotrypsin and combined -- no wider than 0.5 kcal/mol, which is the tightened per-channel rule the staged stability cohort already applies rather than a new one",
        "independent_unit": "the family group of the wild-type background under the frozen grouping contract 6a0a76a4a1f0d8a...cf276759c4 thresholds, 30% identity and 80% coverage of both sequences; a variant is not a unit and neither is a row",
        "noise_floor_source": "the two protease channels of the same construct, which is the structure the admitted stability endpoint's qualification used",
    },
    "domainome_beltran2025": {
        "target": "the `normalized_fitness` column, an abundance score from protein-fragment complementation in Saccharomyces cerevisiae, normalised per domain",
        "unit_of_measurement": "dimensionless normalised fitness, per-domain scaled, with `normalized_fitness_sigma` as its standard error",
        "sign_convention": "larger normalised fitness is greater cellular abundance, so a destabilising or destabilised-expression substitution lowers it",
        "admission_rule": "substitution rows only: the stop-codon rows are a separate class and the table's own `quality_rank` orders domains by library quality, so a gate declares its rank cut before reading any model quantity",
        "independent_unit": "the domain, and more conservatively the Pfam accession carried in its `domain_ID`; a variant is not a unit",
        "noise_floor_source": "the three biological replicates of each variant's input and output counts",
    },
    "skempi2": {
        "target": "the change in binding free energy on mutation, computed as R T ln(Kd_mutant / Kd_wild-type) from the row's own parsed affinities at the row's own temperature",
        "unit_of_measurement": "kcal/mol, with the gas constant at 1.9872041e-3 kcal/mol/K",
        "sign_convention": "positive weakens binding, because Kd_mutant above Kd_wild-type raises the logarithm; a stabilising mutation is negative",
        "admission_rule": "both affinities parsed and positive and the temperature between 250 and 350 K; rows failing any of those carry no derivable value and are excluded rather than imputed",
        "independent_unit": "the complex, and within it the mutation; the 7,085 table rows are not units, because one complex contributes many mutations and one mutation may be reported by several publications",
        "noise_floor_source": "entries whose complex and mutation were measured by more than one publication",
    },
}

#: What each endpoint licenses once qualified, and what it does not. These are
#: the boundaries a gate built on it inherits.
SCOPE: dict[str, dict[str, str]] = {
    "mgnify_stability_cho2026": {
        "licenses": "a measured-stability contrast on metagenome-assembled domains whose wild types reach identity bands no current cohort populates, which is what the retrieval and memorization gate's remote-identity end needs",
        "does_not_license": (
            "external confirmation of anything. It shares the cDNA-display "
            "proteolysis assay technology and an author with Tsuboyama 2023, so "
            "it reproduces exactly the shared-source defect L50 records and may "
            "not be read as independent replication. It also licenses no claim "
            "about a pretraining corpus: an identity band is a property of a "
            "search against one UniRef50 snapshot, and no detected alignment is "
            "not absence from any model's training data."
        ),
    },
    "domainome_beltran2025": {
        "licenses": "an external confirmation contrast for the mutation-ranking axis on targets and an assay technology used in neither development panel",
        "does_not_license": (
            "replication of the same estimand. Abundance by complementation in "
            "yeast is a different measured quantity from proteolysis-derived "
            "stability and from the heterogeneous ProteinGym assays, so "
            "agreement across the two is stronger evidence than agreement "
            "within one and is not a repeat measurement of either. A "
            "disagreement is therefore not by itself a failure to replicate."
        ),
    },
    "skempi2": {
        "licenses": "a binding-energetics contrast on protein-protein complexes whose structure is solved, on an endpoint that is a measurement rather than an annotation, which is the function gate's admission condition",
        "does_not_license": (
            "any claim about catalysis or substrate specificity, which this "
            "table does not measure; and no calibrated uncertainty statement. "
            "The between-reference spread is a lower bound on total uncertainty "
            "and absorbs literature-reporting heterogeneity as well as "
            "measurement error, so it serves as a floor and not as an error "
            "model. Entries are curated from 295 publications with "
            "non-uniform methods, so a contrast has to carry the method column."
        ),
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    logs = REPO_ROOT / "logs" / "dataset_registry_20260923"
    parser.add_argument("--support", type=Path, default=logs / "endpoint_support.json")
    parser.add_argument("--remoteness", type=Path, default=logs / "remoteness.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "endpoint_qualification_20260924.json",
    )
    args = parser.parse_args(argv)

    support = json.loads(args.support.read_text(encoding="utf-8"))
    remoteness = json.loads(args.remoteness.read_text(encoding="utf-8"))
    missing = sorted(set(CONSTRUCTION) - set(support))
    if missing:
        raise SystemExit(f"{args.support} carries no support for {missing}")

    declaration = {
        "schema_version": "endpoint_qualification_v1",
        "declared": "2026-09-24",
        "purpose": (
            "Qualification of the three endpoints staged for the capability map. "
            "Construction, unit definition and scope are authored here; every "
            "quantity is read from the two measurement reports named below and "
            "is not retyped."
        ),
        "measurement_reports": {
            "support": {
                "path": str(args.support.relative_to(REPO_ROOT)),
                "sha256": sha256_file(args.support),
            },
            "remoteness": {
                "path": str(args.remoteness.relative_to(REPO_ROOT)),
                "sha256": sha256_file(args.remoteness),
            },
        },
        "endpoints": {},
    }

    for name in sorted(CONSTRUCTION):
        entry: dict[str, object] = {
            "construction": CONSTRUCTION[name],
            "scope": SCOPE[name],
            "measured_support": support[name],
        }
        if name == "mgnify_stability_cho2026":
            entry["measured_remoteness"] = remoteness
        declaration["endpoints"][name] = entry

    write_json(args.out, declaration)
    print(f"{args.out} sha256 {sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
