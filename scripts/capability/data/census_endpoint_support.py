#!/usr/bin/env python3
"""Count the support the three newly staged endpoints carry, before any claim.

The retrieval gate's saturation was found by counting units rather than
variants, so a dataset's variant count says nothing about whether it lifts a
limitation. This stage counts what does matter -- targets, independent family
groups, measured variants and the uncertainty attached to each -- and verifies
by target sequence, not only by assay name, that the confirmation endpoint
overlaps neither ProteinGym's staged panel nor Tsuboyama 2023.

It reads measurements, so it is a census of support and not a model
measurement: no model is loaded and no score is read against any endpoint.
"""
from collections import Counter
from pathlib import Path
import argparse
import csv
import math
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.core.io import write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
csv.field_size_limit(1 << 24)

#: kcal per mol per kelvin, for the SKEMPI binding free-energy change.
GAS_CONSTANT = 1.9872041e-3


def pearson(pairs: list[tuple[float, float]]) -> float | None:
    """Pearson correlation of paired measurements, or None below two pairs."""

    if len(pairs) < 2:
        return None
    n = len(pairs)
    mean_a = sum(a for a, _ in pairs) / n
    mean_b = sum(b for _, b in pairs) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in pairs)
    var_a = sum((a - mean_a) ** 2 for a, _ in pairs)
    var_b = sum((b - mean_b) ** 2 for _, b in pairs)
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / (var_a * var_b) ** 0.5

#: A variant name in these libraries is ``<background>_<token>``, and the token
#: vocabulary is not fixed: the staged MGnify table uses 31,412 distinct tokens,
#: including scramble controls beside substitutions, insertions and deletions. A
#: declared token grammar therefore misparses most of the file -- the first
#: attempt here folded 1,623,711 of 2,287,291 names into the background count --
#: so backgrounds are derived from the observed name set instead: a name's
#: background is its longest proper underscore-prefix that is itself a name in
#: the table, and a name with no such prefix is its own background. That reads
#: only the names the file actually carries and assumes no vocabulary.
SUBSTITUTION_TOKEN = re.compile(r"^[A-Z]\d+[A-Z*]$")


def spread(values: list[float]) -> dict[str, float | int | None]:
    """Quantiles, mean, standard deviation and root-mean-square of a sample.

    A noise floor is reported as a distribution rather than a single number
    because its tail is what bounds a small effect, not its centre.
    """

    if not values:
        return {"n": 0}
    ordered = sorted(values)
    count = len(ordered)

    def at(fraction: float) -> float:
        return ordered[min(count - 1, int(fraction * count))]

    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / max(1, count - 1)
    return {
        "n": count,
        "min": ordered[0],
        "q1": at(0.25),
        "median": at(0.50),
        "q3": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "max": ordered[-1],
        "mean": mean,
        "sd": variance**0.5,
        "rms": (sum(value * value for value in ordered) / count) ** 0.5,
    }


def proteingym_wildtypes(directory: Path) -> dict[str, str]:
    """Reconstruct each staged assay's wild type from its first single mutant."""

    wildtypes: dict[str, str] = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                mutant = row["mutant"]
                if ":" in mutant:
                    continue
                position = int(mutant[1:-1])
                sequence = row["mutated_sequence"]
                if not 1 <= position <= len(sequence) or sequence[position - 1] != mutant[-1]:
                    continue
                wildtypes[path.stem] = (
                    sequence[: position - 1] + mutant[0] + sequence[position:]
                )
                break
    return wildtypes


def megascale_wildtypes(directory: Path) -> dict[str, str]:
    """Wild-type sequences of the staged Tsuboyama 2023 dataset2 backgrounds."""

    import pyarrow.parquet as parquet

    wildtypes: dict[str, str] = {}
    for path in sorted(directory.rglob("*.parquet")):
        table = parquet.read_table(path, columns=["WT_name", "mut_type", "aa_seq"])
        names = table.column("WT_name").to_pylist()
        kinds = table.column("mut_type").to_pylist()
        sequences = table.column("aa_seq").to_pylist()
        for name, kind, sequence in zip(names, kinds, sequences):
            if kind == "wt":
                wildtypes[name] = sequence
    return wildtypes


def sequence_overlap(
    targets: dict[str, str], reference: dict[str, str]
) -> dict[str, object]:
    """Exact and containment overlap between two named sequence sets.

    Containment is checked in both directions because a domain endpoint and a
    full-length assay target can be the same protein at two different extents,
    which an identifier comparison and an exact sequence comparison both miss.
    """

    exact: list[list[str]] = []
    contained: list[list[str]] = []
    by_sequence = {sequence: name for name, sequence in reference.items()}
    for name, sequence in sorted(targets.items()):
        if sequence in by_sequence:
            exact.append([name, by_sequence[sequence]])
            continue
        for other_name, other in sorted(reference.items()):
            if sequence in other or other in sequence:
                contained.append([name, other_name])
                break
    return {
        "targets": len(targets),
        "reference_targets": len(reference),
        "exact_sequence_matches": exact,
        "containment_matches": contained,
        "disjoint": not exact and not contained,
    }


def domainome_census(path: Path) -> dict[str, object]:
    """Count the staged Domainome table and reconstruct each domain's wild type.

    The wild type is the per-position consensus of the domain's own rows, which
    needs no offset at all. A site-saturation library varies one position per
    row and leaves the rest wild type, so the modal residue at each index is the
    wild-type residue. Two earlier attempts inferred the offset between the
    table's parent-protein ``position`` and its own ``aa_seq`` instead, and both
    failed on this file -- per-row inference reconstructed 106 of 522 domains
    with 71 self-contradictory, and intersecting the candidate offsets over
    every row left 39. The consensus is checked rather than assumed: a row that
    differs from its domain's consensus at more than one position is counted,
    and the count is reported beside the reconstruction.
    """

    archive = zipfile.ZipFile(path)
    member = next(
        name
        for name in archive.namelist()
        if name.endswith(".txt") and not name.startswith("__MACOSX")
    )
    accessions: set[str] = set()
    pfam: set[str] = set()
    rows = 0
    stops = 0
    substitutions = 0
    sigma: list[float] = []
    per_domain = Counter()
    columns: dict[str, list[Counter]] = {}
    sequences: dict[str, list[str]] = {}
    replicate_sd: list[float] = []
    replicate_pairs: list[tuple[float, float]] = []
    complete_replicates = 0
    with archive.open(member) as handle:
        reader = csv.DictReader((line.decode("utf-8") for line in handle), delimiter="\t")
        for row in reader:
            rows += 1
            domain = row["domain_ID"]
            per_domain[domain] += 1
            accessions.add(row["uniprot_ID"])
            parts = domain.split("_")
            if len(parts) >= 2 and parts[1].startswith("PF"):
                pfam.add(parts[1])
            if row["STOP"].strip().upper() == "TRUE":
                stops += 1
                continue
            substitutions += 1
            value = row["normalized_fitness_sigma"]
            if value not in ("", "NA"):
                sigma.append(float(value))
            # The reported sigma is a fitted quantity; the endpoint's own floor is
            # the spread of its three biological replicates, which is measured here
            # from the counts the table carries rather than taken from that fit.
            try:
                inputs = [float(row[f"input_count_rep{k}"]) for k in (1, 2, 3)]
                outputs = [float(row[f"output_count_rep{k}"]) for k in (1, 2, 3)]
            except (KeyError, ValueError):
                inputs = outputs = []
            if inputs and min(inputs) > 0 and min(outputs) > 0:
                complete_replicates += 1
                ratios = [
                    math.log2(out / inp) for inp, out in zip(inputs, outputs)
                ]
                mean = sum(ratios) / 3
                replicate_sd.append(
                    (sum((value - mean) ** 2 for value in ratios) / 2) ** 0.5
                )
                replicate_pairs.append((ratios[0], ratios[1]))

            sequence = row["aa_seq"]
            counts = columns.setdefault(domain, [])
            if not counts:
                counts.extend(Counter() for _ in sequence)
            if len(counts) != len(sequence):
                continue
            for index, residue in enumerate(sequence):
                counts[index][residue] += 1
            sequences.setdefault(domain, []).append(sequence)

    domains: dict[str, str] = {}
    disagreeing_rows = 0
    checked_rows = 0
    for domain, counts in columns.items():
        if not counts:
            continue
        consensus = "".join(column.most_common(1)[0][0] for column in counts)
        for sequence in sequences.get(domain, ()):
            checked_rows += 1
            if sum(1 for a, b in zip(consensus, sequence) if a != b) > 1:
                disagreeing_rows += 1
        domains[domain] = consensus

    sigma.sort()
    return {
        "member": member,
        "rows": rows,
        "substitution_rows": substitutions,
        "stop_rows": stops,
        "domains": len(per_domain),
        "domains_with_reconstructed_wildtype": len(domains),
        "rows_checked_against_their_consensus": checked_rows,
        "rows_differing_from_the_consensus_at_more_than_one_position": disagreeing_rows,
        "uniprot_accessions": len(accessions),
        "pfam_accessions_in_domain_id": len(pfam),
        "variants_per_domain_median": (
            sorted(per_domain.values())[len(per_domain) // 2] if per_domain else 0
        ),
        "normalized_fitness_sigma": spread(sigma),
        "variants_with_all_three_replicates_nonzero": complete_replicates,
        "between_replicate_sd_of_log2_output_over_input": spread(replicate_sd),
        "replicate_1_versus_2_pearson": pearson(replicate_pairs),
        "wildtypes": domains,
    }


def mgnify_census(path: Path) -> dict[str, object]:
    """Count the staged MGnify stability table's rows, backgrounds and series.

    Read with a positional reader rather than ``csv.DictReader``: the file is
    2.23 GB over 38 columns and the dictionary construction dominated the pass.
    """

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        column = {name: index for index, name in enumerate(header)}
        wanted = (
            "name",
            "lib",
            "aa_seq",
            "mgnify",
            "dm_design",
            "deltaG",
            "deltaG_95CI",
            "deltaG_t",
            "deltaG_t_95CI",
            "deltaG_c",
            "deltaG_c_95CI",
        )
        missing = [name for name in wanted if name not in column]
        if missing:
            raise SystemExit(f"{path} is missing columns {missing}")
        records = [
            tuple(row[column[name]] for name in wanted) for row in reader if len(row) == len(header)
        ]

    names = {record[0] for record in records}

    def background(name: str) -> str:
        parts = name.split("_")
        for cut in range(len(parts) - 1, 0, -1):
            candidate = "_".join(parts[:cut])
            if candidate in names:
                return candidate
        return name

    libraries = Counter()
    flags = Counter()
    tokens = Counter()
    resolved_ci = 0
    rows_per_background = Counter()
    variant_names: dict[str, set[str]] = {}
    wildtype: dict[str, str] = {}
    attributes: dict[str, tuple[str, str, str]] = {}
    channel_difference: list[float] = []
    channel_pairs: list[tuple[float, float]] = []
    combined: list[float] = []
    admitted = 0
    for (
        name,
        lib,
        sequence,
        mgnify,
        design,
        joint,
        width,
        trypsin,
        trypsin_width,
        chymotrypsin,
        chymotrypsin_width,
    ) in records:
        libraries[lib] += 1
        flags[(mgnify, design)] += 1
        root = background(name)
        rows_per_background[root] += 1
        attributes.setdefault(root, (mgnify, design, lib))
        if name == root:
            wildtype.setdefault(root, sequence)
        else:
            tokens[name[len(root) + 1 :]] += 1
            variant_names.setdefault(root, set()).add(name)
        if width not in ("", "NA"):
            try:
                if 0.0 <= float(width) <= 0.5:
                    resolved_ci += 1
            except ValueError:
                pass
        # The endpoint's own floor, on the tightened admission rule the staged
        # stability cohort already applies: every one of the three intervals no
        # wider than 0.5 kcal/mol, and then the two protease channels compared.
        try:
            widths = (float(width), float(trypsin_width), float(chymotrypsin_width))
            values = (float(joint), float(trypsin), float(chymotrypsin))
        except ValueError:
            continue
        if not all(0.0 <= value <= 0.5 for value in widths):
            continue
        admitted += 1
        if mgnify == "True" and design == "False":
            channel_difference.append(values[1] - values[2])
            channel_pairs.append((values[1], values[2]))
            combined.append(values[0])

    # A series is a background with at least one distinctly named variant, not one
    # with more than one row: the table repeats 1,999 names, and counting rows
    # promoted 1,669 duplicated wild types into series that carry no variant.
    series = {root: len(names) for root, names in variant_names.items()}
    mgnify_series = sorted(
        root
        for root in series
        if attributes[root][0] == "True" and attributes[root][1] == "False"
    )
    with_wildtype = sorted(root for root in mgnify_series if root in wildtype)
    sizes = sorted(series[root] for root in mgnify_series)
    lengths = sorted(len(wildtype[root]) for root in with_wildtype)
    substitutions = sum(n for token, n in tokens.items() if SUBSTITUTION_TOKEN.match(token))
    insertions = sum(n for token, n in tokens.items() if token.startswith("ins"))
    deletions = sum(n for token, n in tokens.items() if token.startswith("del"))
    return {
        "rows": len(records),
        "distinct_names": len(names),
        "libraries": dict(libraries),
        "mgnify_and_design_flags": {
            f"mgnify={key[0]},dm_design={key[1]}": value for key, value in flags.items()
        },
        "rows_with_combined_95CI_width_at_most_0.5_kcal_per_mol": resolved_ci,
        "rows_admitted_on_all_three_95CI_widths_at_most_0.5_kcal_per_mol": admitted,
        "trypsin_minus_chymotrypsin_deltaG_kcal_per_mol": spread(channel_difference),
        "absolute_channel_difference_kcal_per_mol": spread(
            [abs(value) for value in channel_difference]
        ),
        "channel_pearson": pearson(channel_pairs),
        "combined_deltaG_over_admitted_mgnify_rows_kcal_per_mol": spread(combined),
        "backgrounds": len(rows_per_background),
        "backgrounds_carrying_a_variant_series": len(series),
        "distinct_variants_inside_a_series": sum(series.values()),
        "repeated_names": len(records) - len(names),
        "variant_tokens": {
            "distinct": len(tokens),
            "single_substitution": substitutions,
            "insertion": insertions,
            "deletion": deletions,
            "other": sum(tokens.values()) - substitutions - insertions - deletions,
        },
        "mgnify_backgrounds_with_a_series": len(mgnify_series),
        "mgnify_backgrounds_with_a_series_and_a_wild_type_row": len(with_wildtype),
        "distinct_variants_per_mgnify_background": {
            "min": sizes[0] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "max": sizes[-1] if sizes else 0,
        },
        "wild_type_length_over_those_backgrounds": {
            "min": lengths[0] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "wildtypes": {root: wildtype[root] for root in with_wildtype},
    }


def skempi_census(path: Path) -> dict[str, object]:
    complexes: set[str] = set()
    proteins: set[str] = set()
    rows = 0
    with_both_affinities = 0
    singles = 0
    references = Counter()
    per_complex = Counter()
    measurements: dict[tuple[str, str], set[str]] = {}
    energies: dict[tuple[str, str], list[tuple[str, float]]] = {}
    derived: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            rows += 1
            pdb = row["#Pdb"]
            complexes.add(pdb)
            proteins.add(row["Protein 1"])
            proteins.add(row["Protein 2"])
            references[row["Reference"]] += 1
            mutant = row["Affinity_mut_parsed"]
            wild = row["Affinity_wt_parsed"]
            if mutant and wild:
                with_both_affinities += 1
                key = (pdb, row["Mutation(s)_cleaned"])
                measurements.setdefault(key, set()).add(row["Reference"])
                if "," not in row["Mutation(s)_cleaned"]:
                    singles += 1
                # ddG = RT ln(Kd_mut / Kd_wt): positive means the mutation weakens
                # binding. The temperature is the row's own, not a constant.
                try:
                    kelvin = float(
                        "".join(c for c in row["Temperature"] if c.isdigit() or c == ".")
                    )
                    ratio = float(mutant) / float(wild)
                except (ValueError, ZeroDivisionError):
                    continue
                if ratio > 0 and 250.0 < kelvin < 350.0:
                    value = GAS_CONSTANT * kelvin * math.log(ratio)
                    derived.append(value)
                    per_complex[pdb] += 1
                    energies.setdefault(key, []).append((row["Reference"], value))
    repeated = {
        key: sorted(refs) for key, refs in measurements.items() if len(refs) > 1
    }
    ranges: list[float] = []
    for key in repeated:
        by_reference: dict[str, list[float]] = {}
        for reference, value in energies[key]:
            by_reference.setdefault(reference, []).append(value)
        means = [sum(values) / len(values) for values in by_reference.values()]
        ranges.append(max(means) - min(means))
    return {
        "rows": rows,
        "complexes": len(complexes),
        "named_partners": len(proteins),
        "rows_with_wild_type_and_mutant_affinity": with_both_affinities,
        "distinct_references": len(references),
        "mutation_entries": len(measurements),
        "mutation_entries_measured_by_more_than_one_reference": len(repeated),
        "single_point_mutations": singles,
        "complexes_with_at_least_eight_mutation_entries": sum(
            1 for count in per_complex.values() if count >= 8
        ),
        "binding_free_energy_change_kcal_per_mol": spread(derived),
        "between_reference_range_kcal_per_mol": spread(ranges),
        "repeated_entries_disagreeing_by_more_than_1_kcal_per_mol": sum(
            1 for value in ranges if value > 1.0
        ),
        "per_entry_uncertainty_column": None,
        "uncertainty_note": (
            "SKEMPI carries no per-entry uncertainty. The between-reference range "
            "is a lower bound on total uncertainty and not a calibrated model: two "
            "publications reporting one mutation share the biology and often the "
            "assay principle, while their spread also absorbs literature-reporting "
            "heterogeneity a single laboratory would not incur."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "logs" / "dataset_registry_20260923" / "endpoint_support.json",
    )
    parser.add_argument(
        "--fasta-out",
        type=Path,
        default=REPO_ROOT / "logs" / "dataset_registry_20260923" / "mgnify_backgrounds.fasta",
    )
    args = parser.parse_args(argv)

    report: dict[str, object] = {}
    proteingym = proteingym_wildtypes(args.data_root / "proteingym" / "DMS_ProteinGym_substitutions")
    megascale = megascale_wildtypes(args.data_root / "megascale_tsuboyama2023")
    report["reference_panels"] = {
        "proteingym_assays_with_a_reconstructed_wildtype": len(proteingym),
        "megascale_backgrounds_with_a_wildtype_row": len(megascale),
    }
    print(
        f"reference panels: {len(proteingym)} ProteinGym wild types, "
        f"{len(megascale)} MegaScale wild types"
    )

    domainome = domainome_census(
        args.data_root
        / "domainome_beltran2025"
        / "Supplementary_Table_2_fitness_scores_normalized_domainranks.txt.zip"
    )
    domainome_wildtypes = domainome.pop("wildtypes")
    report["domainome_beltran2025"] = domainome
    report["domainome_beltran2025"]["overlap_with_proteingym"] = sequence_overlap(
        domainome_wildtypes, proteingym
    )
    report["domainome_beltran2025"]["overlap_with_tsuboyama2023"] = sequence_overlap(
        domainome_wildtypes, megascale
    )
    print(
        f"domainome: {domainome['rows']} rows, {domainome['domains']} domains, "
        f"{domainome['pfam_accessions_in_domain_id']} Pfam accessions"
    )

    mgnify_path = (
        args.data_root
        / "mgnify_stability_cho2026"
        / "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv"
    )
    if mgnify_path.is_file():
        mgnify = mgnify_census(mgnify_path)
        mgnify_wildtypes = mgnify.pop("wildtypes")
        report["mgnify_stability_cho2026"] = mgnify
        report["mgnify_stability_cho2026"]["overlap_with_tsuboyama2023"] = sequence_overlap(
            {k: v for k, v in list(mgnify_wildtypes.items())}, megascale
        )
        args.fasta_out.parent.mkdir(parents=True, exist_ok=True)
        with args.fasta_out.open("w", encoding="utf-8") as handle:
            for name, sequence in sorted(mgnify_wildtypes.items()):
                handle.write(f">{name}\n{sequence}\n")
        series_fasta = args.fasta_out.with_name(args.fasta_out.stem + "_series.fasta")
        series_names = set(mgnify_wildtypes)
        with series_fasta.open("w", encoding="utf-8") as handle:
            for name, sequence in sorted(mgnify_wildtypes.items()):
                if name in series_names:
                    handle.write(f">{name}\n{sequence}\n")

        print(
            f"mgnify: {mgnify['rows']} rows, {mgnify['backgrounds']} backgrounds, "
            f"{mgnify['mgnify_backgrounds_with_a_series_and_a_wild_type_row']} "
            f"MGnify-flagged with a series and a wild type -> {args.fasta_out}"
        )
    else:
        print(f"mgnify: {mgnify_path} absent, skipped")

    skempi = skempi_census(args.data_root / "skempi2" / "skempi_v2.csv")
    report["skempi2"] = skempi
    print(
        f"skempi: {skempi['rows']} rows, {skempi['complexes']} complexes, "
        f"{skempi['mutation_entries_measured_by_more_than_one_reference']} "
        "mutation entries with more than one reference"
    )

    write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
