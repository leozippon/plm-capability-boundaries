#!/usr/bin/env python3
"""E19 cohort: the backbone panel, its prescribed pairs and the reciprocal transplant.

A CPU stage, and one that has to run on the host holding the CATH table and the
DIAMOND binary rather than in the pod, which carries neither. Its product is a
frozen cohort artefact; every later stage reads that file and re-derives nothing,
so the panel cannot drift between arms.

The funnel is reported stage by stage rather than summarised, because a cohort
that falls short of its quota owes a reader the filter that took the candidates
away. Nothing in the selection sees a model: the pair preference is reference
C-beta distance then separation then position, and the band quota is reference
pLDDT.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.arms import env_path  # noqa: E402
from src.capability.core.families import CATH_SUPERFAMILY_TSV, load_cath_superfamilies  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.forcing import forcing_design as D  # noqa: E402
from src.capability.forcing.backbone_cohort import (  # noqa: E402
    Census,
    cohort_payload,
    distinct_shingle_fraction,
    read_backbone,
    reciprocal_transplant,
    require_cohort,
    select_pairs,
)
from src.capability.generation.conditioned_generation import SWISSPROT_FASTA, swissprot_sequences  # noqa: E402
from src.capability.generation.structure_inputs import (  # noqa: E402
    ALPHAFOLD_ROOT,
    alphafold_models,
    read_alphafold_model,
)

COMPLETION = "forcing_cohort.json"
DIAMOND = env_path("TRANSFER_DIAMOND_BIN", ROOT / "external/tools/diamond/diamond")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def single_fragment_paths(models: list[Path]) -> dict[str, dict[str, Path]]:
    """Accession to its paired PDB and mmCIF, for accessions covered by one fragment.

    A multi-fragment accession is a chain AlphaFold could not model in one piece,
    so no fragment of it is the protein and the fold of the whole is not on disk.
    The mmCIF is required beside the PDB: the C-beta geometry and the
    accessibility are read from the heavy atoms, which the CA-trace reader does
    not carry.
    """

    fragments: dict[str, set[str]] = defaultdict(set)
    paths: dict[str, Path] = {}
    for path in models:
        parts = path.name.split("-")
        fragments[parts[1]].add(parts[2])
        if parts[2] == "F1":
            paths[parts[1]] = path
    resolved: dict[str, dict[str, Path]] = {}
    for accession, pdb in sorted(paths.items()):
        if fragments[accession] != {"F1"}:
            continue
        cif = pdb.with_name(pdb.name.replace(".pdb.gz", ".cif.gz"))
        if not cif.is_file():
            continue
        resolved[accession] = {"pdb": pdb, "cif": cif}
    return resolved


def _survey(path: Path) -> dict[str, object]:
    """The cheap pass: length, confidence, composition screen, from the PDB alone."""

    model = read_alphafold_model(path)
    return {
        "accession": model.accession,
        "length": len(model),
        "sequence": model.sequence,
        "mean_plddt": float(model.plddt.mean()),
        "n_non_canonical": int(model.n_non_canonical_residues),
        "distinct_shingle_fraction": distinct_shingle_fraction(model.sequence),
    }


def survey(paths: dict[str, dict[str, Path]], *, processes: int) -> list[dict[str, object]]:
    ordered = [paths[accession]["pdb"] for accession in sorted(paths)]
    if processes <= 1:
        return [_survey(path) for path in ordered]
    with Pool(processes) as pool:
        return pool.map(_survey, ordered, chunksize=64)


# ------------------------------------------------- the within-cohort identity screen


def self_identity_exclusions(
    records: list[dict[str, object]], *, diamond: Path, threads: int, work: Path,
) -> dict[str, object]:
    """Drop a backbone that is above the identity ceiling against an already-kept one.

    A self-search rather than a pairwise alignment matrix: the cohort is a few
    hundred sequences and DIAMOND is already the project's declared aligner.
    Identity is taken over the *query*, which is the declaration
    :class:`src.capability.context.homology.Hit` makes and the only form in which
    a 40 %-length perfect fragment does not read as a near-identical sequence.

    Keeping is deterministic and priority-ordered, so the screen does not depend
    on which direction a pair happened to be reported in.
    """

    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "cohort.faa"
    fasta.write_text(
        "".join(f">{record['accession']}\n{record['sequence']}\n" for record in records),
        encoding="utf-8",
    )
    database = work / "cohort"
    hits = work / "cohort_hits.tsv"
    fields = ("qseqid", "sseqid", "nident", "qlen")
    commands = [
        [str(diamond), "makedb", "--in", str(fasta), "-d", str(database),
         "--threads", str(threads), "--quiet"],
        # `--masking 0` is a correctness requirement, not a tuning choice, and it
        # is recorded below rather than left to the source: DIAMOND's default
        # low-complexity masking truncates high-identity alignments, which
        # under-states identity over the query for exactly the near-verbatim
        # pairs this ceiling exists to exclude. A truncated alignment would let
        # two near-identical backbones both into the panel as one unit each.
        [str(diamond), "blastp", "--query", str(fasta), "--db", f"{database}.dmnd",
         "--out", str(hits), "--outfmt", "6", *fields,
         "--very-sensitive", "--masking", "0", "--evalue", "1e-3",
         "--max-target-seqs", str(len(records) + 1), "--threads", str(threads), "--quiet"],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    neighbours: dict[str, set[str]] = defaultdict(set)
    for line in hits.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        query, subject, nident, qlen = line.split("\t")
        if query == subject:
            continue
        identity = 100.0 * int(nident) / int(qlen)
        if identity >= D.PAIRWISE_IDENTITY_MAX:
            neighbours[query].add(subject)
            neighbours[subject].add(query)
    kept: list[str] = []
    excluded: dict[str, str] = {}
    for record in records:
        accession = str(record["accession"])
        clash = sorted(neighbours[accession] & set(kept))
        if clash:
            excluded[accession] = clash[0]
        else:
            kept.append(accession)
    return {
        "kept": kept,
        "excluded": excluded,
        "ceiling_percent": D.PAIRWISE_IDENTITY_MAX,
        "identity_definition": "percent of the query identically matched (nident / qlen)",
        "queries": len(records),
        "pairs_above_ceiling": sum(len(value) for value in neighbours.values()) // 2,
        "commands": commands,
        "masking_disabled": "--masking" in commands[1] and commands[1][
            commands[1].index("--masking") + 1
        ] == "0",
    }


# -------------------------------------------------------------------- the build


def build(args: argparse.Namespace) -> None:
    census = Census()
    work = args.work or (args.out / "diamond_work")
    models = alphafold_models(args.alphafold_dir)
    paths = single_fragment_paths(models)
    census.record(
        "alphafold_models",
        len({path.name.split("-")[1] for path in models}),
        note="distinct accessions with at least one model in the staged release",
    )
    census.record(
        "single_fragment", len(paths),
        note="covered by exactly one fragment and carrying both the PDB and the mmCIF",
    )

    records = survey(paths, processes=args.processes)
    banded = []
    for record in records:
        band = D.length_band(int(record["length"]))
        if band is None:
            continue
        banded.append(dict(record) | {"length_band": band})
    census.record("length_band", len(banded),
                  note=f"within {[f'{c}+-{h}' for _, c, h in D.LENGTH_BANDS]}")

    reviewed = swissprot_sequences(
        [str(record["accession"]) for record in banded], path=args.swissprot_fasta
    )
    in_swissprot = [record for record in banded if record["accession"] in reviewed]
    census.record("swissprot_reviewed", len(in_swissprot),
                  note="the accession is a reviewed Swiss-Prot entry with a canonical sequence")

    exact = [
        record for record in in_swissprot
        if reviewed[str(record["accession"])] == record["sequence"]
    ]
    census.record(
        "sequence_matches_swissprot", len(exact),
        note=(
            "the model's sequence is exactly the Swiss-Prot sequence, so the model is of "
            "the whole canonical protein rather than a differing isoform"
        ),
    )

    canonical = [record for record in exact if int(record["n_non_canonical"]) == 0]
    census.record("canonical_residues", len(canonical))

    confident = [
        record for record in canonical if float(record["mean_plddt"]) >= D.MODEL_PLDDT_FLOOR
    ]
    census.record(
        "model_plddt", len(confident),
        note=(
            f"whole-model mean reference pLDDT >= {D.MODEL_PLDDT_FLOOR}; the design's own "
            f"{D.WINDOW_PLDDT_FLOOR} floor applies to the window the intervention reads and "
            "is enforced per candidate pair, not here"
        ),
    )

    complex_enough = [
        record for record in confident
        if float(record["distinct_shingle_fraction"]) >= D.MIN_DISTINCT_SHINGLE_FRACTION
    ]
    census.record(
        "distinct_shingle_fraction", len(complex_enough),
        note=f"distinct 5-mer fraction >= {D.MIN_DISTINCT_SHINGLE_FRACTION}",
    )

    superfamilies = load_cath_superfamilies(
        {str(record["accession"]) for record in complex_enough}, path=args.cath_table
    ) if complex_enough else {}
    single_domain = []
    for record in complex_enough:
        codes = superfamilies.get(str(record["accession"]))
        if codes is None or len(codes) != 1:
            continue
        single_domain.append(dict(record) | {"superfamily": sorted(codes)[0]})
    census.record("single_cath_superfamily", len(single_domain), note=D.SINGLE_DOMAIN_RULE)

    by_superfamily: dict[str, dict[str, object]] = {}
    for record in sorted(
        single_domain, key=lambda row: (-float(row["mean_plddt"]), str(row["accession"]))
    ):
        by_superfamily.setdefault(str(record["superfamily"]), record)
    deduplicated = sorted(by_superfamily.values(), key=lambda row: str(row["accession"]))
    census.record(
        "one_per_superfamily", len(deduplicated),
        note=f"one backbone per CATH superfamily, chosen by {D.BAND_QUOTA_PRIORITY}",
    )

    ranked = sorted(
        deduplicated, key=lambda row: (-float(row["mean_plddt"]), str(row["accession"]))
    )
    identity = self_identity_exclusions(
        ranked, diamond=args.diamond, threads=args.threads, work=work,
    ) if ranked else {"kept": [], "excluded": {}, "queries": 0}
    kept = set(identity["kept"])
    screened = [record for record in ranked if str(record["accession"]) in kept]
    census.record(
        "pairwise_identity", len(screened),
        note=f"below {D.PAIRWISE_IDENTITY_MAX}% identity over the query against every kept backbone",
    )

    quota: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in screened:
        quota[str(record["length_band"])].append(record)
    selected: list[dict[str, object]] = []
    with_pairs = 0
    units: list[dict[str, object]] = []
    per_band_examined: dict[str, int] = {}
    for band, _centre, _half in D.LENGTH_BANDS:
        taken = 0
        examined = 0
        for record in quota[band]:
            if taken >= args.backbones_per_band:
                break
            examined += 1
            backbone = read_backbone(
                pdb=paths[str(record["accession"])]["pdb"],
                cif=paths[str(record["accession"])]["cif"],
                superfamily=str(record["superfamily"]),
                band=band,
            )
            pairs = select_pairs(backbone, limit=args.pairs_per_backbone)
            if not pairs:
                continue
            with_pairs += 1
            taken += 1
            selected.append(record)
            units.extend(pairs)
        per_band_examined[band] = examined
    census.record(
        "prescribed_pairs_available", with_pairs,
        note=(
            f"at least one prescribed pair with {D.N_REFERENCE_PARTNERS} matched "
            f"non-contacting reference partners and window mean pLDDT >= "
            f"{D.WINDOW_PLDDT_FLOOR}; backbones are examined in quota order "
            f"and one that yields no pair is skipped rather than counted ({per_band_examined})"
        ),
    )
    census.record(
        "band_quota", len(selected),
        note=f"up to {args.backbones_per_band} per band by {D.BAND_QUOTA_PRIORITY}",
    )

    matching = reciprocal_transplant(units)
    matched = matching["units"]
    census.record(
        "transplant_partner_available",
        len({str(unit["accession"]) for unit in matched}),
        note=(
            f"{len(matched)} of {len(units)} units matched reciprocally inside their "
            "(length band, anchor burial band, anchor secondary-structure class, "
            "separation stratum) cell to a unit of a different backbone carrying a "
            "different anchor residue"
        ),
    )
    if not matched:
        raise SystemExit(
            "no unit found a reciprocal transplant partner; the cohort cannot be built. "
            f"Cells: {matching['cells']}"
        )

    payload = cohort_payload(
        matched,
        census=census.payload(),
        sequences={
            str(record["accession"]): str(record["sequence"])
            for record in selected
        },
        sources={
            "alphafold_dir": str(args.alphafold_dir),
            "swissprot_fasta": {
                "path": str(args.swissprot_fasta), "sha256": sha256_file(args.swissprot_fasta),
            },
            "cath_table": {
                "path": str(args.cath_table), "sha256": sha256_file(args.cath_table),
            },
            "pairwise_identity_screen": identity,
            "requested": {
                "backbones_per_band": int(args.backbones_per_band),
                "pairs_per_backbone": int(args.pairs_per_backbone),
            },
            "unmatched_unit_ids": matching["unmatched_unit_ids"],
            "match_cells": matching["cells"],
            "match_rule": matching["rule"],
        },
    )
    require_cohort(payload)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / COMPLETION, {"created_utc": _now(), "status": "complete"} | payload)
    print(json.dumps({
        "backbones": payload["backbones"],
        "backbones_per_band": payload["backbones_per_band"],
        "units": payload["units"],
        "units_per_band": payload["units_per_band"],
        "pairs_per_backbone": payload["pairs_per_backbone"],
        "forced_residue_multisets_match": payload["forced_residue_multisets"]["matches"],
        "census": payload["census"]["stages"],
        "artefact": str(args.out / COMPLETION),
    }, indent=1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alphafold-dir", type=Path, default=ALPHAFOLD_ROOT)
    parser.add_argument("--swissprot-fasta", type=Path, default=SWISSPROT_FASTA)
    parser.add_argument("--cath-table", type=Path, default=CATH_SUPERFAMILY_TSV)
    parser.add_argument("--diamond", type=Path, default=DIAMOND)
    parser.add_argument("--backbones-per-band", type=int, default=D.BACKBONES_PER_BAND)
    parser.add_argument("--pairs-per-backbone", type=int, default=D.PAIRS_PER_BACKBONE)
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument(
        "--work", type=Path, default=None,
        help="scratch directory for the within-cohort DIAMOND self-search; default <out>/diamond_work",
    )
    return parser


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
