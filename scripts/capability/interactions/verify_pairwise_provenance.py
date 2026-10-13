#!/usr/bin/env python3
"""Bind the named assets of one reviewed subset, and qualify its reference MSAs.

The 12 backgrounds this stage covers are the reviewed human and mouse
intracellular domain constructs the frozen protocol admits. This file establishes
two things about them and nothing else: that every named asset is the exact byte
sequence it is declared to be, and that the retained DIAMOND alignments of these
12 wild types reduce to the reference alignment and profile the pairwise fit
would consume.

It is a *reference preparation* step. It launches no DIAMOND search, fits no
``plmc`` model and reads no phenotype, mutation value or model output. The
lambda grid and the held-out split it reports are the ones declared by
``fit_pairwise_sequence_baseline``: they are read from that module so the
declaration cannot drift, and the split is constructed here by calling the fit's
own row-hash function on the natural homolog rows. A background's experimental
mutation panel is never an input to any quantity on this path.

``stability_singles_cohort_v1`` is a WT-plus-variants panel, so its sequence
count is an experimental panel size, not MSA depth. The pairwise fit requires the
paired-cycle schema and refuses a singles cohort explicitly; this stage records
that contract rather than papering over it with a fallback reference.

Every emitted artefact lands under a gitignored output directory: the provenance
document, the 12-query FASTA, and one retained aligned MSA FASTA per background.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.context.homology import ALIGNMENT_FIELDS, parse_hits
from src.capability.context.profiles import (
    GAP_CODE, NEFF_IDENTITY_FLOOR, PROFILE_COVERAGE_FLOOR, REWEIGHT_IDENTITY_FLOOR,
    build_profile)
from src.capability.core.io import sha256_file, write_json
from src.capability.core.sequence_alignment import alignment_rows, verify_rows_against_profile

REPO = Path(__file__).resolve().parents[3]
SCHEMA = "pairwise_reference_provenance_v1"
ALLOWLIST_SCHEMA = "measured_selection_allowlist_v1"
SINGLES_SCHEMA = "stability_singles_cohort_v1"
PAIRWISE_SCHEMA = "draft_pairwise_stability_v1"
PLAN_SCHEMA = "stability_singles_extraction_v1"
PROFILE_SCHEMA = "pairwise_profile_features_v1"
SEARCH_SCHEMA = "pairwise_homolog_search_v1"
GROUPING_SCHEMA = "megascale_family_groups_v1"
UNIREF90_SCHEMA = "uniref90_corpus_staging_v1"
PRIOR_BASELINE_SCHEMA = "pairwise_sequence_baseline_v1"
#: The reviewed subset size and the panel size the review itself validates. A
#: background carrying 256 variants retains 256 experimental constructs; that is
#: a statement about the mutation panel, never about homolog depth.
REVIEWED_BACKGROUNDS = 12
VARIANTS_PER_BACKGROUND = 256
#: UniRef50 sequence count this host's index is declared to hold. Asserted rather
#: than adopted so a silently rebuilt or partial index cannot pass as the one the
#: retained hits were searched against.
UNIREF50_INDEXED_SEQUENCES = 60315044
#: Cached profiles were written as float32 columns and float64 scalars; the
#: reader rebuilds both and requires the cached bytes back.
ARRAY_TOLERANCE = 1e-6
SCALAR_TOLERANCE = 1e-9
#: Assets whose recomputation reads tens of gigabytes. Default is to recompute;
#: ``--no-hash-corpora`` adopts the declared digest and says so in the document.
CORPORA = ("uniref50_index", "uniref90_index", "uniref90_fasta")


class ProvenanceMismatch(RuntimeError):
    """A declared provenance fact and the re-hashed or rebound asset disagree."""


def require_equal(label: str, declared, observed) -> None:
    if declared != observed:
        raise ProvenanceMismatch(f"{label}: declared {declared!r}, observed {observed!r}")


def require_schema(label: str, document: dict, schema: str) -> None:
    observed = document.get("schema")
    if observed != schema:
        raise ProvenanceMismatch(f"{label}: schema {observed!r}, expected {schema!r}")


def read_json(path: Path, label: str) -> dict:
    """Read a named JSON asset, naming it when it is absent or unreadable."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProvenanceMismatch(f"{label} at {path} is not readable JSON: {error}") from error


def load_module(path: Path, name: str):
    """Load a sibling script by path, the repository's device for one declaration."""

    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def asset_record(path: Path, declared: str | None = None, *, recompute: bool = True) -> dict:
    """Re-hash a named asset, or adopt its declared digest when told not to read it."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    record = {"path": str(path), "bytes": path.stat().st_size,
              "declared_sha256": declared, "recomputed": bool(recompute)}
    if recompute:
        observed = sha256_file(path)
        record["sha256"] = observed
        if declared is not None:
            require_equal(f"{path} sha256", declared, observed)
    else:
        if declared is None:
            raise ProvenanceMismatch(f"{path}: cannot adopt an undeclared digest")
        record["sha256"] = declared
    return record


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].strip()
            if name in records:
                raise ProvenanceMismatch(f"{path}: duplicate FASTA record {name!r}")
            records[name] = ""
        elif name is None:
            raise ProvenanceMismatch(f"{path}: sequence before any FASTA header")
        else:
            records[name] += line.strip()
    if not records:
        raise ProvenanceMismatch(f"{path}: no FASTA records")
    return records


def require_aligned_fields(name: str, hits) -> None:
    """Every retained hit for a queried wild type must carry its alignment columns."""

    for hit in hits:
        if hit.qseq_gapped is None or hit.sseq_gapped is None:
            raise ProvenanceMismatch(
                f"{name}: hit against {hit.subject} carries no aligned sequences; the "
                "retained retrieval was not asked for homology.ALIGNMENT_FIELDS")
        if not hit.qseq_gapped or not hit.sseq_gapped:
            raise ProvenanceMismatch(
                f"{name}: hit against {hit.subject} carries an empty alignment column set")
        if len(hit.qseq_gapped) != len(hit.sseq_gapped):
            raise ProvenanceMismatch(
                f"{name}: hit against {hit.subject} has {len(hit.qseq_gapped)} query "
                f"columns and {len(hit.sseq_gapped)} subject columns; it is not a gapped "
                "column-aligned HSP")


def bind_references(names: list[str], cohort: dict, plan: dict, queries: dict[str, str]) -> list[dict]:
    """One exact wild-type binding across the allowlist, cohort, plan and query FASTA."""

    require_schema("cohort", cohort, SINGLES_SCHEMA)
    require_schema("extraction plan", plan, PLAN_SCHEMA)
    cohort_by = {row["name"]: row for row in cohort["backgrounds"]}
    plan_by = {row["name"]: row for row in plan["backgrounds"]}
    bound = []
    for name in names:
        if name not in cohort_by:
            raise ProvenanceMismatch(f"{name}: not a background of the frozen cohort")
        if name not in plan_by:
            raise ProvenanceMismatch(f"{name}: not a background of the extraction plan")
        if name not in queries:
            raise ProvenanceMismatch(f"{name}: no archived DIAMOND query record")
        wildtype = cohort_by[name].get("wildtype")
        if not wildtype:
            raise ProvenanceMismatch(f"{name}: cohort background carries no wild type")
        require_equal(f"{name} plan wild type", wildtype, plan_by[name].get("wildtype"))
        require_equal(f"{name} query sequence", wildtype, queries[name])
        require_equal(f"{name} length", len(wildtype), cohort_by[name].get("length"))
        variants = cohort_by[name].get("variants") or []
        if len(variants) != VARIANTS_PER_BACKGROUND:
            raise ProvenanceMismatch(
                f"{name}: carries {len(variants)} variants, not the reviewed "
                f"{VARIANTS_PER_BACKGROUND}")
        bound.append({
            "name": name,
            "length": len(wildtype),
            "wildtype_sha256": sha256_file_of_text(wildtype),
            "group": cohort_by[name].get("group"),
            "cluster": cohort_by[name].get("cluster"),
            "variants": len(variants),
            "pairwise_cycles_present": "cycles" in cohort_by[name],
        })
    return bound


def sha256_file_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def reference_record(name: str, wildtype: str, hits, fit, *, max_sequences: int,
                     ceiling: int) -> tuple[dict, np.ndarray | None, object | None]:
    """Rebuild this wild type's reference alignment, profile and held-out support.

    The held-out split is the fit's own :func:`held_out_mask`, computed from the
    aligned rows' own bytes. No mutation panel, phenotype or model output reaches
    this function, and the counters reported beside it are the ones the fit's
    support floors are expressed in.
    """

    record = {"name": name, "length": len(wildtype), "hits_retrieved": len(hits),
              "retrieval_ceiling": ceiling,
              "retrieval_ceiling_reached": len(hits) >= ceiling}
    if not hits:
        record["status"] = "no_homolog_hits"
        return record, None, None
    require_aligned_fields(name, hits)
    rows, weights, neff = alignment_rows(
        wildtype, name, hits, max_sequences=max_sequences,
        coverage_floor=PROFILE_COVERAGE_FLOOR, reweight_identity=REWEIGHT_IDENTITY_FLOOR)
    profile = build_profile(
        wildtype, name, hits, max_sequences=max_sequences,
        coverage_floor=PROFILE_COVERAGE_FLOOR, reweight_identity=REWEIGHT_IDENTITY_FLOOR,
        neff_identity_floor=NEFF_IDENTITY_FLOOR)
    verify_rows_against_profile(rows, weights, profile)
    if not len(rows):
        record["status"] = "no_retained_rows"
        return record, None, None

    coverage = np.array([100.0 * (hit.qend - hit.qstart + 1) / len(wildtype) for hit in hits])
    held = fit.held_out_mask(rows, fit.SPLIT_SEED)
    scored = fit.scored_support(rows[held]) if held.any() else np.zeros((0, len(wildtype)), bool)
    record.update({
        "status": "supported",
        "retained_rows": int(len(rows)),
        "distinct_hit_subjects": len({hit.subject for hit in hits}),
        "gap_fraction": round(float((rows == GAP_CODE).mean()), 6),
        "row_coverage_percent": {"min": round(float(coverage.min()), 3),
                                 "median": round(float(np.median(coverage)), 3)},
        "supported_columns": int((profile.column_weight > 0).sum()),
        "supported_column_fraction": round(float((profile.column_weight > 0).mean()), 6),
        "neff_at_30_percent_identity": round(float(neff), 6),
        "effective_depth_per_site": round(float(neff) / len(wildtype), 6),
        "max_identity_over_query": round(float(profile.max_identity_over_query), 4),
        "coordinate_weight_validation": "passed",
        "train_rows": int((~held).sum()),
        "held_out_rows": int(held.sum()),
        "held_out_scored_rows": int(scored.any(axis=1).sum()) if scored.size else 0,
        "held_out_scored_positions": int(scored.sum()) if scored.size else 0,
        "clears_train_floor": bool(int((~held).sum()) >= fit.MIN_TRAIN_ROWS),
        "clears_held_out_floor": bool(int(held.sum()) >= fit.MIN_HELD_OUT_ROWS),
        "reference_inputs": "natural homolog rows only",
    })
    if not (record["clears_train_floor"] and record["clears_held_out_floor"]):
        record["status"] = "insufficient_homolog_support"
    return record, rows, profile


def cached_parity(name: str, frequencies, column_weight, scalars, cached) -> dict:
    """Require rebuilt columns and scalars to reproduce the cached bytes exactly.

    Both sources a reader can rebuild the profile from -- the fit's own reader in
    :func:`profile_parity` and the unchanged builder in :func:`builder_rebuild`
    -- are held to the same cached arrays, so neither can report a number the
    shipped cache does not carry.
    """

    record = {
        "frequencies_max_absolute_difference":
            float(np.abs(frequencies - cached[f"{name}|frequencies"]).max()),
        "column_weight_max_absolute_difference":
            float(np.abs(column_weight - cached[f"{name}|column_weight"]).max()),
        "scalars_max_absolute_difference":
            float(np.abs(scalars - cached[f"{name}|scalars"]).max()),
    }
    if (record["frequencies_max_absolute_difference"] > ARRAY_TOLERANCE
            or record["column_weight_max_absolute_difference"] > ARRAY_TOLERANCE):
        raise ProvenanceMismatch(
            f"{name}: the rebuilt profile does not reproduce the cached frequencies "
            f"(max |difference| {record})")
    if record["scalars_max_absolute_difference"] > SCALAR_TOLERANCE:
        raise ProvenanceMismatch(
            f"{name}: the rebuilt profile does not reproduce the cached scalars "
            f"({record['scalars_max_absolute_difference']:.3e} > {SCALAR_TOLERANCE:.0e})")
    record["reproduced"] = True
    return record


def profile_parity(name: str, profile, cached) -> dict:
    """The fit's own reader against the cached profile for this background."""

    scalars = np.array([profile.neff, profile.log10_neff, profile.max_identity_over_query,
                        profile.n_sequences, profile.n_hits], dtype=np.float64)
    return cached_parity(name, profile.frequencies.astype(np.float32),
                         profile.column_weight.astype(np.float32), scalars, cached)


def relatedness(bindings: list[dict], family_groups: dict, purge: dict) -> dict:
    """Group eligibility from the frozen labels and the cohort's own purge, no phenotype."""

    require_schema("family groups", family_groups, GROUPING_SCHEMA)
    assignments = family_groups["assignments"]
    labels = []
    for binding in bindings:
        name, group = binding["name"], binding["group"]
        declared = assignments.get(name)
        if group is None or declared is None:
            raise ProvenanceMismatch(f"{name}: no frozen group label to relate")
        require_equal(f"{name} frozen group", group, declared)
        labels.append(group)
    duplicated = sorted({group for group in labels if labels.count(group) > 1})
    if duplicated:
        raise ProvenanceMismatch(
            f"two allowlisted backgrounds share a frozen group: {duplicated}; they are one "
            "family and cannot both stand as independent holdouts")
    within = {
        threshold: sorted({tuple(sorted((left, right)))
                           for left, targets in mapping.items() if left in set(labels)
                           for right in targets if right in set(labels)})
        for threshold, mapping in sorted(purge.items())
    }
    return {
        "group_labels": {binding["name"]: binding["group"] for binding in bindings},
        "distinct_groups": len(set(labels)),
        "remote_purge_within_allowlist": within,
        "reading": (
            "the labels are the frozen cohort homology groups and remote_purge is that "
            "cohort's own declared remote-similarity stratification; both are "
            "sequence-derived and outcome-blind, and neither proves evolutionary "
            "independence"),
    }


def prior_fit(prior: dict, names: list[str], *, hits_sha256: str, plmc_sha256: str,
              cohort_sha256: str) -> dict:
    """The archived pairwise fit's own records for these 12, bound to its inputs.

    This is reported as prior evidence only. It was fitted on the archived
    pairwise-cycle cohort, whose wild types equal these 12; its cohort artefact is
    not the singles cohort used for reference readiness.
    """

    require_schema("prior baseline", prior, PRIOR_BASELINE_SCHEMA)
    summary = prior["summary"]
    require_equal("prior fit hits", summary["hits"]["sha256"], hits_sha256)
    require_equal("prior fit plmc", summary["plmc"]["sha256"], plmc_sha256)
    require_equal("prior fit cohort", summary["cohort"]["sha256"], cohort_sha256)
    by_name = {record["name"]: record for record in prior["backgrounds"]}
    records = []
    for name in names:
        record = by_name.get(name)
        if record is None:
            raise ProvenanceMismatch(f"{name}: absent from the archived pairwise fit")
        records.append({
            "name": name,
            "status": record["status"],
            "retained_rows": record.get("rows"),
            "neff_at_30_percent_identity": record.get("neff_at_30_percent_identity"),
            "selected_lambda_e": record.get("selected_lambda_e"),
            "selected_at_grid_edge": record.get("selected_at_grid_edge"),
            "held_out_gain_over_independent_site": record.get(
                "held_out_gain_over_independent_site"),
            "retrieval_ceiling_reached": record.get("retrieval_ceiling_reached"),
        })
    fitted = [record for record in records if record["status"] == "fitted"]
    return {
        "cohort_sha256": cohort_sha256,
        "cohort_schema": PAIRWISE_SCHEMA,
        "selection_criterion": summary["grid"]["selection_criterion"],
        "fitted": len(fitted),
        "selected_at_grid_edge": sum(bool(r["selected_at_grid_edge"]) for r in fitted),
        "backgrounds": records,
        "reading": (
            "these are the archived pairwise fit's own selected regularizations; the "
            "reference qualification it used is held-out homolog sequence prediction, "
            "and no phenotype enters it"),
    }


def builder_rebuild(builder: Path, plan: Path, hits: Path, out: Path, names: list[str],
                    cached) -> dict:
    """Re-run the unchanged profile builder and require the cached 12 back.

    The builder is executed as shipped, on the same plan and hit table the cache
    binds, and its own output arrays are then compared against the cache for the
    reviewed subset. The fresh archive lives under the stage's output directory so
    the cached one is never rewritten.
    """

    if out.exists():
        raise FileExistsError(out)
    completed = subprocess.run(
        [sys.executable, str(builder), "--plan", str(plan), "--hits", str(hits),
         "--out", str(out)], capture_output=True, text=True, cwd=str(REPO))
    if completed.returncode != 0:
        raise ProvenanceMismatch(
            f"the profile builder failed ({completed.returncode}): "
            f"{completed.stderr.strip().splitlines()[-3:]}")
    rebuilt = np.load(out, allow_pickle=False)
    records = [{"name": name, **cached_parity(
        name, rebuilt[f"{name}|frequencies"], rebuilt[f"{name}|column_weight"],
        rebuilt[f"{name}|scalars"], cached)} for name in names]
    return {"path": str(out), "sha256": sha256_file(out), "bytes": out.stat().st_size,
            "backgrounds": records}


def parse_ceiling(manifest: dict) -> int:
    command = list(manifest.get("command") or [])
    if "--max-target-seqs" not in command:
        raise ProvenanceMismatch("the search manifest declares no --max-target-seqs ceiling")
    value = command[command.index("--max-target-seqs") + 1]
    try:
        return int(value)
    except ValueError as error:
        raise ProvenanceMismatch(f"--max-target-seqs {value!r} is not an integer") from error


def run(args) -> dict:
    plan = read_json(args.plan, "extraction plan")
    cohort = read_json(args.cohort, "frozen cohort")
    search = read_json(args.search_manifest, "search manifest")
    roles = read_json(args.roles, "measured-selection protocol")
    allowlist = read_json(args.allowlist, "reviewed allowlist")
    family_groups = read_json(args.family_groups, "frozen family groups")
    prior = read_json(args.prior_baseline, "archived pairwise fit")
    uniref90 = read_json(args.uniref90_manifest, "UniRef90 staging manifest")
    fit = load_module(args.fit, "pairwise_sequence_baseline")

    require_schema("allowlist", allowlist, ALLOWLIST_SCHEMA)
    require_schema("search manifest", search, SEARCH_SCHEMA)
    require_schema("UniRef90 staging manifest", uniref90, UNIREF90_SCHEMA)
    names = [row["name"] for row in allowlist["backgrounds"]]
    if len(names) != REVIEWED_BACKGROUNDS or len(set(names)) != REVIEWED_BACKGROUNDS:
        raise ProvenanceMismatch(f"the allowlist holds {len(names)} backgrounds, not "
                                 f"{REVIEWED_BACKGROUNDS} distinct ones")
    review = allowlist["review"]
    require_equal("reviewed panel count", review["frozen_panel_count_validated"],
                  REVIEWED_BACKGROUNDS * VARIANTS_PER_BACKGROUND)
    declared_cohort = {entry["sha256"] for row in allowlist["backgrounds"]
                       for entry in row["provenance"]}
    if len(declared_cohort) != 1:
        raise ProvenanceMismatch(f"the allowlist declares several cohort digests: "
                                 f"{sorted(declared_cohort)}")

    plan_sha256 = sha256_file(args.plan)
    hits_sha256 = sha256_file(args.hits)
    try:
        cached = np.load(args.profiles, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ProvenanceMismatch(f"{args.profiles} is not a readable array archive: "
                                 f"{error}") from error
    try:
        profile_metadata = json.loads(str(cached["metadata"]))
    except ValueError as error:
        raise ProvenanceMismatch(f"{args.profiles} carries no readable metadata: "
                                 f"{error}") from error
    require_schema("cached profile features", profile_metadata, PROFILE_SCHEMA)
    require_equal("cached profile plan", profile_metadata["plan_sha256"], plan_sha256)
    require_equal("cached profile hits", profile_metadata["hits_sha256"], hits_sha256)

    recompute = {name: args.hash_corpora for name in CORPORA}
    assets = {
        "allowlist": asset_record(args.allowlist, roles["protocol"]["allowlist_sha256"]),
        "cohort": asset_record(args.cohort, next(iter(declared_cohort))),
        "extraction_plan": asset_record(args.plan),
        "profile_features": asset_record(args.profiles),
        "hits": asset_record(args.hits, search["hits"]["sha256"]),
        "query_fasta": asset_record(args.query_fasta, search["query"]["sha256"]),
        "search_manifest": asset_record(args.search_manifest),
        "diamond": asset_record(args.diamond, search["tool"]["binary_sha256"]),
        "plmc": asset_record(args.plmc, prior["summary"]["plmc"]["sha256"]),
        "prior_baseline": asset_record(args.prior_baseline),
        "prior_cohort": asset_record(args.prior_cohort, prior["summary"]["cohort"]["sha256"]),
        "family_groups": asset_record(args.family_groups),
        "roles": asset_record(args.roles),
        "fit_script": asset_record(args.fit),
        "uniref50_index": asset_record(args.uniref50_index, search["database"]["sha256"],
                                       recompute=recompute["uniref50_index"]),
        "uniref90_manifest": asset_record(args.uniref90_manifest),
        "uniref90_index": asset_record(args.uniref90_index, uniref90["index"]["sha256"],
                                       recompute=recompute["uniref90_index"]),
        "uniref90_fasta": asset_record(args.uniref90_fasta, uniref90["payload"]["digests"]["sha256"],
                                       recompute=recompute["uniref90_fasta"]),
    }
    require_equal("UniRef50 indexed sequences", UNIREF50_INDEXED_SEQUENCES,
                  search["database"]["indexed_sequences"])
    require_equal("plan cohort digest", plan["cohort_sha256"], assets["cohort"]["sha256"])
    require_equal("profile metadata hits", profile_metadata["hits_sha256"],
                  assets["hits"]["sha256"])
    require_equal("UniRef90 index bytes", uniref90["index"]["bytes"],
                  assets["uniref90_index"]["bytes"])
    require_equal("UniRef90 fasta bytes", uniref90["payload"]["bytes"],
                  assets["uniref90_fasta"]["bytes"])

    queries = read_fasta(args.query_fasta)
    require_equal("query record count", search["query"]["records"], len(queries))
    bindings = bind_references(names, cohort, plan, queries)
    ceiling = parse_ceiling(search)

    grouped: dict[str, list] = {name: [] for name in names}
    for hit in parse_hits(args.hits, fields=ALIGNMENT_FIELDS):
        if hit.query in grouped:
            grouped[hit.query].append(hit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "msa").mkdir(parents=True, exist_ok=True)
    references, parity, built = [], [], []
    for binding in bindings:
        name = binding["name"]
        wildtype = queries[name]
        record, rows, profile = reference_record(
            name, wildtype, grouped[name], fit, max_sequences=args.max_sequences, ceiling=ceiling)
        references.append(record)
        if rows is not None:
            built.append(name)
            fit.write_alignment(args.out_dir / "msa" / f"{name.replace('/', '_')}.fasta",
                                wildtype, rows)
        if profile is not None and not args.skip_profile_parity:
            parity.append({"name": name, **profile_parity(name, profile, cached)})

    query_path = args.out_dir / "query_12.faa"
    query_path.write_text("".join(f">{name}\n{queries[name]}\n" for name in sorted(names)))
    rebuild = None
    if not args.skip_builder_rebuild:
        rebuild = builder_rebuild(args.builder, args.plan, args.hits,
                                  args.out_dir / "rebuild" / "profile_features.npz",
                                  [binding["name"] for binding in bindings], cached)

    document = {
        "schema": SCHEMA,
        "verifier_script": asset_record(Path(__file__)),
        "assets": assets,
        "reviewed_subset": {
            "count": len(names),
            "names": sorted(names),
            "length_range": [min(b["length"] for b in bindings),
                             max(b["length"] for b in bindings)],
            "length_sum": sum(b["length"] for b in bindings),
            "organism_scope": review["organism_scope"],
            "hazard_exclusion": review["hazard_exclusion"],
            "group_semantics": review["group_semantics"],
        },
        "singles_contract": {
            "cohort_schema": SINGLES_SCHEMA,
            "pairwise_schema": PAIRWISE_SCHEMA,
            "variants_per_background": VARIANTS_PER_BACKGROUND,
            "reading": (
                "WT plus 256 variants is an experimental mutation panel; the variant "
                "count is not MSA depth, and the pairwise fit refuses a singles cohort "
                "because it needs paired-cycle contrasts"),
        },
        "bindings": bindings,
        "references": references,
        "profile_parity": {
            "cached_path": str(args.profiles),
            "cached_sha256": assets["profile_features"]["sha256"],
            "rebuilt_in_process": parity,
            "rebuilt_by_builder": rebuild,
        },
        "reference_constraints": {
            "fit_script_sha256": assets["fit_script"]["sha256"],
            "lambda_e_grid": list(fit.LAMBDA_E_GRID),
            "lambda_h": fit.LAMBDA_H,
            "max_iter": fit.MAX_ITER,
            "gap_ignoring": True,
            "split_seed": fit.SPLIT_SEED,
            "held_out_fraction": fit.HELD_OUT_FRACTION,
            "min_train_rows": fit.MIN_TRAIN_ROWS,
            "min_held_out_rows": fit.MIN_HELD_OUT_ROWS,
            "pseudocount": fit.PSEUDOCOUNT,
            "max_sequences": args.max_sequences,
            "selection_criterion": (
                "mean gap-ignoring per-site conditional log likelihood of held-out "
                "natural homolog rows; no mutation, phenotype or model output is read"),
        },
        "group_relatedness": relatedness(bindings, family_groups, cohort["remote_purge"]),
        "prior_fit": prior_fit(prior, names, hits_sha256=assets["hits"]["sha256"],
                               plmc_sha256=assets["plmc"]["sha256"],
                               cohort_sha256=assets["prior_cohort"]["sha256"]),
        "outputs": {
            "query_fasta": {"path": str(query_path), "sha256": sha256_file(query_path),
                            "records": len(names)},
            "msa_directory": str(args.out_dir / "msa"),
            "msa_sha256": {name: sha256_file(args.out_dir / "msa" / f"{name}.fasta")
                           for name in sorted(built)},
        },
        "verification": {
            "corpora_rehashed": [name for name in CORPORA if recompute[name]],
            "corpora_adopted_from_manifest": [name for name in CORPORA if not recompute[name]],
            "disabled": {
                "profile_parity": bool(args.skip_profile_parity),
                "builder_rebuild": bool(args.skip_builder_rebuild),
            },
            "no_search_launched": True,
            "no_fit_launched": True,
            "no_label_read": True,
        },
    }
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path,
                        default=REPO / "data/r6_mechanisms_20261011/measured_stability_eligibility.json")
    parser.add_argument("--roles", type=Path,
                        default=REPO / "data/r6_mechanisms_20261011/measured_stability_roles.json")
    parser.add_argument("--cohort", type=Path,
                        default=REPO / "results/R4/gate_stability_20260924/cohort/cohort.json")
    parser.add_argument("--plan", type=Path,
                        default=REPO / "results/R4/gate_stability_20260924/cohort/extraction_plan.json")
    parser.add_argument("--profiles", type=Path,
                        default=REPO / "results/R4/gate_stability_20260924/cohort/profile_features.npz")
    parser.add_argument("--hits", type=Path,
                        default=REPO / "archive/logs/R3/pairwise_homologs_20260924/search/hits.tsv")
    parser.add_argument("--search-manifest", type=Path,
                        default=REPO / "archive/logs/R3/pairwise_homologs_20260924/search/search_manifest.json")
    parser.add_argument("--query-fasta", type=Path,
                        default=REPO / "archive/logs/R3/pairwise_homologs_20260924/search/query.faa")
    parser.add_argument("--prior-baseline", type=Path,
                        default=REPO / "archive/logs/R3/pairwise_homologs_20260924/baseline/baseline_q.json")
    parser.add_argument("--prior-cohort", type=Path,
                        default=REPO / "archive/logs/R3/pairwise_cohort_20260924/cohort.json")
    parser.add_argument("--family-groups", type=Path,
                        default=REPO / "data/pairwise_assets/megascale_family_groups_20260924.json")
    parser.add_argument("--diamond", type=Path, default=REPO / "external/tools/diamond/diamond")
    parser.add_argument("--plmc", type=Path,
                        default=REPO / "data/pairwise_assets/plmc/debbiemarkslab-plmc-18c9e55/bin/plmc")
    parser.add_argument("--uniref50-index", type=Path,
                        default=REPO / "data/homology_db/uniref50_full.dmnd")
    parser.add_argument("--uniref90-manifest", type=Path,
                        default=REPO / "data/pairwise_assets/uniref90_2026_03/uniref90_staging_manifest.json")
    parser.add_argument("--uniref90-index", type=Path,
                        default=REPO / "data/pairwise_assets/uniref90_2026_03/uniref90_full.dmnd")
    parser.add_argument("--uniref90-fasta", type=Path,
                        default=REPO / "data/pairwise_assets/uniref90_2026_03/uniref90.fasta.gz")
    parser.add_argument("--fit", type=Path,
                        default=REPO / "scripts/capability/interactions/fit_pairwise_sequence_baseline.py")
    parser.add_argument("--builder", type=Path,
                        default=REPO / "scripts/capability/interactions/build_pairwise_profile_features.py")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "data/pairwise_assets/same12_provenance_20260924")
    parser.add_argument("--max-sequences", type=int, default=20000,
                        help="rows retained per background; keep it at or above the "
                             "search's --max-target-seqs, because truncating on tied "
                             "bitscores can make the rows and the profile disagree and "
                             "the two-reader check then refuses the background")
    parser.add_argument("--no-hash-corpora", dest="hash_corpora", action="store_false",
                        help="adopt the declared digests of the large corpora instead of "
                             "re-reading tens of gigabytes")
    parser.add_argument("--skip-profile-parity", action="store_true")
    parser.add_argument("--skip-builder-rebuild", action="store_true")
    args = parser.parse_args()

    report = args.out_dir / "pairwise_provenance.json"
    if report.exists():
        raise FileExistsError(report)
    document = run(args)
    write_json(report, document)
    parity_records = list(document["profile_parity"]["rebuilt_in_process"])
    if document["profile_parity"]["rebuilt_by_builder"] is not None:
        parity_records += document["profile_parity"]["rebuilt_by_builder"]["backgrounds"]
    summary = {
        "report": str(report),
        "report_sha256": sha256_file(report),
        "reviewed": document["reviewed_subset"]["count"],
        "length_range": document["reviewed_subset"]["length_range"],
        "length_sum": document["reviewed_subset"]["length_sum"],
        "supported": sum(r["status"] == "supported" for r in document["references"]),
        "retrieval_ceiling_reached": [r["name"] for r in document["references"]
                                      if r.get("retrieval_ceiling_reached")],
        "parity_reproduced": (all(p["reproduced"] for p in parity_records)
                              if parity_records else None),
        "related_pairs": document["group_relatedness"]["remote_purge_within_allowlist"],
        "prior_fit_fitted": document["prior_fit"]["fitted"],
        "prior_fit_at_grid_edge": document["prior_fit"]["selected_at_grid_edge"],
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()