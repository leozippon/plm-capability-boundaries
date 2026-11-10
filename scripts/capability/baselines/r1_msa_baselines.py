#!/usr/bin/env python3
"""Export and fit a cohort's homologous-support statistical ladder.

One procedure, one code path, one cohort per run. ``--cohort`` names a descriptor
in ``configs/explicit_statistics_cohorts.json``, which states the wild-type
references, the variant rows, the retained hit table, the depth cap, the optional
retained profile store to cross-check against and the references whose fits may
be reused. ``r1`` is the first such descriptor and remains the default, so the
frozen mutation-effect ladder is one instance of this procedure rather than its
hard-wired centre.

Three stages, in order, each refusing to overwrite the previous one's evidence:

``export``
    Resolve the cohort, rebuild every reference's homologous support at the
    declared depth cap, cross-check it against a retained profile store wherever
    one describes an alignment of the same depth, and write the query-first
    alignment, the weights, the variant table and the support statistics under
    ``inputs/``, together with the cohort's ``contract.json`` and the enumerated
    refusals. For ``r1`` it additionally validates the frozen 25,728-sample table
    against the measured cohort and the frozen identifier lineage and recomputes
    the LOOKUP delta of every sample.

``fit``
    Fit one plmc Potts model per wild type on the exported alignment, select the
    coupling regularization on held-out homolog sequences only, and score every
    variant of that wild type by the full-sequence energy difference. Each fit
    escalates plmc's iteration cap (200, 1000, 5000) on the identical alignment,
    weights and ``lambda`` until plmc reports convergence, and selection requires
    the whole grid to have converged. The fit tree is ``--fit-out`` (default
    ``<out>/potts-v2``) and holds its own execution manifest binding the frozen
    export's digest, every alignment and weight file, the executing code and the
    optimization protocol; each wild type is one checkpoint, and ``--resume``
    reuses a checkpoint only when that execution manifest is unchanged.

``assemble``
    Rewrite the unified score tables from the checkpoints that exist, stating
    which backgrounds are fitted, not converged, refused by the convergence gate,
    failed or still pending. A background is scoreable when a delivered model
    exists and complete only when every fit behind it converged.

A reference with fewer than two retained homolog rows carries no model. It is
enumerated as a refusal with its reason in ``inputs/refusals.json`` and in the
contract, and its samples stay in the variant table with no Potts value, so a
later lane can see that the feature is absent rather than zero.

No stage reads a phenotype, a measured score or a model output. Resources are
checked and logged before any work starts.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any
import argparse
import json
import os
import platform
import resource
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.capability.baselines import cohorts as C  # noqa: E402
from src.capability.baselines import msa_baselines as B  # noqa: E402
from src.capability.context.homology import ALIGNMENT_FIELDS, parse_hits  # noqa: E402
from src.capability.context.profiles import AA20, PSEUDOCOUNT_ALPHA  # noqa: E402

PLMC = Path("data/pairwise_assets/plmc/debbiemarkslab-plmc-18c9e55/bin/plmc")
CONFIG = Path("configs/explicit_statistics_cohorts.json")
DEFAULT_COHORT = "r1"

#: Code whose change would invalidate an existing export or checkpoint.
CODE = {
    "msa_baselines": Path("src/capability/baselines/msa_baselines.py"),
    "cohorts": Path("src/capability/baselines/cohorts.py"),
    "entrypoint": Path("scripts/capability/baselines/r1_msa_baselines.py"),
    "pairwise_stability": Path("src/capability/stability/pairwise_stability.py"),
    "sequence_alignment": Path("src/capability/core/sequence_alignment.py"),
    "profiles": Path("src/capability/context/profiles.py"),
}

#: Wall-clock model of one background's Potts fits, fitted to the 1,579 retained
#: per-attempt records of the completed R1 run (174 wild types, 4 plmc threads
#: each) by least squares in the log of the per-background total. It replaces the
#: single ``rows * length**2`` constant this stage used to extrapolate with,
#: which underpredicted the measured R1 total by a factor of 3.7 because the cost
#: grows faster than linearly in neither factor alone.
COST_MODEL = {
    "cv_selected": {"constant": 4.989e-05, "length": 2.3996, "rows": 0.7706,
                    "r_squared_log": 0.9883, "within_two_fold": 0.972, "backgrounds": 144},
    "fixed_low_depth": {"constant": 3.024e-04, "length": 2.2244, "rows": 0.0,
                        "r_squared_log": 0.9754, "within_two_fold": 0.967, "backgrounds": 30},
    "basis": "seconds = constant * length**length_exponent * rows**rows_exponent, per background "
             "and inclusive of the lambda sweep, at 4 plmc threads; fitted to the retained R1 "
             "attempt records, which reproduce their own 390.9 measured wall-hours as 396.8",
    "threads": 4,
}
#: Peak resident memory of one plmc fit, stated here because concurrency is
#: bounded by it rather than by core count. No coefficient is quoted: the
#: quantity is exactly what :func:`fit_memory_bytes` computes from the wild
#: type's length, and a rounded constant beside it would be a second, weaker
#: source for the same number.
MEMORY_MODEL = (
    "fit_memory_bytes(length): 25 double-precision parameter vectors of "
    "length*20 + length*(length-1)/2 * 400 entries, plus the marginals. It grows with the "
    "square of the length and is independent of the alignment's depth")

#: The official GEMME multi-mutant semantics, and what they cost the interaction
#: lane. Verified twice: ``pred.R``'s own ``normPred[i] = sum(predTMP * trace[myPos])``
#: sums one term per mutated site with no cross-term, and the 11 R1 double mutants
#: whose two single substitutions were scored in the same runs reproduce the
#: double's score as the sum of the singles to at most 4.1e-14 per run.
GEMME_ADDITIVITY = (
    "The official GEMME -m score of a multi-mutant is the sum of its single-site scores, so a "
    "double-mutant cycle contrast -- s(AB) - s(A) - s(B) + s(WT), with s(WT) = 0 by construction "
    "-- is identically zero and GEMME can contribute nothing non-additive to the interaction "
    "lane. It remains a per-site evolutionary-background statistic and is reported as one. Potts "
    "couplings are not additive: a multi-mutant's energy difference is taken on the complete "
    "sequence and carries every pair coupling among the changed sites, so E does contribute a "
    "non-additive term where G cannot."
)

#: Resource floors. plmc's L-BFGS keeps a history of parameter vectors, so one fit's
#: memory is a multiple of the parameter vector at the alignment's length, not of a
#: pilot's small RSS. The floor is the concurrent worst case plus declared headroom
#: that the rest of the host keeps free; the fit stage measures the real peak too.
LBFGS_VECTORS = 19
WORKING_VECTORS = 6
MIN_HEADROOM_BYTES = 96 << 30


class Log:
    """Append-only run log; the only durable record of a killed stage."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.write(f"# {path.name}")

    def write(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def digest(path: Path) -> str:
    return B.sha256_file(Path(path))


def save(path: Path, value: Any) -> None:
    """Atomic JSON publication; a half-written table is never left behind."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def record_path(path: Path) -> str:
    """How a derived file is named in a manifest: relative to the repository.

    Every reader resolves such a path against the repository root, so a path
    inside it is recorded relative and one outside it -- an export written to a
    scratch directory -- is recorded absolute rather than crashing on a
    relative_to that cannot hold.
    """

    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is not reported by this host")


def parameter_vector_bytes(length: int) -> int:
    """One double-precision parameter vector: fields plus every upper pair block."""

    q = len(AA20)
    pairs = length * (length - 1) // 2
    return 8 * (length * q + pairs * q * q)


def parameter_bytes(length: int) -> int:
    """plmc's serialized parameter size for one fit, from the reader's own layout."""

    q = len(AA20)
    pairs = length * (length - 1) // 2
    return (40 + q + 4 * (B.MAX_SEQUENCES + 1) + length + 4 * length
            + 8 * length * q + 8 * pairs * q * q)


def fit_memory_bytes(length: int) -> int:
    """Peak memory of one plmc fit: the L-BFGS history, working vectors and marginals."""

    vector = parameter_vector_bytes(length)
    return (LBFGS_VECTORS + WORKING_VECTORS) * vector + 8 * length * len(AA20)


def resource_record(work: Path) -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "host_threads": os.cpu_count(),
        "blas_environment": {name: os.environ.get(name) for name in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "mem_available_bytes": available_memory(),
        "disk_free_bytes": shutil.disk_usage(work).free,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def resource_check(work: Path, *, lengths: list[int], workers: int, log: Log) -> dict:
    """Refuse to start a stage the host cannot carry, and say why.

    The memory floor is computed from the longest alignment that will run in this
    batch, not from a pilot's measured RSS: the parameter vector grows with the
    square of the wild type's length, and eight small pilots say nothing about one
    934-residue fit. ``--workers`` fits are budgeted concurrently.
    """

    work.mkdir(parents=True, exist_ok=True)
    largest = max(lengths) if lengths else 1
    retained = sum(parameter_bytes(length) for length in lengths)
    per_fit = fit_memory_bytes(largest)
    memory_floor = per_fit * workers + MIN_HEADROOM_BYTES
    disk_floor = 2 * parameter_bytes(largest) * workers + retained + MIN_HEADROOM_BYTES
    record = {"workers": workers, "largest_length": largest,
              "parameter_vector_bytes": parameter_vector_bytes(largest),
              "lbfgs_vectors": LBFGS_VECTORS, "working_vectors": WORKING_VECTORS,
              "fit_memory_bytes": per_fit,
              "largest_parameters_bytes": parameter_bytes(largest),
              "retained_parameters_bytes_estimate": retained,
              "memory_floor_bytes": memory_floor, "disk_floor_bytes": disk_floor,
              **resource_record(work)}
    log.write(f"resources {json.dumps(record)}")
    if record["mem_available_bytes"] < memory_floor:
        raise RuntimeError(
            f"{record['mem_available_bytes']} bytes of available memory is below the "
            f"{memory_floor} byte floor: {workers} concurrent fits of length {largest} at "
            f"{per_fit} bytes each plus {MIN_HEADROOM_BYTES} bytes of headroom")
    if record["disk_free_bytes"] < disk_floor:
        raise RuntimeError(
            f"{record['disk_free_bytes']} bytes free on {work} is below the {disk_floor} byte "
            f"floor for {workers} concurrent fits and the retained parameters")
    return record


def plmc_record() -> dict:
    return {"path": str(PLMC), "sha256": digest(ROOT / PLMC)}


def code_record() -> dict:
    return {label: digest(ROOT / path) for label, path in CODE.items()}


def input_record(cohort: C.Cohort) -> dict:
    """Every byte the export depends on, by digest, as the cohort declares them."""

    record = C.input_record(cohort, extra={"plmc": str(PLMC)})
    for name, reuse in sorted(cohort.reuse.items()):
        record["reused_scores"] = {
            "path": reuse["scores"],
            "sha256": digest(ROOT / reuse["scores"]),
            "note": f"the {reuse['cohort']} score table whose backgrounds name the "
                    f"{len(cohort.reuse)} reusable fits",
        }
        break
    return record


# --------------------------------------------------------------------- export


def load_cohort(args: argparse.Namespace) -> C.Cohort:
    """Resolve the named cohort, with R1's frozen identifier function when asked for."""

    config = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    descriptor = C.load_config(config)["cohorts"].get(args.cohort)
    if descriptor is None:
        raise SystemExit(f"unknown cohort {args.cohort!r}; {config} declares "
                         f"{sorted(C.load_config(config)['cohorts'])}")
    identifier = None
    if descriptor.get("sample_id") == "frozen_cohort_sha":
        from scripts.capability.reporting.export_prediction_details import sample_id as frozen
        admission = load(ROOT / descriptor["inputs"]["admission"])
        cohort_sha = admission["cohort_sha256"]
        identifier = lambda assay, mutation: frozen(cohort_sha, assay, mutation)  # noqa: E731
    return C.load_cohort(args.cohort, config_path=config, root=ROOT, sample_id_of=identifier)


def profile_rung(cohort: C.Cohort, log: Log) -> dict[str, float]:
    """R1's LOOKUP delta per sample, recomputed and checked against the measured cohort.

    This is the ladder's first rung and it exists only where the cohort declares
    a retained profile score to check against: recomputing it is what ties the
    exported table to the published profile channel, and a cohort with no such
    retained score gets no silently invented one.
    """

    if cohort.profile_delta is None:
        return {}
    if cohort.profile_delta != "retained_cohort_profile_scores":
        raise ValueError(f"{cohort.name}: unknown profile rung {cohort.profile_delta!r}")
    catalogue = load(ROOT / cohort.inputs["wildtypes"])
    lookup = load(ROOT / cohort.inputs["lookup"])
    measured = {row["assay"]: row for row in load(ROOT / cohort.inputs["cohort"])["assays"]}
    stored = np.load(ROOT / cohort.inputs["profiles_npz"], allow_pickle=False)
    background = np.array([catalogue["corpus"]["background"][residue] for residue in AA20],
                          dtype=np.float64)
    if not np.isclose(float(lookup["settings"]["alpha"]), PSEUDOCOUNT_ALPHA):
        raise ValueError("the retained lookup alpha is not the declared profile pseudocount")
    deltas: dict[str, float] = {}
    for name, rows in cohort.by_reference.items():
        variants = [tuple(tuple(token) for token in row["substitutions"]) for row in rows]
        values = B.profile_deltas(np.asarray(stored[name], dtype=np.float64), background, variants)
        for row, value in zip(rows, values):
            retained = measured[row["assay"]]["profile_scores"][row["variant_index"]]
            if abs(value - retained) > 1e-12:
                raise RuntimeError(
                    f"{row['sample_id']}: recomputed LOOKUP score {value!r} is not the retained "
                    f"cohort's {retained!r}")
            deltas[row["sample_id"]] = value
    log.write(f"profile deltas reproduce {len(deltas)} retained cohort profile scores")
    return deltas


def comparable_store(cohort: C.Cohort, query_id: str) -> dict | None:
    """The retained profile to cross-check one rebuild against, or None with a reason.

    A store built at a cap that binds where this export's does not describes a
    different alignment, so it is not compared. The two describe the same row set
    when the two caps are equal, or when the retained row count is below both.
    """

    store = cohort.profile_store
    if store is None:
        return None
    expectation = store.expectation(query_id)
    if expectation is None:
        return None
    if store.max_sequences == cohort.max_sequences:
        return expectation
    retained = expectation["n_profile_sequences"]
    if retained < store.max_sequences and retained <= cohort.max_sequences:
        return expectation
    return None


def reused_support(cohort: C.Cohort, name: str, rows: list[dict], inputs: Path,
                   log: Log) -> dict:
    """Take a reusable reference's alignment from the cohort it is reused from.

    The bytes are copied, not referenced: the fit and the GEMME stage hash the
    alignment they are given, and a retained parameter file is only the model of
    one alignment. Copying under this cohort's own identifier keeps that binding
    checkable from this export alone.
    """

    reuse = cohort.reuse[name]
    if reuse["max_sequences"] != cohort.max_sequences:
        raise RuntimeError(
            f"{name}: {reuse['cohort']} built its alignment at {reuse['max_sequences']} rows and "
            f"this cohort declares {cohort.max_sequences}; a model of another depth is not this "
            "cohort's model")
    source = reuse["support"]
    fasta, weights = inputs / "msa" / f"{name}.fa", inputs / "msa" / f"{name}.weights"
    for origin, destination, recorded in (
            (ROOT / source["msa"], fasta, source["msa_sha256"]),
            (ROOT / source["weights"], weights, source["weights_sha256"])):
        if digest(origin) != recorded:
            raise RuntimeError(
                f"{name}: {origin} is not the alignment {reuse['cohort']} recorded; its fits "
                "cannot be reused onto it")
        destination.write_bytes(origin.read_bytes())
    entry = {key: source[key] for key in (
        "rows", "neff_at_30_percent_identity", "weights_sum", "effective_depth_per_site",
        "frequencies_max_absolute_deviation", "frequencies_verified") if key in source}
    log.write(f"reuse {name} from {reuse['cohort']}/{reuse['wildtype_id']} "
              f"rows={entry['rows']} status={reuse['status']}")
    return {
        "wildtype_id": name,
        "length": len(cohort.sequences[name]),
        "query_id": reuse["wildtype_id"],
        "aliases": list(next(entry.aliases for entry in cohort.references
                             if entry.wildtype_id == name)),
        "assays": sorted({row["assay"] for row in rows}),
        "samples": len(rows),
        "max_sequences": cohort.max_sequences,
        "support_source": {
            "cohort": reuse["cohort"],
            "wildtype_id": reuse["wildtype_id"],
            "msa": source["msa"],
            "msa_sha256": source["msa_sha256"],
            "weights_sha256": source["weights_sha256"],
            "reason": "byte-identical reference sequence; the reused alignment is the model's own",
        },
        "params_available_for_reuse": reuse.get("params"),
        "msa": record_path(fasta),
        "msa_sha256": digest(fasta),
        "weights": record_path(weights),
        "weights_sha256": digest(weights),
        **entry,
    }


def grouped_hits(cohort: C.Cohort, log: Log) -> dict[str, list]:
    """Every reference's retained alignment hits, read once from the cohort's table."""

    wanted = {entry.query_id: entry.wildtype_id for entry in cohort.references
              if entry.wildtype_id not in cohort.reuse}
    hits: dict[str, list] = {name: [] for name in wanted}
    for hit in parse_hits(cohort.hits, fields=ALIGNMENT_FIELDS):
        if hit.query in hits:
            hits[hit.query].append(hit)
    log.write(f"{sum(len(value) for value in hits.values())} retained alignments over "
              f"{sum(1 for value in hits.values() if value)} of {len(hits)} searched references")
    return {wanted[name]: value for name, value in hits.items()}


def stage_export(args: argparse.Namespace, log: Log) -> dict:
    cohort = load_cohort(args)
    out = args.out or cohort.out
    if (out / "inputs").exists():
        raise FileExistsError(
            f"{out / 'inputs'} exists; the export is written once. Remove that directory "
            "deliberately if the inputs genuinely changed.")
    resource_check(out, lengths=[1], workers=args.workers, log=log)
    log.write(f"cohort {cohort.name}: {json.dumps(cohort.counts())} cap={cohort.max_sequences} "
              f"search_cap={cohort.search_max_target_seqs} reuse={len(cohort.reuse)}")

    deltas = profile_rung(cohort, log)
    hits = grouped_hits(cohort, log)
    grouped = cohort.by_reference
    inputs = out / "inputs"
    (inputs / "msa").mkdir(parents=True, exist_ok=True)
    support: dict[str, dict] = {}
    refusals: dict[str, dict] = {}
    saturated: list[str] = []
    for entry in cohort.references:
        name, wildtype = entry.wildtype_id, entry.sequence
        if name in cohort.reuse:
            support[name] = reused_support(cohort, name, grouped[name], inputs, log)
            continue
        if not hits.get(name):
            refusals[name] = {"reason": "no retained alignment hit", "rows": 0,
                              "samples": len(grouped[name])}
            log.write(f"refuse {name}: no retained alignment hit")
            continue
        expectation = comparable_store(cohort, entry.query_id)
        rows, weights, _, statistics = B.reconstruct_support(
            wildtype, entry.query_id, hits[name], max_sequences=cohort.max_sequences,
            expect=expectation)
        if statistics["rows"] < B.MIN_POTTS_ROWS:
            refusals[name] = {
                "reason": f"{statistics['rows']} retained homolog rows is below the "
                          f"{B.MIN_POTTS_ROWS}-row floor of a Potts fit",
                "rows": statistics["rows"], "samples": len(grouped[name])}
            log.write(f"refuse {name}: {refusals[name]['reason']}")
            continue
        fasta, weights_path = inputs / "msa" / f"{name}.fa", inputs / "msa" / f"{name}.weights"
        B.write_query_first_alignment(fasta, weights_path, wildtype, rows, weights)
        if statistics["depth_cap_reached"]:
            saturated.append(name)
        support[name] = {
            "wildtype_id": name,
            "length": len(wildtype),
            "query_id": entry.query_id,
            "aliases": list(entry.aliases),
            "assays": sorted({row["assay"] for row in grouped[name]}),
            "samples": len(grouped[name]),
            "hits_retrieved": len(hits[name]),
            "msa": record_path(fasta),
            "msa_sha256": digest(fasta),
            "weights": record_path(weights_path),
            "weights_sha256": digest(weights_path),
            **statistics,
        }
        log.write(f"support {name} L={len(wildtype)} rows={statistics['rows']} "
                  f"neff={statistics['neff_at_30_percent_identity']:.3f} "
                  f"verified={statistics['frequencies_verified']}")
    verified = [entry for entry in support.values() if entry.get("frequencies_verified")]
    declared = cohort.expect.get("refusals")
    if declared is not None and declared != len(refusals):
        raise RuntimeError(
            f"{cohort.name}: {len(refusals)} references are refused where the descriptor declares "
            f"{declared}; the cohort's support changed and the contract would misstate it")

    table = [dict(row) for row in cohort.rows]
    for row in table:
        if deltas:
            row["profile_delta"] = deltas[row["sample_id"]]
    counts = cohort.counts()
    save(inputs / "variant_table.json", {
        "schema": B.SCHEMA_VARIANT_TABLE,
        "cohort": cohort.name,
        "source": str(cohort.inputs[cohort.variant_source_label]),
        "source_label": cohort.variant_source_label,
        "source_sha256": C.path_digest(ROOT / cohort.inputs[cohort.variant_source_label]),
        "profile_pseudocount": PSEUDOCOUNT_ALPHA if deltas else None,
        "profile_score": B.PROFILE_SCORE_DEFINITION if deltas else None,
        "counts": counts,
        "samples": table,
    })
    protocol = B.manifest_protocol(plmc_record=plmc_record())
    protocol["max_sequences"] = cohort.max_sequences
    save(inputs / "support.json", {
        "schema": B.SCHEMA_SUPPORT,
        "cohort": cohort.name,
        "protocol": {key: protocol[key] for key in (
            "max_sequences", "coverage_floor", "reweight_identity", "neff_identity",
            "focus_weight")},
        "wildtypes": support,
    })
    save(inputs / "refusals.json", {
        "schema": "explicit_statistics_refusals_v1",
        "cohort": cohort.name,
        "rule": cohort.protocol["refusal_rule"],
        "count": len(refusals),
        "samples_without_support": sum(entry["samples"] for entry in refusals.values()),
        "refused": refusals,
    })
    lengths = [entry["length"] for entry in support.values()]
    manifest = {
        "schema": B.SCHEMA_MANIFEST,
        "cohort": cohort.name,
        "cohort_title": cohort.title,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": code_record(),
        "inputs": input_record(cohort),
        "protocol": protocol,
        "cohort_protocol": cohort.protocol,
        "counts": counts | {
            "backgrounds_exported": len(support),
            "backgrounds_refused": len(refusals),
            "backgrounds_reused": len(cohort.reuse),
            "backgrounds_profile_verified": len(verified),
            "backgrounds_at_depth_cap": len(saturated),
            "homolog_rows": int(sum(entry["rows"] for entry in support.values())),
            # A cohort whose every reference is refused exports no length at all.
            # It is an honest empty export, not a crash.
            "length_max": max(lengths, default=None),
            "length_median": int(np.median(lengths)) if lengths else None},
        "largest_profile_frequency_deviation": max(
            (entry["frequencies_max_absolute_deviation"] for entry in verified), default=None),
        "depth_cap_declaration": {
            "max_sequences": cohort.max_sequences,
            "rule": cohort.protocol["max_sequences_rule"],
            "search_max_target_seqs": cohort.search_max_target_seqs,
            "search_depth_limitation": cohort.protocol["search_depth_limitation"],
            "backgrounds_at_depth_cap": saturated,
            "profile_store_role": cohort.protocol["profile_store_role"],
            "profile_store_max_sequences": (
                cohort.profile_store.max_sequences if cohort.profile_store else None),
            "backgrounds_profile_verified": len(verified),
            "backgrounds_profile_not_comparable": len(support) - len(verified),
        },
    }
    save(inputs / "manifest.json", manifest)
    save(out / "contract.json", contract(cohort, manifest, refusals, out))
    log.write(f"export complete: {len(support)} exported, {len(refusals)} refused, "
              f"{len(cohort.reuse)} reused, {manifest['counts']['homolog_rows']} homolog rows")
    return manifest


def contract(cohort: C.Cohort, manifest: dict, refusals: dict, out: Path) -> dict:
    """The analysis contract, written before anything is fitted.

    It states what this cohort's explicit-statistics features are, the depth the
    support was built at, which references are refused and why, and the two
    limitations that are properties of the retained inputs rather than of this
    run. Nothing here is decided after a fit has been read.
    """

    return {
        "analysis": f"explicit_statistics_{cohort.name}_20261109",
        "cohort": cohort.name,
        "title": cohort.title,
        "purpose": "Make the explicit evolutionary statistics of the R1 ladder -- a plmc Potts "
                   "energy difference and an official GEMME score -- available for this cohort's "
                   "reference sequences under R1's own protocol, so that a later stage can add "
                   "the E and G control blocks to this lane.",
        "code_sha256": manifest["code"],
        "inputs": manifest["inputs"],
        "support": {
            "references": manifest["counts"]["wildtypes"],
            "exported": manifest["counts"]["backgrounds_exported"],
            "reused": manifest["counts"]["backgrounds_reused"],
            "refused": manifest["counts"]["backgrounds_refused"],
            "samples": manifest["counts"]["samples"],
            "single_substitution": manifest["counts"]["single_substitution"],
            "multiple_substitution": manifest["counts"]["multiple_substitution"],
            "assays": manifest["counts"]["assays"],
        },
        "depth_cap": manifest["depth_cap_declaration"],
        "potts": {
            "protocol": manifest["protocol"],
            "score": B.POTTS_SCORE_DEFINITION,
            "reuse_rule": "a reference whose sequence is byte-identical to an already fitted R1 "
                          "wild type takes that wild type's frozen alignment and its retained "
                          "parameter file; only this cohort's own mutation list is rescored, "
                          "because the retained per-variant scores cover other mutations",
            "cost_model": COST_MODEL,
            "memory_model": MEMORY_MODEL,
        },
        "gemme": {
            "runs_per_reference": 3,
            "aggregate": "arithmetic mean of the raw official combined score over three "
                         "independent complete official runs",
            "additivity_limitation": GEMME_ADDITIVITY,
        },
        "refusals": {
            "rule": cohort.protocol["refusal_rule"],
            "count": len(refusals),
            "refused": refusals,
        },
        "limitations": [
            cohort.protocol["search_depth_limitation"],
            cohort.protocol["profile_store_role"],
            GEMME_ADDITIVITY,
        ],
        "policy": "Frozen before any fit of this cohort is read. No reference is excluded on an "
                  "outcome, no refusal is imputed, and no score is read from a different depth "
                  "cap than the one declared here.",
        "resource": resource_record(out),
    }


# ------------------------------------------------------------------------ fit


def fit_job(args: argparse.Namespace, fit_out: Path, name: str, wildtype: str, support: dict,
            variants: list[dict]) -> dict:
    return {
        "wildtype_id": name,
        "wildtype": wildtype,
        "support": support,
        "variants": [{"sample_id": row["sample_id"],
                      "substitutions": [tuple(token) for token in row["substitutions"]]}
                     for row in variants],
        "msa": str(ROOT / support["msa"]),
        "weights": str(ROOT / support["weights"]),
        "msa_sha256": support["msa_sha256"],
        "weights_sha256": support["weights_sha256"],
        "plmc": str(ROOT / PLMC),
        "plmc_record": plmc_record(),
        "work": str(fit_out / "work" / name),
        "params": str(fit_out / "params" / f"{name}.params"),
        "threads": args.threads,
        "sweep_workers": args.sweep_workers,
    }


def fit_one(job: dict) -> dict:
    """Process-pool entry point: one wild type, one checkpoint."""

    return B.fit_background(job)


def reuse_params(name: str, support: dict, variants: list[dict], staged: Path,
                 log: Log) -> dict | None:
    """Score this cohort's mutation list with an already fitted model, or decline.

    A reference whose sequence is byte-identical to an already fitted R1 wild
    type needs no second fit: the alignment is that wild type's own, so the
    retained parameter file is exactly the model of it. What is *not* reusable is
    the retained per-variant score, because this cohort asks about other
    mutations; they are rescored here, which costs milliseconds.

    The staged file is accepted only at the digest the reused cohort recorded. If
    it is absent, this returns None and the caller fits the reference instead:
    declining to reuse is a slower run, reusing the wrong bytes is a wrong score.
    """

    recorded = support.get("params_available_for_reuse")
    source = support.get("support_source")
    if not recorded or not source:
        return None
    candidate = staged / Path(recorded["path"]).name
    if not candidate.is_file():
        log.write(f"reuse-miss {name}: {candidate} is not staged; it will be fitted")
        return None
    found = digest(candidate)
    if found != recorded["sha256"]:
        raise RuntimeError(
            f"{name}: {candidate} hashes {found}, not the {recorded['sha256']} that "
            f"{source['cohort']}/{source['wildtype_id']} recorded; it is not that model")
    if candidate.stat().st_size != recorded["bytes"]:
        raise RuntimeError(f"{name}: {candidate} is not {recorded['bytes']} bytes")

    from src.capability.stability.pairwise_stability import read_plmc

    wildtype = B.read_fasta(ROOT / support["msa"])[0][1]
    scorer = B.potts_scorer(read_plmc(candidate), wildtype)
    record = {
        "wildtype_id": name,
        "length": support["length"],
        "status": B.STATUS_FITTED,
        "support": support,
        "regularization_policy": None,
        "cv_qualified": False,
        "selected_lambda_e": None,
        "lambda_h": B.LAMBDA_H,
        "max_iter_caps": [int(cap) for cap in B.MAX_ITER_CAPS],
        "params": {"path": str(candidate), "sha256": found, "bytes": candidate.stat().st_size},
        "sign_check": B.shuffle_sign_check(scorer),
        "sample_scores": {
            row["sample_id"]: scorer.delta(B.mutant_codes(wildtype, row["substitutions"]))
            for row in variants},
    }
    reused = {key: source[key] for key in ("cohort", "wildtype_id", "msa_sha256")}
    record["reused_from"] = reused | {
        "params_sha256": found,
        "status": "the fit, its convergence and its selected regularization are the reused "
                  "cohort's; only this cohort's mutation list is scored here",
    }
    log.write(f"reuse-fit {name} from {reused['cohort']}/{reused['wildtype_id']} "
              f"variants={len(variants)}")
    return record


def checkpoint_path(fit_out: Path, name: str) -> Path:
    return fit_out / "backgrounds" / f"{name}.json"


def reuse_checkpoint(fit_out: Path, name: str, execution_sha: str, log: Log) -> dict:
    record = load(checkpoint_path(fit_out, name))
    if record.get("execution_manifest_sha256") != execution_sha:
        raise RuntimeError(
            f"{name}: the checkpoint was written under a different execution manifest; refit "
            "it instead of resuming onto it")
    if record["max_iter_caps"] != [int(cap) for cap in B.MAX_ITER_CAPS]:
        raise RuntimeError(
            f"{name}: the checkpoint used iteration caps {record['max_iter_caps']}, not "
            f"{list(B.MAX_ITER_CAPS)}")
    if record["status"] in B.SCOREABLE:
        params = Path(record["params"]["path"])
        if not params.is_file() or digest(params) != record["params"]["sha256"]:
            raise RuntimeError(f"{name}: the checkpoint's parameters are missing or altered")
    log.write(f"reuse {name} status={record['status']}")
    return record


def verify_export(support: dict, refused: dict, selected: Sequence[str], table: dict) -> None:
    """The derived export must still be the bytes the manifest describes."""

    named = {row["wildtype_id"] for row in table["samples"]}
    if named != set(support) | set(refused):
        raise RuntimeError(
            "the variant table names references the support table neither exports nor refuses")
    if table["counts"]["samples"] != len(table["samples"]):
        raise RuntimeError("the variant table does not hold the sample count it declares")
    for name in selected:
        entry = support[name]
        for key, path in (("msa_sha256", entry["msa"]), ("weights_sha256", entry["weights"])):
            if digest(ROOT / path) != entry[key]:
                raise RuntimeError(
                    f"{path} is not the export the manifest describes; rebuild the export")


def read_export(out: Path) -> tuple[dict, dict, dict]:
    table = load(out / "inputs" / "variant_table.json")
    support = load(out / "inputs" / "support.json")["wildtypes"]
    refusals = load(out / "inputs" / "refusals.json")["refused"]
    return table, support, refusals


def stage_fit(args: argparse.Namespace, log: Log) -> dict:
    out, fit_out = args.out, args.fit_out
    inputs = B.verify_inputs(out, root=ROOT)
    execution = B.load_or_create_execution(
        fit_out, inputs=inputs, code_paths={label: ROOT / path for label, path in CODE.items()},
        plmc_record=plmc_record())
    execution_sha = digest(fit_out / "execution_manifest.json")
    log.write(f"frozen inputs verified; execution manifest {execution_sha[:16]} "
              f"created_utc={execution['created_utc']}")
    table, support, refusals = read_export(out)
    by_wildtype: dict[str, list[dict]] = {}
    for row in table["samples"]:
        by_wildtype.setdefault(row["wildtype_id"], []).append(row)
    selected = sorted(support) if not args.wildtypes else list(args.wildtypes)
    unknown = [name for name in selected if name not in support]
    if unknown:
        raise SystemExit(f"references with no exported support: {unknown[:5]}")
    if len(set(selected)) != len(selected):
        raise SystemExit("duplicate references requested")
    # One fit's memory is set by its reference's length alone, so a length band
    # is how a batch is admitted against the host's memory rather than against
    # its core count. The band is part of the run record, not a convenience.
    banded = [name for name in selected
              if args.min_length < support[name]["length"] <= args.max_length]
    if not banded:
        raise SystemExit(
            f"no exported reference has a length in ({args.min_length}, {args.max_length}]")
    if len(banded) != len(selected):
        log.write(f"length band ({args.min_length}, {args.max_length}] admits {len(banded)} of "
                  f"{len(selected)} requested references")
    selected = banded
    verify_export(support, refusals, selected, table)
    log.write(f"export verified for {len(selected)} of {len(support)} exported references "
              f"({len(refusals)} refused)")

    resource_check(fit_out, lengths=[support[name]["length"] for name in selected],
                   workers=args.workers, log=log)
    staged = None if args.reuse_params is None else (
        args.reuse_params if args.reuse_params.is_absolute() else ROOT / args.reuse_params)
    records: dict[str, dict] = {}
    jobs = []
    for name in selected:
        if checkpoint_path(fit_out, name).exists():
            if not args.resume:
                raise FileExistsError(
                    f"{checkpoint_path(fit_out, name)} exists; pass --resume to reuse "
                    "checkpoints whose execution manifest still matches")
            records[name] = reuse_checkpoint(fit_out, name, execution_sha, log)
            continue
        wildtype = B.read_fasta(ROOT / support[name]["msa"])[0][1]
        reused = None if staged is None else reuse_params(
            name, support[name], by_wildtype[name], staged, log)
        if reused is not None:
            reused["execution_manifest_sha256"] = execution_sha
            save(checkpoint_path(fit_out, name), reused)
            records[name] = reused
            continue
        jobs.append(fit_job(args, fit_out, name, wildtype, support[name], by_wildtype[name]))
    if jobs:
        log.write(f"fitting {len(jobs)} references with {args.workers} workers x "
                  f"{args.threads} plmc threads, caps {list(B.MAX_ITER_CAPS)}")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fit_one, job): job["wildtype_id"] for job in jobs}
            for future in as_completed(futures):
                record = future.result()
                record["execution_manifest_sha256"] = execution_sha
                save(checkpoint_path(fit_out, record["wildtype_id"]), record)
                records[record["wildtype_id"]] = record
                log.write(progress_line(record))
    summary = assemble(args, log)
    summary["peak_child_rss_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    log.write(f"peak child RSS {summary['peak_child_rss_bytes'] / 2**30:.2f} GiB")
    return summary


def progress_line(record: dict) -> str:
    final = record.get("final") or {}
    attempts = len(final.get("attempts", [])) + sum(
        len(entry.get("attempts", [])) for entry in record.get("sweep", []))
    return (f"fit {record['wildtype_id']} status={record['status']} "
            f"policy={record['regularization_policy']} "
            f"lambda_e={record['selected_lambda_e']} "
            f"gain={record.get('held_out_gain_over_independent_site')} "
            f"attempts={attempts} cap={final.get('max_iter')} "
            f"iterations={final.get('iterations')} "
            f"seconds={record.get('final_seconds')} converged={final.get('converged')}")


# ------------------------------------------------------------------- assembly


def assemble(args: argparse.Namespace, log: Log) -> dict:
    """Rewrite the unified tables from whatever checkpoints exist."""

    out, fit_out = args.out, args.fit_out
    inputs = B.verify_inputs(out, root=ROOT)
    execution = B.load_or_create_execution(
        fit_out, inputs=inputs, code_paths={label: ROOT / path for label, path in CODE.items()},
        plmc_record=plmc_record())
    table, support, refusals = read_export(out)
    records: dict[str, dict] = {}
    for name in sorted(support):
        path = checkpoint_path(fit_out, name)
        if not path.exists():
            continue
        record = load(path)
        if record.get("execution_manifest_sha256") != digest(
                fit_out / "execution_manifest.json"):
            raise RuntimeError(
                f"{name}: the checkpoint was written under a different execution manifest; it "
                "cannot enter a table presented as this execution's result")
        records[name] = record
    document, arrays = B.assemble_scores(
        table["samples"], records, wildtype_ids=sorted(support), refused=refusals,
        protocol=execution["optimization"],
        inputs={"inputs_manifest_sha256": inputs["manifest"]["sha256"],
                "variant_table_sha256": inputs["derived"]["variant_table"]["sha256"],
                "support_sha256": inputs["derived"]["support"]["sha256"],
                "source": {label: entry["sha256"]
                           for label, entry in inputs["source"].items()}})
    document["cohort"] = table.get("cohort")
    document["execution_manifest_sha256"] = digest(fit_out / "execution_manifest.json")
    document["counts"]["cost"] = cost_projection(records, support)
    save(fit_out / "scores.json", document)
    np.savez_compressed(fit_out / "scores.npz", **arrays)
    save(fit_out / "completion.json",
         completion(fit_out, document, records, support, refusals))
    log.write(f"assembly {json.dumps(document['counts'], default=str)}")
    return document


def completion(fit_out: Path, document: dict, records: dict[str, dict], support: dict,
               refusals: dict) -> dict:
    """What this run delivered, stated last and only from what exists on disk."""

    counts = document["counts"]
    return {
        "status": "complete" if counts["complete"] else "incomplete",
        "cohort": document.get("cohort"),
        "rung": "potts",
        "counts": counts,
        "refused": {name: entry["reason"] for name, entry in sorted(refusals.items())},
        "reused": {name: records[name]["reused_from"] for name in sorted(records)
                   if "reused_from" in records[name]},
        "pending": [name for name in sorted(support) if name not in records],
        "output_sha256": {name: digest(fit_out / name)
                          for name in ("scores.json", "scores.npz")
                          if (fit_out / name).is_file()},
    }


def cost_projection(records: dict[str, dict], support: dict) -> dict:
    """What this cohort's fits cost, and what the ones still to run will cost.

    The projection uses :data:`COST_MODEL`, fitted to the completed R1 run's own
    per-attempt records, rather than a single ``rows * length**2`` constant: that
    constant underpredicted R1's measured 390.9 wall-hours by a factor of 3.7,
    because the cost grows faster than linearly in the alignment width and
    sublinearly in its depth, and no single product of the two reproduces both
    ends of the cohort. It remains a projection and is labelled as one.
    """

    measured, seconds, attempts = [], 0.0, 0
    for name, record in records.items():
        if name not in support or "reused_from" in record:
            continue
        entries = list(record.get("sweep", [])) + (
            [record["final"]] if "final" in record else [])
        total = sum(entry.get("seconds") or 0.0 for attempt in entries
                    for entry in attempt.get("attempts", []))
        if not total:
            continue
        attempts += sum(len(entry.get("attempts", [])) for entry in entries)
        seconds += total
        measured.append({
            "wildtype_id": name, "length": support[name]["length"],
            "rows": support[name]["rows"], "seconds": total,
            "predicted_seconds": model_seconds(support[name], record.get("regularization_policy")),
        })
    remaining = [name for name in support if name not in records]
    projected = sum(model_seconds(support[name], None) for name in remaining)
    ratios = [entry["seconds"] / entry["predicted_seconds"] for entry in measured
              if entry["predicted_seconds"]]
    return {
        "model": COST_MODEL,
        "measured_backgrounds": len(measured),
        "measured_seconds": seconds,
        "measured_fit_attempts": attempts,
        "measured_over_predicted_median": float(np.median(ratios)) if ratios else None,
        "remaining_backgrounds": len(remaining),
        "projected_seconds_remaining": projected,
        "projected_hours_remaining_single_worker": projected / 3600.0,
        "note": "projected at the model's 4 plmc threads per fit; a worker runs one background "
                "at a time and the lambda sweep inside one background is sequential, so the "
                "longest single background is a wall-clock floor no concurrency removes",
    }


def model_seconds(entry: dict, policy: str | None) -> float:
    """One background's projected fit seconds under :data:`COST_MODEL`.

    Without a measured policy the branch is taken from the support itself, by the
    same floors the fit uses: a background that cannot hold out rows is fitted
    once at the pinned regularization and never sweeps.
    """

    rows, length = entry["rows"], entry["length"]
    if policy is None:
        policy = (B.POLICY_CROSS_VALIDATED
                  if rows >= 2 * (B.MIN_TRAIN_ROWS + B.MIN_HELD_OUT_ROWS)
                  else B.POLICY_FIXED_LOW_DEPTH)
    model = COST_MODEL.get(policy) or COST_MODEL[B.POLICY_CROSS_VALIDATED]
    return model["constant"] * length ** model["length"] * max(rows, 1) ** model["rows"]


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("export", "fit", "assemble"))
    parser.add_argument("--cohort", default=DEFAULT_COHORT,
                        help=f"a cohort declared in {CONFIG} (default {DEFAULT_COHORT})")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=None,
                        help="the export directory holding inputs/; defaults to the cohort's own")
    parser.add_argument("--fit-out", type=Path, default=None,
                        help="the fit tree; defaults to the cohort's own subdirectory of <out>")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4, help="plmc OpenMP threads per fit")
    parser.add_argument("--sweep-workers", type=int, default=1,
                        help="run this many of one background's five independent lambda fits at "
                             "once; the default 1 is the order every existing fit tree was "
                             "produced with. One worker then holds sweep_workers * threads "
                             "threads and sweep_workers fits' memory, so lower --workers to match")
    parser.add_argument("--resume", action="store_true",
                        help="reuse per-wild-type checkpoints whose execution manifest still "
                             "matches")
    parser.add_argument("--reuse-params", type=Path, default=None,
                        help="a directory of staged plmc parameter files of the reused cohort; "
                             "a reference whose export names a reusable fit is scored from the "
                             "staged file at its recorded digest instead of being refitted")
    parser.add_argument("--wildtypes", nargs="*", default=None,
                        help="fit only these references; assembly marks the rest pending")
    parser.add_argument("--min-length", type=int, default=0,
                        help="fit only references longer than this; one fit's memory is set by "
                             "its length, so a band is how concurrency is admitted")
    parser.add_argument("--max-length", type=int, default=1 << 30,
                        help="fit only references of at most this length")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        raise SystemExit("workers and threads must be positive")
    if args.min_length >= args.max_length:
        raise SystemExit("--min-length must be below --max-length")
    if not 1 <= args.sweep_workers <= len(B.LAMBDA_E_GRID):
        raise SystemExit(f"--sweep-workers must lie in 1..{len(B.LAMBDA_E_GRID)}")
    descriptor = C.load_config(
        args.config if args.config.is_absolute() else ROOT / args.config)["cohorts"]
    if args.cohort not in descriptor:
        raise SystemExit(f"unknown cohort {args.cohort!r}; declared: {sorted(descriptor)}")
    declared = descriptor[args.cohort]
    args.out = (ROOT / declared["out"] if args.out is None
                else (args.out if args.out.is_absolute() else ROOT / args.out))
    args.fit_out = (args.out / declared["fit_subdir"] if args.fit_out is None
                    else (args.fit_out if args.fit_out.is_absolute() else ROOT / args.fit_out))
    args.log_dir = (ROOT / declared["log_dir"] if args.log_dir is None
                    else (args.log_dir if args.log_dir.is_absolute() else ROOT / args.log_dir))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = Log(args.log_dir / f"{args.cohort}_{args.stage}_{stamp}.log")
    log.write(f"cohort={args.cohort} stage={args.stage} out={args.out} fit_out={args.fit_out} "
              f"workers={args.workers} threads={args.threads} resume={args.resume} "
              f"reuse_params={args.reuse_params} sweep_workers={args.sweep_workers} "
              f"length=({args.min_length}, {args.max_length}] caps={list(B.MAX_ITER_CAPS)}")
    if args.stage == "export":
        stage_export(args, log)
    elif args.stage == "fit":
        stage_fit(args, log)
    else:
        assemble(args, log)


if __name__ == "__main__":
    main()
