#!/usr/bin/env python3
"""Export and fit the R1 homologous-support statistical ladder.

Three stages, in order, each refusing to overwrite the previous one's evidence:

``export``
    Validate the frozen 25,728-sample R1 table against the measured cohort and
    the frozen identifier lineage, rebuild every wild type's homologous support,
    prove it is the support the retained profile store was built from, and write
    the query-first alignment, the original weights, the variant table (with the
    recomputed LOOKUP delta) and the support statistics under ``inputs/``.

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

from src.capability.baselines import msa_baselines as B  # noqa: E402
from src.capability.context.homology import ALIGNMENT_FIELDS, parse_hits  # noqa: E402
from src.capability.context.profiles import AA20, PSEUDOCOUNT_ALPHA  # noqa: E402

SAMPLES = Path("results/extensions/information_progression_20261006/full/samples.json")
SUPPORT = Path("results/R5/retrieval_bound_20260807")
COHORT = Path("results/R5/context_mutation_rescue_20260923/cohort.json")
ADMISSION = Path(
    "results/R5/local_context_20260923/20260923233257_e429ce7f31e4/lcgp_admission/"
    "local_context_measurement.json")
PLMC = Path("data/pairwise_assets/plmc/debbiemarkslab-plmc-18c9e55/bin/plmc")
DEFAULT_OUT = Path("results/extensions/statistical_ladder_20261106")
DEFAULT_LOG = Path("logs/operations/statistical-ladder-20261106")

#: The frozen R1 table's shape. A stage that produced a different table would be
#: measuring a different cohort under the same identifiers, so it stops.
EXPECTED_SAMPLES = 25728
EXPECTED_ASSAYS = 201
EXPECTED_SINGLE_SUBSTITUTION = 20568
EXPECTED_MULTIPLE_SUBSTITUTION = 5160

#: Code whose change would invalidate an existing export or checkpoint.
CODE = {
    "msa_baselines": Path("src/capability/baselines/msa_baselines.py"),
    "entrypoint": Path("scripts/capability/baselines/r1_msa_baselines.py"),
    "pairwise_stability": Path("src/capability/stability/pairwise_stability.py"),
    "sequence_alignment": Path("src/capability/core/sequence_alignment.py"),
    "profiles": Path("src/capability/context/profiles.py"),
}

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


def input_record() -> dict:
    paths = {
        "samples": SAMPLES,
        "corpus_hits": SUPPORT / "corpus_hits.tsv",
        "wildtypes_fasta": SUPPORT / "wildtypes.faa",
        "wildtypes": SUPPORT / "wildtypes.json",
        "profiles_npz": SUPPORT / "profiles.npz",
        "profiles": SUPPORT / "profiles.json",
        "lookup": SUPPORT / "lookup.json",
        "cohort": COHORT,
        "admission": ADMISSION,
        "plmc": PLMC,
    }
    return {label: {"path": str(path), "sha256": digest(ROOT / path)}
            for label, path in paths.items()}


# --------------------------------------------------------------------- export


def wildtype_sequences() -> dict[str, str]:
    sequences: dict[str, str] = {}
    for name, sequence in B.read_fasta(ROOT / SUPPORT / "wildtypes.faa"):
        if name in sequences:
            raise ValueError(f"duplicate wild-type record {name}")
        sequences[name] = sequence
    return sequences


def cohort_by_assay() -> dict:
    cohort = load(ROOT / COHORT)
    assays = {row["assay"]: row for row in cohort["assays"]}
    if len(assays) != len(cohort["assays"]):
        raise ValueError("duplicate cohort assays")
    return cohort | {"assays": assays}


def stage_export(args: argparse.Namespace, log: Log) -> dict:
    out = args.out
    if (out / "inputs").exists():
        raise FileExistsError(
            f"{out / 'inputs'} exists; the export is written once. Remove that directory "
            "deliberately if the inputs genuinely changed.")
    resource_check(out, lengths=[1], workers=args.workers, log=log)

    samples = load(ROOT / SAMPLES)
    catalogue = load(ROOT / SUPPORT / "wildtypes.json")
    profiles = load(ROOT / SUPPORT / "profiles.json")
    lookup = load(ROOT / SUPPORT / "lookup.json")
    cohort = cohort_by_assay()
    admission = load(ROOT / ADMISSION)
    sequences = wildtype_sequences()
    stored = np.load(ROOT / SUPPORT / "profiles.npz", allow_pickle=False)
    background = np.array([catalogue["corpus"]["background"][residue] for residue in AA20],
                          dtype=np.float64)
    if not np.isclose(float(lookup["settings"]["alpha"]), PSEUDOCOUNT_ALPHA):
        raise ValueError("the retained lookup alpha is not the declared profile pseudocount")

    from scripts.capability.reporting.export_prediction_details import sample_id as frozen_id
    cohort_sha = admission["cohort_sha256"]
    variant_table = B.build_variant_table(
        samples, wildtype_of_assay=catalogue["assay_to_wildtype"], sequences=sequences,
        cohort=cohort, sample_id_of=lambda assay, mutation: frozen_id(cohort_sha, assay, mutation))
    counts = B.substitution_counts(variant_table)
    log.write(f"variant table {counts}")
    if (counts["samples"], counts["single_substitution"], counts["multiple_substitution"]) != (
            EXPECTED_SAMPLES, EXPECTED_SINGLE_SUBSTITUTION, EXPECTED_MULTIPLE_SUBSTITUTION):
        raise ValueError(f"the frozen R1 table's shape changed: {counts}")
    assays = sorted({row["assay"] for row in variant_table})
    wildtype_ids = sorted({row["wildtype_id"] for row in variant_table})
    if len(assays) != EXPECTED_ASSAYS:
        raise ValueError(f"{len(assays)} assays, expected {EXPECTED_ASSAYS}")
    for assay in assays:
        record = cohort["assays"][assay]
        if record["wildtype"] != sequences[record["wildtype_id"]]:
            raise ValueError(f"{assay}: cohort wild type differs from the retained FASTA")
    log.write(f"{len(assays)} assays over {len(wildtype_ids)} wild types")

    hits: dict[str, list] = {name: [] for name in wildtype_ids}
    for hit in parse_hits(ROOT / SUPPORT / "corpus_hits.tsv", fields=ALIGNMENT_FIELDS):
        if hit.query in hits:
            hits[hit.query].append(hit)
    if any(not value for value in hits.values()):
        raise ValueError("a wild type in the R1 support has no alignment hit")

    inputs = out / "inputs"
    (inputs / "msa").mkdir(parents=True, exist_ok=True)
    support_by_wildtype: dict[str, dict] = {}
    by_wildtype: dict[str, list[dict]] = {}
    for row in variant_table:
        by_wildtype.setdefault(row["wildtype_id"], []).append(row)
    profile_scores = {assay: row["profile_scores"] for assay, row in cohort["assays"].items()}
    profile_deltas: dict[str, dict[str, float]] = {}
    for name in wildtype_ids:
        wildtype = sequences[name]
        metadata = profiles["profiles"][name]
        rows, weights, _, statistics = B.reconstruct_support(
            wildtype, name, hits[name], frequencies=stored[name], record=metadata)
        fasta, weights_path = inputs / "msa" / f"{name}.fa", inputs / "msa" / f"{name}.weights"
        B.write_query_first_alignment(fasta, weights_path, wildtype, rows, weights)
        variants = [tuple(tuple(token) for token in row["substitutions"]) for row in by_wildtype[name]]
        deltas = B.profile_deltas(np.asarray(stored[name], dtype=np.float64), background, variants)
        profile_deltas[name] = dict(zip([row["sample_id"] for row in by_wildtype[name]], deltas))
        support_by_wildtype[name] = {
            "wildtype_id": name,
            "length": len(wildtype),
            "assays": sorted({row["assay"] for row in by_wildtype[name]}),
            "samples": len(by_wildtype[name]),
            "hits_retrieved": len(hits[name]),
            "hit_list_saturated": metadata["hit_list_saturated"],
            "diamond_hit_list_saturated": metadata["diamond_hit_list_saturated"],
            "max_identity_over_query": metadata["max_identity_over_query"],
            "msa": str(fasta.relative_to(ROOT)),
            "msa_sha256": digest(fasta),
            "weights": str(weights_path.relative_to(ROOT)),
            "weights_sha256": digest(weights_path),
            **statistics,
        }
        log.write(f"support {name} L={len(wildtype)} rows={statistics['rows']} "
                  f"neff={statistics['neff_at_30_percent_identity']:.3f} "
                  f"deviation={statistics['frequencies_max_absolute_deviation']:.2e}")
    if not all(entry["frequencies_verified"] for entry in support_by_wildtype.values()):
        raise RuntimeError("a rebuilt alignment does not reproduce the retained profile")

    for row in variant_table:
        row["profile_delta"] = profile_deltas[row["wildtype_id"]][row["sample_id"]]
        retained = profile_scores[row["assay"]][row["variant_index"]]
        if abs(row["profile_delta"] - retained) > 1e-12:
            raise RuntimeError(
                f"{row['sample_id']}: recomputed LOOKUP score {row['profile_delta']!r} is not the "
                f"retained cohort's {retained!r}")
    log.write(f"profile deltas reproduce {len(variant_table)} retained cohort profile scores")

    save(inputs / "variant_table.json", {
        "schema": "r1_msa_variant_table_v1",
        "source": str(SAMPLES),
        "source_sha256": digest(ROOT / SAMPLES),
        "profile_pseudocount": PSEUDOCOUNT_ALPHA,
        "profile_score": B.PROFILE_SCORE_DEFINITION,
        "counts": counts | {"assays": len(assays), "wildtypes": len(wildtype_ids)},
        "samples": variant_table,
    })
    protocol = B.manifest_protocol(plmc_record=plmc_record())
    save(inputs / "support.json", {
        "schema": B.SCHEMA_SUPPORT,
        "protocol": {key: protocol[key] for key in (
            "max_sequences", "coverage_floor", "reweight_identity", "neff_identity", "focus_weight")},
        "wildtypes": support_by_wildtype,
    })
    lengths = [entry["length"] for entry in support_by_wildtype.values()]
    driver = sum(entry["rows"] * entry["length"] ** 2 for entry in support_by_wildtype.values())
    manifest = {
        "schema": B.SCHEMA_MANIFEST,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": code_record(),
        "inputs": input_record(),
        "protocol": protocol,
        "cohort_sha256": cohort_sha,
        "counts": counts | {
            "assays": len(assays), "wildtypes": len(wildtype_ids),
            "homolog_rows": int(sum(entry["rows"] for entry in support_by_wildtype.values())),
            "length_max": max(lengths), "length_median": int(np.median(lengths)),
            "fit_cost_driver_rows_times_length_squared": int(driver)},
        "largest_profile_frequency_deviation": max(
            entry["frequencies_max_absolute_deviation"] for entry in support_by_wildtype.values()),
    }
    save(inputs / "manifest.json", manifest)
    log.write(f"export complete: {len(wildtype_ids)} wild types, "
              f"{manifest['counts']['homolog_rows']} homolog rows, "
              f"max profile deviation {manifest['largest_profile_frequency_deviation']:.2e}")
    return manifest


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
    }


def fit_one(job: dict) -> dict:
    """Process-pool entry point: one wild type, one checkpoint."""

    return B.fit_background(job)


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


def verify_export(support: dict, selected: Sequence[str], table: dict) -> None:
    """The derived export must still be the bytes the manifest describes."""

    wildtypes = {row["wildtype_id"] for row in table["samples"]}
    if wildtypes != set(support):
        raise RuntimeError("the variant table and the support table name different wild types")
    for name in selected:
        entry = support[name]
        for key, path in (("msa_sha256", entry["msa"]), ("weights_sha256", entry["weights"])):
            if digest(ROOT / path) != entry[key]:
                raise RuntimeError(
                    f"{path} is not the export the manifest describes; rebuild the export")
    if len(table["samples"]) != EXPECTED_SAMPLES:
        raise RuntimeError("the variant table does not hold the frozen sample count")


def stage_fit(args: argparse.Namespace, log: Log) -> dict:
    out, fit_out = args.out, args.fit_out
    inputs = B.verify_inputs(out, root=ROOT)
    execution = B.load_or_create_execution(
        fit_out, inputs=inputs, code_paths={label: ROOT / path for label, path in CODE.items()},
        plmc_record=plmc_record())
    execution_sha = digest(fit_out / "execution_manifest.json")
    log.write(f"frozen inputs verified; execution manifest {execution_sha[:16]} "
              f"created_utc={execution['created_utc']}")
    table = load(out / "inputs" / "variant_table.json")
    support = load(out / "inputs" / "support.json")["wildtypes"]
    sequences = wildtype_sequences()
    by_wildtype: dict[str, list[dict]] = {}
    for row in table["samples"]:
        by_wildtype.setdefault(row["wildtype_id"], []).append(row)
    selected = sorted(by_wildtype) if not args.wildtypes else list(args.wildtypes)
    unknown = [name for name in selected if name not in by_wildtype]
    if unknown:
        raise SystemExit(f"unknown wild types: {unknown[:5]}")
    if len(set(selected)) != len(selected):
        raise SystemExit("duplicate wild types requested")
    verify_export(support, selected, table)
    log.write(f"export verified for {len(selected)} wild types")

    resource_check(fit_out, lengths=[support[name]["length"] for name in selected],
                   workers=args.workers, log=log)
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
        jobs.append(fit_job(args, fit_out, name, sequences[name], support[name],
                            by_wildtype[name]))
    if jobs:
        log.write(f"fitting {len(jobs)} wild types with {args.workers} workers x "
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
    table = load(out / "inputs" / "variant_table.json")["samples"]
    support = load(out / "inputs" / "support.json")["wildtypes"]
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
        table, records, wildtype_ids=sorted(support), protocol=execution["optimization"],
        inputs={"inputs_manifest_sha256": inputs["manifest"]["sha256"],
                "variant_table_sha256": inputs["derived"]["variant_table"]["sha256"],
                "support_sha256": inputs["derived"]["support"]["sha256"],
                "source": {label: entry["sha256"]
                           for label, entry in inputs["source"].items()}})
    document["execution_manifest_sha256"] = digest(fit_out / "execution_manifest.json")
    document["counts"]["cost"] = cost_projection(records, support)
    save(fit_out / "scores.json", document)
    np.savez_compressed(fit_out / "scores.npz", **arrays)
    log.write(f"assembly {json.dumps(document['counts'], default=str)}")
    return document


def cost_projection(records: dict[str, dict], support: dict) -> dict:
    """Measured cost per fit, projected onto the backgrounds that have not run.

    plmc's iteration cost grows with the number of sequences and with the square
    of the alignment width, so one fit's driver is ``rows * length**2``. The
    constant is measured on the fits that have run; the projection counts the
    fits each remaining background will need (six when it can hold out, one when
    it is fitted at the fixed low-depth regularization) and is labelled as a
    projection rather than a measurement.
    """

    measured = []
    for name, record in records.items():
        if name not in support or not record.get("final_seconds"):
            continue
        fits = len(record.get("final", {}).get("attempts", [])) + sum(
            len(entry.get("attempts", [])) for entry in record.get("sweep", []))
        driver = support[name]["rows"] * support[name]["length"] ** 2
        seconds = record["final_seconds"] + sum(entry.get("seconds") or 0.0
                                                for entry in record.get("sweep", []))
        measured.append((seconds, fits, driver, name))
    if not measured:
        return {"measured_backgrounds": 0}
    # Ratio of sums rather than a median of per-background constants: with one or
    # two measured backgrounds a per-background median is dominated by whichever
    # tiny fit happened to run first.
    per_fit_driver = sum(entry[0] for entry in measured) / sum(
        entry[1] * entry[2] for entry in measured)
    remaining, projected = [], 0.0
    for name in support:
        if name in records:
            continue
        # A background that cannot clear the held-out floor is fitted once; one
        # that might is budgeted at the full six-fit sweep.
        fits = 1 if support[name]["rows"] < 2 * (B.MIN_TRAIN_ROWS + B.MIN_HELD_OUT_ROWS) else 6
        driver = support[name]["rows"] * support[name]["length"] ** 2
        projected += fits * driver
        remaining.append(name)
    return {
        "measured_backgrounds": len(measured),
        "measured_seconds": float(sum(entry[0] for entry in measured)),
        "measured_fits": int(sum(entry[1] for entry in measured)),
        "seconds_per_fit_rows_length_squared": per_fit_driver,
        "remaining_backgrounds": len(remaining),
        "remaining_fit_driver": float(projected),
        "projected_seconds_remaining": per_fit_driver * projected,
        "projected_hours_remaining_single_worker": per_fit_driver * projected / 3600.0,
        "driver": "rows * length**2 per fit; constant is the measured ratio of total seconds "
                  "to total fit-driver, extrapolated, not a measurement of the long alignments",
    }


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("export", "fit", "assemble"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="the frozen export directory holding inputs/")
    parser.add_argument("--fit-out", type=Path, default=None,
                        help="the fit tree; defaults to <out>/potts-v2")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4, help="plmc OpenMP threads per fit")
    parser.add_argument("--resume", action="store_true",
                        help="reuse per-wild-type checkpoints whose execution manifest still "
                             "matches")
    parser.add_argument("--wildtypes", nargs="*", default=None,
                        help="fit only these wild types; assembly marks the rest pending")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        raise SystemExit("workers and threads must be positive")
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    default_fit = args.out / "potts-v2"
    args.fit_out = default_fit if args.fit_out is None else (
        args.fit_out if args.fit_out.is_absolute() else ROOT / args.fit_out)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = Log(ROOT / args.log_dir / f"{args.stage}_{stamp}.log")
    log.write(f"stage={args.stage} out={args.out} fit_out={args.fit_out} "
              f"workers={args.workers} threads={args.threads} resume={args.resume} "
              f"caps={list(B.MAX_ITER_CAPS)}")
    if args.stage == "export":
        stage_export(args, log)
    elif args.stage == "fit":
        stage_fit(args, log)
    else:
        assemble(args, log)


if __name__ == "__main__":
    main()
