#!/usr/bin/env python3
"""The GEMME rung of the R1 statistical ladder: three official runs per wild type.

    GEMME is stochastic and exposes no seed, so one run is not a score: the same
    alignment and mutant list moved single-mutation values by up to 11.5 % in the
    tool smoke. The ladder's decision is therefore a fixed three independent
    complete official runs per wild type, the arithmetic mean of the raw combined
    scores, and no selection by phenotype and no top-up: the number of runs is a
    property of the protocol, not of the result.

    Three consequences are enforced rather than documented. A wild type is
    *complete* only when all three runs completed and each produced its own
    official output file; a partial mean is never emitted, so a failed repeat
    removes the wild type from the score table instead of shrinking its sample
    size. A wild type whose every mutation scored 0.0 in every run is *refused*
    rather than scored: JET2 placed no position, so the tool returned no
    estimate, and a column of zeros averaged into a feature would read
    "neutral everywhere" off a tool that said nothing. The raw runs are kept as
    the evidence of the refusal and the score itself is null. And the three runs are three draws of one protocol, not biological
    replicates: their spread is reported per variant as a standard deviation and
    as run-to-run rank agreement, which is what the caller needs to know before
    reading a small difference, and nothing in this module calls them replicates.

    What is scored. The official ``-m`` entry point, once per wild type, over the
    distinct mutation strings that wild type's assays ask for -- a mutation
    shared by two assays of the same wild type is scored once and mapped back to
    every ``sample_id`` that names it, so the deduplication is of the computation
    and never of the samples. Multi-mutants go to GEMME as multi-mutants.

    Reading the score. The value is GEMME's normalised predicted effect: a global
    evolutionary-background estimate, higher meaning less deleterious. For a
    multi-mutant the official code sums the per-site terms, so the output is
    additive over the mutation's sites and is **not** a non-additive epistasis
    prediction; the value's magnitude grows with the number of changed sites and
    is not comparable in size with a single substitution's. That is the official
    semantics, reported here rather than repaired.

    Inputs are the export this repository already produces for the ladder
    (``inputs/manifest.json``, ``inputs/support.json``, ``inputs/variant_table.json``
    and the per-wild-type alignments under ``inputs/msa``); this stage rebuilds no
    alignment. That export layout is cohort-neutral, so ``--root`` is the only
    thing that changes between the R1 cohort and any other one the export
    entry point produces. Every alignment is hash-checked against the export before it is
    run, and every run is laired in its own new directory with its own record, so
    a resumed stage can only ever reuse runs whose inputs, parameters and image
    digest hash identically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.capability.baselines import gemme_runtime as G  # noqa: E402
from src.capability.baselines import cohorts as C  # noqa: E402
from src.capability.baselines import msa_baselines as B  # noqa: E402

SCHEMA_CHECKPOINT = "r1_gemme_wildtype"
SCHEMA_SCORES = "r1_gemme_scores"

#: ``run_key`` is a protocol identity, not a label. The 174x3 official runs that already
#: exist hashed these recorded spellings into their keys, and a completed GEMME run cannot
#: be reproduced (JET2 samples cluster partitions and exposes no seed), so renaming a label
#: must never move a key. They stay in the key material only: the checkpoint and the score
#: table carry the stable IDs above, and old and new checkpoints share one key space.
RUN_KEY_SCHEMA = "r1_gemme_wildtype_v2"
RUN_KEY_RUNTIME = "gemme_runtime_v1"

#: The export's own variant-table schema tag, read from the module that defines
#: it; checking it is what stops this stage reading a table of a different shape.
SCHEMA_VARIANT_TABLE = B.SCHEMA_VARIANT_TABLE

#: Fixed by protocol. Not a flag: a caller who could choose the number of runs
#: could choose the one that agreed with their expectation.
REPEATS = 3
MAX_WORKERS = 4

DEFAULT_ROOT = Path("results/extensions/statistical_ladder_20261106")
DEFAULT_LOG = Path("logs/operations/statistical-ladder-20261106")

AGGREGATE = (
    "arithmetic mean of the raw official combined score over three independent complete "
    "official runs; sd is the sample standard deviation (ddof=1) over those three values. "
    "The three runs are draws of one protocol, not biological replicates"
)

MULTI_MUTANT_NOTE = (
    "official GEMME semantics: the score is a global evolutionary-background estimate and, for "
    "a multi-mutant, the official -m output sums the per-site normalised predicted effects. The "
    "multi-mutant value is therefore ADDITIVE over its sites and is not a non-additive epistasis "
    "prediction; its magnitude grows with the number of changed sites and is not comparable in "
    "size with a single substitution's. This is the official entry point's own behaviour, used "
    "as published and not replaced by a surrogate"
)


class Log:
    """Append-only run log; the durable record of a killed stage."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.write(f"# {path.name}")

    def write(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def save(path: Path, value: Any) -> None:
    """Atomic JSON publication; a half-written table is never left behind."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    return G.sha256_file(Path(path))


def display(path: Path) -> str:
    """Repository-relative when the path is inside it, absolute otherwise."""

    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------- the inputs


def verify_export(root: Path) -> dict[str, Any]:
    """The export this stage consumes, and proof it is the export it was given.

    Only what GEMME actually depends on is checked: the exported alignments (hash-
    checked per wild type when they are run) and the manifest's own record of the
    source tables. The export's plmc protocol is irrelevant here because no
    coupling is fitted, so it is recorded rather than re-verified.
    """

    inputs = root / "inputs"
    manifest = load(inputs / "manifest.json")
    if manifest.get("schema") != B.SCHEMA_MANIFEST:
        raise ValueError(f"{inputs / 'manifest.json'}: schema {manifest.get('schema')!r} is not {B.SCHEMA_MANIFEST!r}")
    drift = []
    for label, entry in manifest["inputs"].items():
        path = Path(entry["path"])
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            drift.append(f"input {label}: {path} is missing")
        elif C.path_digest(path) != entry["sha256"]:
            # ``path_digest`` is the export's own rule, so a cohort defined by a
            # directory of measured rows is bound here exactly as a file is.
            drift.append(f"input {label}: {path} changed since the export")
    if drift:
        raise RuntimeError("the export manifest does not describe these files -- " + "; ".join(drift))

    support = load(inputs / "support.json")
    if support.get("schema") != B.SCHEMA_SUPPORT:
        raise ValueError(f"{inputs / 'support.json'}: schema {support.get('schema')!r} is not {B.SCHEMA_SUPPORT!r}")
    table = load(inputs / "variant_table.json")
    if table.get("schema") != SCHEMA_VARIANT_TABLE:
        raise ValueError(
            f"{inputs / 'variant_table.json'}: schema {table.get('schema')!r} is not "
            f"{SCHEMA_VARIANT_TABLE!r}")
    return {
        "manifest": manifest,
        "support": support,
        "table": table,
        "files": {
            label: {"path": display(inputs / name), "sha256": sha256_file(inputs / name)}
            for label, name in (
                ("manifest", "manifest.json"),
                ("support", "support.json"),
                ("variant_table", "variant_table.json"),
            )
        },
        "counts": manifest["counts"],
    }


def plan_wildtype(support_entry: dict, rows: list[dict]) -> dict[str, Any]:
    """One wild type's work: its alignment and the distinct mutations it asks for.

    A mutation may be named by more than one assay of the same wild type. It is
    one computation and one official ``-m`` line, and it stays attached to every
    ``sample_id`` that names it: collapsing those samples would delete
    measurements, not duplicates.
    """

    by_mutation: dict[str, list[dict]] = {}
    for row in rows:
        by_mutation.setdefault(row["mutation"], []).append(
            {"sample_id": row["sample_id"], "assay": row["assay"], "n_substitutions": row["n_substitutions"]}
        )
    mutations = sorted(
        by_mutation, key=lambda mutation: (by_mutation[mutation][0]["n_substitutions"], mutation))
    return {
        "wildtype_id": support_entry["wildtype_id"],
        "length": int(support_entry["length"]),
        "assays": sorted({row["assay"] for row in rows}),
        "alignment": support_entry["msa"],
        "alignment_sha256": support_entry["msa_sha256"],
        "samples": len(rows),
        "mutations": mutations,
        "samples_by_mutation": by_mutation,
        "shared_mutations": sum(1 for names in by_mutation.values() if len(names) > 1),
    }


def plan(root: Path, export: dict[str, Any], wildtypes: list[str] | None) -> list[dict[str, Any]]:
    support = export["support"]["wildtypes"]
    samples = export["table"]["samples"]
    known = sorted(support)
    if wildtypes is not None:
        unknown = sorted(set(wildtypes) - set(known))
        if unknown:
            raise ValueError(f"unknown wild type(s) {unknown}; the export holds {len(known)}")
        known = sorted(set(wildtypes))
    rows_by_wildtype: dict[str, list[dict]] = {name: [] for name in known}
    for row in samples:
        if row["wildtype_id"] in rows_by_wildtype:
            rows_by_wildtype[row["wildtype_id"]].append(row)
    missing = [name for name in known if not rows_by_wildtype[name]]
    if missing:
        raise ValueError(f"wild type(s) {missing} have no variants in the variant table")
    plans = [plan_wildtype(support[name], rows_by_wildtype[name]) for name in known]
    for entry in plans:
        path = Path(entry["alignment"])
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"{entry['wildtype_id']}: exported alignment {path} is missing")
        if sha256_file(path) != entry["alignment_sha256"]:
            raise RuntimeError(
                f"{entry['wildtype_id']}: {path} is not the alignment the export hashed")
        entry["alignment_path"] = str(path)
    return plans


# ---------------------------------------------------------------- one wild type


def run_key(entry: dict[str, Any], *, image_digest: str, nseqs: int, iterations: int) -> str:
    """The identity a resumed checkpoint must match exactly.

    Anything that could change a score is in here: the alignment bytes, the
    mutant list the run will receive, the two official parameters, the image
    digest, the number of repeats and this module's own schema.
    """

    material = {
        "schema": RUN_KEY_SCHEMA,
        "gemme_runtime": RUN_KEY_RUNTIME,
        "wildtype_id": entry["wildtype_id"],
        "alignment_sha256": entry["alignment_sha256"],
        "mutations_sha256": hashlib.sha256("\n".join(entry["mutations"]).encode()).hexdigest(),
        "nseqs": nseqs,
        "iterations": iterations,
        "repeats": REPEATS,
        "image_digest": image_digest,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def next_attempt(base: Path) -> Path:
    """A directory this execution owns, because a run may only write into an empty one.

    A checked-in complete checkpoint is reused before this is called, so an
    attempt directory exists only when a previous attempt failed or was killed.
    That attempt keeps its official outputs -- deleting evidence to make room for
    a retry is exactly what this stage refuses to do.
    """

    base.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = base / f"attempt{index}"
        try:
            candidate.mkdir()
        except FileExistsError:
            index += 1
            continue
        return candidate


def run_wildtype(entry: dict[str, Any], args: argparse.Namespace, image_digest: str, key: str, log_path: Path) -> dict[str, Any]:
    """Three complete official runs, or a checkpoint that says which one failed."""

    log = Log(log_path)
    stem = next_attempt(Path(args.out) / entry["wildtype_id"] / key[:16])
    mutants = stem / "mutations.txt"
    mutants.write_text("\n".join(entry["mutations"]) + "\n", encoding="utf-8")

    checkpoint: dict[str, Any] = {
        "schema": SCHEMA_CHECKPOINT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wildtype_id": entry["wildtype_id"],
        "length": entry["length"],
        "assays": entry["assays"],
        "samples": entry["samples"],
        "mutations": len(entry["mutations"]),
        "shared_mutations": entry["shared_mutations"],
        "alignment": entry["alignment"],
        "alignment_sha256": entry["alignment_sha256"],
        "mutants_file": display(mutants),
        "repeats": REPEATS,
        "run_key": key,
        "image_digest": image_digest,
        "runs": [],
        "status": "running",
    }
    started = time.time()
    for index in range(1, REPEATS + 1):
        out = stem / f"repeat{index}"
        log.write(f"{entry['wildtype_id']} repeat {index}/{REPEATS} -> {out}")
        attempt = {"index": index, "out_dir": str(out), "status": "running"}
        try:
            record = G.execute(
                Path(entry["alignment_path"]), mutants, out, args.image, args.nseqs, args.iterations, args.timeout_seconds)
        except Exception as error:  # one failed repeat fails the wild type, and says why
            attempt["status"] = "failed"
            attempt["error"] = f"{type(error).__name__}: {error}"
            attempt["wall_seconds"] = None
            checkpoint["runs"].append(attempt)
            checkpoint["status"] = "failed"
            checkpoint["failure"] = f"repeat {index} of {REPEATS} failed: {attempt['error']}"
            checkpoint["wall_seconds"] = round(time.time() - started, 1)
            log.write(f"{entry['wildtype_id']} FAILED at repeat {index}: {attempt['error']}")
            break
        rows = record_scores(out)
        if len(rows) != len(entry["mutations"]):
            raise RuntimeError(
                f"repeat {index}: {len(rows)} scored mutants for {len(entry['mutations'])} requested")
        # The official table preserves the input order and lists the canonical comma form
        # of each mutant. Keying the checkpoint by the export's own mutation string keeps
        # the colon-joined multi-mutants of the variant table and GEMME's commas apart.
        scores = {mutation: float(row["gemme_score"]) for mutation, row in zip(entry["mutations"], rows)}
        labels = {mutation: row["gemme_mutant"] for mutation, row in zip(entry["mutations"], rows)}
        attempt.update(
            status="completed",
            wall_seconds=record["run"]["wall_seconds"],
            command=record["run"]["command"],
            record_sha256=sha256_file(out / G.RECORD),
            alignment_sha256=record["alignment"]["staged_sha256"],
            mutations_sha256=record["mutations"]["staged_sha256"],
            combined_file=record["outputs"]["combined_file"],
            combined_sha256=record["outputs"]["combined_sha256"],
            official_files=record["outputs"]["official_files"],
            jet_selection=record["jet_sequence_selection"],
            labels=labels,
            scores=scores,
        )
        checkpoint["runs"].append(attempt)
        log.write(
            f"{entry['wildtype_id']} repeat {index} done in {attempt['wall_seconds']}s, "
            f"{len(scores)} mutants, JET retained {attempt['jet_selection'].get('retained')}")
    if checkpoint["status"] != "failed":
        checkpoint["status"] = "complete"
        checkpoint["wall_seconds"] = round(time.time() - started, 1)
    checkpoint["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return checkpoint


#: A wild type the official tool returned no estimate for, and one of its
#: samples. Named so that a reader of either table can refuse the row by status
#: rather than by inferring it from a column of zeros.
REFUSED_DEGENERATE = "refused_degenerate"


def degenerate_refusal(aggregated: dict[str, Any], support_entry: dict[str, Any]) -> dict[str, Any]:
    """Why one wild type carries no GEMME score, with the support that explains it.

    Every mutation scored 0.0 in every run: JET2 placed no position, so the
    official tool returned no estimate. A column of zeros is not a score of zero,
    and averaging it into a feature would read "neutral everywhere" off a tool
    that said nothing. The alignment depth is recorded beside the reason because
    that is what the refusal tracks -- in the completed R1 run every degenerate
    wild type had at most 14 homolog rows and none at 15 or more.
    """

    if not aggregated["degenerate"]:
        raise ValueError("this wild type is not degenerate; it carries scores")
    return {
        "reason": aggregated["deficient_note"],
        "alignment_rows": support_entry.get("rows"),
        "samples": len(aggregated["variants"]),
    }


def refused_variant(variant: dict[str, Any]) -> dict[str, Any]:
    """One refused sample: the raw runs kept as evidence, the score itself null."""

    return variant | {"gemme_score": None, "gemme_sd": None, "status": REFUSED_DEGENERATE}


def record_scores(out: Path) -> list[dict[str, Any]]:
    """The per-mutant rows of one run, read back from the official output."""

    path = out / G.SCORES
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def aggregate(entry: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Mean and spread per variant, plus how well the runs agree on the ranking."""

    runs = checkpoint["runs"]
    mutations = entry["mutations"]
    per_mutation: dict[str, list[float]] = {mutation: [] for mutation in mutations}
    for run in runs:
        for mutation, value in run["scores"].items():
            per_mutation[mutation].append(float(value))
    label_maps = [run["labels"] for run in runs]
    if any(labels != label_maps[0] for labels in label_maps):
        raise RuntimeError(
            f"{entry['wildtype_id']}: the runs do not agree on the mutant labels they were given")
    short = [mutation for mutation, values in per_mutation.items() if len(values) != REPEATS]
    if short:
        raise RuntimeError(
            f"{entry['wildtype_id']}: {len(short)} mutations do not carry {REPEATS} values; a mean "
            "over fewer runs would misreport this wild type's sample size")

    means = {mutation: statistics.fmean(values) for mutation, values in per_mutation.items()}
    sds = {mutation: statistics.stdev(values) for mutation, values in per_mutation.items()}
    raw_agreements = []
    for first, second in combinations(runs, 2):
        paired = [mutation for mutation in mutations if mutation in first["scores"] and mutation in second["scores"]]
        raw_agreements.append(spearman([first["scores"][m] for m in paired], [second["scores"][m] for m in paired]))
    agreements = [value if value == value else None for value in raw_agreements]
    variants = []
    for mutation in mutations:
        values = per_mutation[mutation]
        for sample in entry["samples_by_mutation"][mutation]:
            variants.append(
                {
                    "sample_id": sample["sample_id"],
                    "assay": sample["assay"],
                    "wildtype_id": entry["wildtype_id"],
                    "mutation": mutation,
                    "n_substitutions": sample["n_substitutions"],
                    "n_runs": REPEATS,
                    "gemme_score": means[mutation],
                    "gemme_sd": sds[mutation],
                    "gemme_run1": values[0],
                    "gemme_run2": values[1],
                    "gemme_run3": values[2],
                }
            )
    defined = [value for value in agreements if value is not None]
    every_value = {value for run in runs for value in run["scores"].values()}
    return {
        "variants": variants,
        "sd_median": statistics.median(sds.values()),
        "sd_max": max(sds.values()),
        "run_rank_spearman": agreements,
        "run_rank_spearman_min": min(defined) if defined else None,
        "rank_agreement_undefined": len(agreements) - len(defined),
        "degenerate": len(every_value) <= 1,
        "deficient_note": (
            "every mutation scored 0.0 in every run: JET2 placed no position, so this wild type's "
            "support is too thin for GEMME and its scores carry no signal"
            if every_value == {0.0}
            else None
        ),
    }


def spearman(first: list[float], second: list[float]) -> float:
    """Rank correlation with average ranks for ties, so repeated values are not ordered."""

    if len(first) < 3:
        return float("nan")
    ranks_first, ranks_second = _ranks(first), _ranks(second)
    n = len(first)
    mean_first, mean_second = sum(ranks_first) / n, sum(ranks_second) / n
    numerator = sum((a - mean_first) * (b - mean_second) for a, b in zip(ranks_first, ranks_second))
    left = sum((a - mean_first) ** 2 for a in ranks_first)
    right = sum((b - mean_second) ** 2 for b in ranks_second)
    if left == 0 or right == 0:
        return float("nan")
    return numerator / (left * right) ** 0.5


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


# ------------------------------------------------------------------- the stage


def resolve(path: str | Path) -> Path:
    """A path argument, absolute, with a relative one taken against the repository."""

    candidate = Path(path)
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def check_resources(out: Path, workers: int) -> dict[str, Any]:
    """Every concurrent run asks JET2 for a fixed 9 GB JVM, so the floor is per worker."""

    record = G.resource_record(out)
    needed = G.MIN_AVAILABLE_MEMORY * workers
    if record["memory_available_bytes"] < needed:
        raise RuntimeError(
            f"MemAvailable {record['memory_available_bytes']} is below the {needed} that {workers} "
            "concurrent JET2 runs need at their fixed -Xmx9126m")
    if record["disk_free_bytes"] < G.MIN_FREE_DISK * workers:
        raise RuntimeError(f"only {record['disk_free_bytes']} bytes free at {out}")
    record["workers"] = workers
    record["memory_floor_bytes"] = needed
    record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return record


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve(args.root)
    out = resolve(args.out)
    log = Log(Path(args.log))
    export = verify_export(root)
    log.write(
        f"export verified: {export['counts']['wildtypes']} wild types, "
        f"{export['counts']['samples']} samples, "
        f"{export['counts']['single_substitution']} single / "
        f"{export['counts']['multiple_substitution']} multiple substitution")
    selected = None
    if args.wildtypes:
        selected = [name.strip() for name in args.wildtypes.split(",") if name.strip()]
    plans = plan(root, export, selected)
    log.write(f"planned {len(plans)} wild types, {sum(len(p['mutations']) for p in plans)} distinct mutations")

    image = G.image_identity(args.image)
    digest = image["image_id"]
    out.mkdir(parents=True, exist_ok=True)
    resources = check_resources(out, args.workers)
    log.write(f"image {digest} · workers {args.workers} · {json.dumps(resources['load_average'])}")

    checkpoints: dict[str, dict[str, Any]] = {}
    pending = []
    for entry in plans:
        key = run_key(entry, image_digest=digest, nseqs=args.nseqs, iterations=args.iterations)
        path = out / entry["wildtype_id"] / key[:16] / "checkpoint.json"
        if args.resume and path.is_file():
            existing = load(path)
            if existing.get("run_key") == key and existing.get("status") == "complete":
                missing = [run["out_dir"] for run in existing["runs"] if not Path(run["out_dir"]).is_dir()]
                if missing:
                    raise RuntimeError(
                        f"{entry['wildtype_id']}: checkpoint {path} is complete but {len(missing)} of "
                        f"its run directories are gone ({missing}); the evidence it cites no longer "
                        "exists, so it cannot be reused. Restore them or use a new --out")
                log.write(f"{entry['wildtype_id']}: reusing complete checkpoint {path}")
                checkpoints[entry["wildtype_id"]] = existing
                continue
            log.write(f"{entry['wildtype_id']}: checkpoint {path} is not a complete run of this key")
        pending.append((entry, key, path))

    def job(entry: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
        checkpoint = run_wildtype(entry, args, digest, key, Path(args.log))
        save(path, checkpoint)
        return checkpoint

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, entry, key, path): entry["wildtype_id"] for entry, key, path in pending}
        for future in as_completed(futures):
            name = futures[future]
            checkpoints[name] = future.result()
            log.write(f"{name}: {checkpoints[name]['status']}")

    return assemble(args, log, export, image, resources, plans, checkpoints)


def assemble(args: argparse.Namespace, log: Log, export: dict[str, Any], image: dict[str, Any],
             resources: dict[str, Any], plans: list[dict[str, Any]],
             checkpoints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One table keyed by sample_id, stating exactly which wild types are complete."""

    scored: dict[str, dict[str, Any]] = {}
    degenerate: dict[str, dict[str, Any]] = {}
    refused: dict[str, dict[str, Any]] = {}
    wildtypes: dict[str, dict[str, Any]] = {}
    incomplete: list[dict[str, Any]] = []
    for entry in plans:
        name = entry["wildtype_id"]
        checkpoint = checkpoints[name]
        block = {
            "wildtype_id": name,
            "length": entry["length"],
            "assays": entry["assays"],
            "samples_expected": entry["samples"],
            "mutations": len(entry["mutations"]),
            "shared_mutations": entry["shared_mutations"],
            "repeats": REPEATS,
            "status": checkpoint["status"],
            "run_key": checkpoint["run_key"],
            "wall_seconds": checkpoint.get("wall_seconds"),
            "runs": [
                {
                    "index": run["index"],
                    "status": run["status"],
                    "out_dir": run["out_dir"],
                    "wall_seconds": run.get("wall_seconds"),
                    "command": run.get("command"),
                    "record_sha256": run.get("record_sha256"),
                    "combined_sha256": run.get("combined_sha256"),
                    "jet_retained": (run.get("jet_selection") or {}).get("retained"),
                    "error": run.get("error"),
                }
                for run in checkpoint["runs"]
            ],
        }
        if checkpoint["status"] != "complete":
            block["failure"] = checkpoint.get("failure")
            incomplete.append({"wildtype_id": name, "failure": checkpoint.get("failure")})
            wildtypes[name] = block
            continue
        try:
            aggregated = aggregate(entry, checkpoint)
        except Exception as error:  # no value is emitted, and the reason is kept
            block["status"] = "failed"
            block["failure"] = f"aggregation: {type(error).__name__}: {error}"
            incomplete.append({"wildtype_id": name, "failure": block["failure"]})
            wildtypes[name] = block
            continue
        block.update(
            sd_median=aggregated["sd_median"],
            sd_max=aggregated["sd_max"],
            run_rank_spearman=aggregated["run_rank_spearman"],
            run_rank_spearman_min=aggregated["run_rank_spearman_min"],
            rank_agreement_undefined=aggregated["rank_agreement_undefined"],
            degenerate=aggregated["degenerate"],
            deficient_note=aggregated["deficient_note"],
            samples_scored=len(aggregated["variants"]),
        )
        wildtypes[name] = block
        if aggregated["degenerate"]:
            block["status"] = REFUSED_DEGENERATE
            block["failure"] = aggregated["deficient_note"]
            refused[name] = degenerate_refusal(aggregated, export["support"]["wildtypes"][name])
            for variant in aggregated["variants"]:
                degenerate[variant["sample_id"]] = refused_variant(variant)
            continue
        for variant in aggregated["variants"]:
            scored[variant["sample_id"]] = variant

    # Every sample of the frozen table appears exactly once, in the frozen order,
    # carrying a score when -- and only when -- its wild type completed three
    # runs. A sample of an incomplete or unplanned wild type keeps its row with a
    # null score and its status, so coverage is never faked by dropping rows and
    # no partial mean can stand in for one: the ladder's numeric column only
    # exists when the whole panel does.
    samples: list[dict[str, Any]] = []
    for row in export["table"]["samples"]:
        variant = scored.get(row["sample_id"]) or degenerate.get(row["sample_id"])
        if variant is not None:
            samples.append(variant)
            continue
        block = wildtypes.get(row["wildtype_id"])
        status = "wildtype_pending"
        if block is not None:
            status = "wildtype_complete" if block["status"] == "complete" else "wildtype_incomplete"
        samples.append(
            {
                "sample_id": row["sample_id"],
                "assay": row["assay"],
                "wildtype_id": row["wildtype_id"],
                "mutation": row["mutation"],
                "n_substitutions": row["n_substitutions"],
                "n_runs": 0,
                "gemme_score": None,
                "gemme_sd": None,
                "gemme_run1": None,
                "gemme_run2": None,
                "gemme_run3": None,
                "status": status,
            }
        )
    for variant in scored.values():
        variant["status"] = "scored"

    document = {
        "schema": SCHEMA_SCORES,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/capability/baselines/score_r1_gemme.py",
        "root": display(resolve(args.root)),
        "out": display(resolve(args.out)),
        "repeats": REPEATS,
        "aggregate": AGGREGATE,
        "score_direction": G.SCORE_DIRECTION,
        "multi_mutant_note": MULTI_MUTANT_NOTE,
        "reproducibility_limit": G.REPRODUCIBILITY,
        "gemme": {
            "image": image,
            "reference": G.DEFAULT_IMAGE,
            "distribution": G.OFFICIAL_DISTRIBUTION,
            "nseqs": args.nseqs,
            "iterations": args.iterations,
            "entrypoint": str(Path(__file__).relative_to(ROOT)),
            "entrypoint_sha256": sha256_file(Path(__file__)),
            "runtime_sha256": sha256_file(Path(G.__file__)),
        },
        "export": export["files"],
        "export_counts": export["counts"],
        "resources": resources,
        "wildtypes": wildtypes,
        "samples": samples,
        "incomplete": incomplete,
        "refused": refused,
        "counts": {
            "wildtypes_expected": len(wildtypes),
            "wildtypes_complete": sum(1 for block in wildtypes.values() if block["status"] == "complete"),
            "wildtypes_incomplete": len(incomplete),
            "wildtypes_refused": len(refused),
            "samples": len(samples),
            "samples_scored": len(scored),
            "samples_refused": len(degenerate),
            "samples_expected": export["counts"]["samples"],
            "wildtypes_degenerate": sum(
                1 for block in wildtypes.values() if block.get("degenerate") is True),
        },
    }
    # A declared refusal is a stated limit of the support, not an unfinished run:
    # the panel is complete when every sample is either scored or refused with a
    # reason, and never when a refusal has been counted as a score.
    document["counts"]["complete"] = (
        document["counts"]["samples_scored"] + document["counts"]["samples_refused"]
        == document["counts"]["samples_expected"]
        and document["counts"]["wildtypes_incomplete"] == 0)
    save(Path(args.out) / "scores.json", document)
    write_arrays(Path(args.out) / "scores.npz", samples)
    log.write(
        f"assembled {document['counts']['samples_scored']}/{document['counts']['samples_expected']} samples "
        f"over {document['counts']['wildtypes_complete']} complete wild types "
        f"({document['counts']['wildtypes_incomplete']} incomplete, "
        f"{document['counts']['wildtypes_refused']} refused as degenerate)")
    return document


def write_arrays(path: Path, samples: list[dict[str, Any]]) -> None:
    """The same table as float arrays, for the ladder's numeric join.

    Unscored samples are NaN here and null in the document, exactly as the
    sibling rung writes them, and both files carry the same sample order so a
    reader can bind one to the other.
    """

    import numpy as np

    def numeric(name: str) -> "np.ndarray":
        return np.array([row[name] if row[name] is not None else np.nan for row in samples], dtype=np.float64)

    np.savez_compressed(
        path,
        sample_id=np.array([row["sample_id"] for row in samples]),
        assay=np.array([row["assay"] for row in samples]),
        wildtype_id=np.array([row["wildtype_id"] for row in samples]),
        mutation=np.array([row["mutation"] for row in samples]),
        status=np.array([row["status"] for row in samples]),
        n_substitutions=np.array([row["n_substitutions"] for row in samples], dtype=np.int64),
        n_runs=np.array([row["n_runs"] for row in samples], dtype=np.int64),
        gemme_score=numeric("gemme_score"),
        gemme_sd=numeric("gemme_sd"),
        gemme_run1=numeric("gemme_run1"),
        gemme_run2=numeric("gemme_run2"),
        gemme_run3=numeric("gemme_run3"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="ladder root holding inputs/")
    parser.add_argument("--out", default=None, help="output directory (default <root>/gemme)")
    parser.add_argument("--wildtypes", default=None, help="comma-separated wild-type ids, for a pilot")
    parser.add_argument("--workers", type=int, default=2, help=f"concurrent official runs, at most {MAX_WORKERS}")
    parser.add_argument("--resume", action="store_true", help="reuse checkpoints whose run key matches exactly")
    parser.add_argument("--image", default=G.DEFAULT_IMAGE)
    parser.add_argument("--nseqs", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--log", default=None, help=f"append-only log (default {DEFAULT_LOG}/gemme.log)")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.workers <= MAX_WORKERS:
        print(f"--workers must be between 1 and {MAX_WORKERS}: each run holds a fixed-size JVM", file=sys.stderr)
        return 2
    if arguments.out is None:
        arguments.out = str(resolve(arguments.root) / "gemme")
    if arguments.log is None:
        arguments.log = str(DEFAULT_LOG / "gemme.log")
    document = execute(arguments)
    print(
        f"{document['counts']['samples_scored']} samples over "
        f"{document['counts']['wildtypes_complete']} complete wild types -> "
        f"{Path(arguments.out) / 'scores.json'}")
    return 0 if not document["incomplete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
