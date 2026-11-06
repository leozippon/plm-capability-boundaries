#!/usr/bin/env python3
"""The official GEMME baseline, run in its author-published image.

    GEMME is the statistical baseline in the ladder, so it has to be the
    distribution the authors ship, not a re-implementation. The upstream
    ProteinGym wrapper this repository vendored points at a cluster path
    (`/n/groups/marks/software/GEMME`) that does not exist here, and GEMME
    itself is Python 2 plus R plus JET2 plus Java, none of which the `ct`
    interpreter provides. The authors' own distribution channel is
    `elodielaine/gemme:gemme` (Docker Hub, published by the corresponding
    author; the download page documents the same image and the same
    `python2.7 $GEMME_PATH/gemme.py` invocation). This module executes that
    image, byte-identified by its digest, and never substitutes a surrogate:
    if the image, the mount, or the official output is missing, the run fails
    with a recorded reason.

    The interface is the baseline contract of the statistical ladder: a
    query-first aligned FASTA, the mutants of interest (single or multi),
    and an output directory. Multi-mutants are genuinely supported by the
    official `-m` entry point -- `pred.R:normalizePredWithNbSeqsPCSelMult`
    sums the combined-model normalised predicted effect over the mutated
    positions -- so a colon-joined multi-mutant is one official call and not
    a sum of separately run singles.

    Everything the panel needs to interpret a number is recorded: the image
    digest and its environment, the exact command, the sha256 of the staged
    alignment and mutant list, the raw official output files, the score
    direction, the matrix indexing convention, and the R function that
    defines the multi-mutant semantics. Scores are copied out and never
    transformed; the sign, the wild-type convention, and the position base
    are reported as the official code defines them.

    Two properties of the official model decide how its numbers may be read.
    A multi-mutant's value is the sum of the per-position terms, so its
    magnitude grows with the number of changed sites and is not a joint
    energy comparable across mutation orders. And JET2 is not bit-reproducible:
    it samples cluster partitions and filters by CPU load, exposes no seed, and
    the same alignment scored twice gives different values -- about a tenth of
    the value in this smoke -- which the smoke record measures. A caller that
    needs that noise below a threshold has to repeat the run; this module
    reports one official run and says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402

# The run record's own shape label. What a run actually is comes from the image
# digest and the probed package digests, never from this string.
SCHEMA_VERSION = "gemme_runtime"
RECORD = "gemme_run.json"
SCORES = "gemme_scores.tsv"

# The image the authors publish and the download page documents. Pinned by name
# here and by digest in every run record.
DEFAULT_IMAGE = "elodielaine/gemme:gemme"

# The official package archive (GEMME.tgz, mtime 2023-09-15) that the image
# release corresponds to. Recorded for lineage; the artifact actually executed
# is the image, whose digest the record captures.
OFFICIAL_DISTRIBUTION = {
    "archive": "GEMME.tgz",
    "release": "2023-09-15",
    "sha256": "477938d6a3604ed3394f94ec2990043451db4ea3f94f69759020f7a6e8a39661",
    "bytes": 1149680,
    "url": "http://www.lcqb.upmc.fr/GEMME/",
    "license": "MIT (LICENSE.txt: Copyright (c) 2018 Elodie Laine, Yasaman Karami, Alessandra Carbone)",
    "citation": "Laine E, Karami Y, Carbone A. Mol Biol Evol 36(11):2604-2619 (2019), doi:10.1093/molbev/msz179",
}

COMBINED_SUFFIX = "_normPred_evolCombi.txt"
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
GAP = frozenset("-.")
MUTATION = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")

# JET2 asks the JVM for a fixed -Xmx9126m inside the official driver, so a host
# with less available memory would fail mid-run rather than at admission.
MIN_AVAILABLE_MEMORY = 12 * 2**30
MIN_FREE_DISK = 2 * 2**30

SCORE_DIRECTION = (
    "higher = less deleterious. The value is the normalised predicted effect "
    "(NPE) and equals minus the sum over the mutant's positions of "
    "trace[i] * (alpha*epi[i,aa] + (1-alpha)*independent_logodds[i,aa]) with "
    "alpha=0.6, computed by pred.R:normalizePredWithNbSeqsPCSelMult for a "
    "-m run and by pred.R:normalizePredWithNbSeqsPC for the full-matrix run. "
    "PE is a log-odds ratio of mutant over wild-type sequence counts, so "
    "fewer sequences carrying a substitution make it more negative; a "
    "substitution equal to the wild type is not a valid query. A value can be "
    "positive, which means the model prefers the mutant residue to the one the "
    "query carries at that site. A position whose JET2 trace is zero scores "
    "exactly 0 for every substitution, which is the model declining to place a "
    "mutation there rather than a neutral prediction."
)

REPRODUCIBILITY = (
    "one official run, not an expectation. JET2 samples cluster partitions and "
    "filters candidate sequences by CPU load, and the driver exposes no seed: "
    "re-scoring the same alignment and mutant list moves every value slightly "
    "(see the measured spread in the smoke note). Repeats belong to the caller "
    "and must be recorded with the run, since averaging would hide that the "
    "input is what was scored here."
)


class GemmeError(RuntimeError):
    """An official run could not be produced; nothing is substituted."""


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Records of a FASTA file, headers without the marker and sequences joined."""

    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:].strip(), []
        else:
            if header is None:
                raise GemmeError(f"{path}: sequence data before the first header")
            chunks.append(re.sub(r"\s+", "", line))
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise GemmeError(f"{path}: no FASTA records")
    return records


def query_first_alignment(path: Path) -> dict[str, Any]:
    """The staged alignment: one line per record, uppercase, ungapped query.

    GEMME takes the first sequence as the query and slices the alignment with
    `grep -m N+1 -A nl`, where `nl` is the line count of the query block, so
    non-uniform wrapping silently truncates homologs. One line per record
    removes that failure mode without touching sequence content.
    """

    if not Path(path).is_file():
        raise GemmeError(f"{path}: the alignment file does not exist")
    records = read_fasta(path)
    query_header, query = records[0]
    if not query:
        raise GemmeError(f"{path}: the query record is empty")
    if GAP & set(query):
        raise GemmeError(
            f"{path}: the query sequence contains gaps; GEMME positions are query "
            "indices of the ungapped first sequence, so a gapped query would shift "
            "every mutation"
        )
    unknown = set(query.upper()) - AA20
    if unknown:
        raise GemmeError(f"{path}: the query sequence contains non-standard residues {sorted(unknown)}")

    headers = [header for header, _ in records]
    if len(set(headers)) != len(headers):
        raise GemmeError(f"{path}: duplicate FASTA headers would make homologs ambiguous")

    return {
        "query_header": query_header,
        "query": query.upper(),
        "depth": len(records),
        "records": [(header, sequence.upper()) for header, sequence in records],
    }


def parse_mutations(path: Path, query: str) -> list[dict[str, Any]]:
    """Mutants of interest, canonicalised to the form GEMME's `-m` reads.

    The official parser (`pred.R:readMut`) splits a line on commas, reads the
    position as everything between the first and last character, and lowercases
    the mutant residue. It sums over the positions, so order does not change the
    value, but the README asks for positions in ascending order and we deliver
    that. Colon-joined multi-mutants are translated to the commas the official
    parser expects; the raw string is preserved for the record.
    """

    if not Path(path).is_file():
        raise GemmeError(f"{path}: the mutation list does not exist")
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    raw_mutants = [line for line in lines if line]
    if not raw_mutants:
        raise GemmeError(f"{path}: no mutants to score")

    mutants: list[dict[str, Any]] = []
    for line in raw_mutants:
        sites: list[dict[str, Any]] = []
        for token in re.split(r"[:,]", line):
            token = token.strip()
            match = MUTATION.match(token)
            if not match:
                raise GemmeError(f"{path}: {line!r} has a site {token!r} that is not like D136R")
            wild, position, mutant = match.group(1).upper(), int(match.group(2)), match.group(3).upper()
            if not 1 <= position <= len(query):
                raise GemmeError(
                    f"{path}: {line!r} names position {position} outside the {len(query)}-residue query"
                )
            if query[position - 1] != wild:
                raise GemmeError(
                    f"{path}: {line!r} declares {wild} at position {position} but the query has "
                    f"{query[position - 1]}"
                )
            if mutant == wild:
                raise GemmeError(f"{path}: {line!r} is not a substitution ({wild} to {mutant})")
            if mutant not in AA20:
                raise GemmeError(f"{path}: {line!r} substitutes non-standard residue {mutant}")
            sites.append({"position": position, "wild": wild, "mutant": mutant})
        if len({site["position"] for site in sites}) != len(sites):
            raise GemmeError(f"{path}: {line!r} mutates the same position twice")
        sites.sort(key=lambda site: site["position"])
        positions = [site["position"] for site in sites]
        mutants.append(
            {
                "mutant": line,
                "gemme_mutant": ",".join(f"{s['wild']}{s['position']}{s['mutant']}" for s in sites),
                "positions": positions,
                "n_sites": len(sites),
            }
        )
    return mutants


def memory_available() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise GemmeError("/proc/meminfo has no MemAvailable")


def resource_record(out_dir: Path) -> dict[str, Any]:
    """Host facts taken before the run, since JET2 fails late if memory is short."""

    load = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average": [float(value) for value in load[:3]],
        "memory_available_bytes": memory_available(),
        "disk_free_bytes": shutil.disk_usage(out_dir).free,
        "gpu_used": False,
    }


def require_resources(out_dir: Path) -> dict[str, Any]:
    """Refuse to start a run the host cannot finish."""

    record = resource_record(out_dir)
    if record["memory_available_bytes"] < MIN_AVAILABLE_MEMORY:
        raise GemmeError(
            f"MemAvailable {record['memory_available_bytes']} is below the {MIN_AVAILABLE_MEMORY} JET2 "
            "needs at its fixed -Xmx9126m"
        )
    if record["disk_free_bytes"] < MIN_FREE_DISK:
        raise GemmeError(f"only {record['disk_free_bytes']} bytes free at {out_dir}")
    return record


def docker(*arguments: str, timeout: float | None = 120) -> subprocess.CompletedProcess:
    """A docker CLI call, raising with the daemon's own words when it fails."""

    if shutil.which("docker") is None:
        raise GemmeError("the docker CLI is not on PATH; the official GEMME image cannot be run")
    completed = subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise GemmeError(
            f"docker {' '.join(arguments)} failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    return completed


def image_identity(image: str) -> dict[str, Any]:
    """The digest actually on disk plus the environment the authors baked in."""

    docker("info")
    inspect = docker("image", "inspect", image)
    try:
        details = json.loads(inspect.stdout)[0]
    except (json.JSONDecodeError, IndexError) as error:
        raise GemmeError(f"docker image inspect {image} returned no image description: {error}") from error
    digests = details.get("RepoDigests") or []
    if not digests:
        raise GemmeError(f"{image} has no RepoDigest; it is not a pinned image and cannot be cited")
    return {
        "image": image,
        "repo_digests": digests,
        "image_id": details.get("Id"),
        "created": details.get("Created"),
        "architecture": details.get("Architecture"),
        "env": details.get("Config", {}).get("Env", []),
    }


PROBE = (
    'printf "GEMME_PATH=%s\\nJET_PATH=%s\\n" "$GEMME_PATH" "$JET_PATH"; '
    'python2.7 -V 2>&1; Rscript -e \'cat(R.version.string, "\\n"); cat("seqinr ", '
    'as.character(packageVersion("seqinr")), "\\n")\' 2>&1; java -version 2>&1; '
    'sha256sum "$GEMME_PATH"/gemme.py "$GEMME_PATH"/gemmeAnal.py "$GEMME_PATH"/computePred.R '
    '"$GEMME_PATH"/pred.R "$GEMME_PATH"/default.conf 2>&1; '
    'ls "$GEMME_PATH"'
)


def probe_tool(image: str, log_path: Path, timeout: float) -> dict[str, Any]:
    """Read back what the image contains, so a record never assumes a version."""

    completed = docker("run", "--rm", image, "bash", "-c", PROBE, timeout=timeout)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    facts: dict[str, Any] = {"probe_log": str(log_path), "sha256": {}}
    for line in completed.stdout.splitlines():
        if "=" in line and line.split("=")[0] in {"GEMME_PATH", "JET_PATH"}:
            name, value = line.split("=", 1)
            facts[name.lower()] = value
        fields = line.split()
        if len(fields) == 2 and "/gemme" in fields[1].lower() and len(fields[0]) == 64:
            facts["sha256"][Path(fields[1]).name] = fields[0]
        if line.startswith("Python "):
            facts["python"] = line
        if line.startswith("R version"):
            facts["r"] = line
        if line.startswith("seqinr"):
            facts["seqinr"] = line
        if "version" in line and ("openjdk" in line.lower() or "java version" in line.lower()):
            facts.setdefault("java", line.strip())
    for required in ("gemme_path", "jet_path"):
        if required not in facts:
            raise GemmeError(
                f"the image did not export {required.upper()}; it is not the official GEMME image"
            )
    if "gemme.py" not in facts["sha256"]:
        raise GemmeError("gemme.py is not under $GEMME_PATH in this image")
    return facts


def stage_inputs(alignment: Path, mutations: Path, work: Path) -> dict[str, Any]:
    """Write the two files the official driver reads into its own directory."""

    staged = query_first_alignment(alignment)
    mutants = parse_mutations(mutations, staged["query"])
    work.mkdir(parents=True, exist_ok=True)

    alignment_path = work / "alignment.fasta"
    alignment_path.write_text(
        "".join(f">{header}\n{sequence}\n" for header, sequence in staged["records"]), encoding="utf-8"
    )
    mutation_path = work / "mutants.txt"
    mutation_path.write_text(
        "".join(f"{mutant['gemme_mutant']}\n" for mutant in mutants), encoding="utf-8"
    )

    query = staged["query"]
    identities = []
    for _, sequence in staged["records"][1:]:
        pairs = [(a, b) for a, b in zip(query, sequence) if a in AA20 and b in AA20]
        identities.append(sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0)

    return {
        "staged": staged,
        "mutants": mutants,
        "alignment_path": alignment_path,
        "mutation_path": mutation_path,
        "alignment": {
            "source_sha256": sha256_file(alignment),
            "staged_sha256": sha256_file(alignment_path),
            "query_header": staged["query_header"],
            "query_length": len(query),
            "depth": staged["depth"],
            "query_first": True,
            "one_line_per_record": True,
            "uppercased": True,
            "identity_to_query": {
                "min": round(min(identities), 6) if identities else None,
                "max": round(max(identities), 6) if identities else None,
            },
        },
        "mutations": {
            "source_sha256": sha256_file(mutations),
            "staged_sha256": sha256_file(mutation_path),
            "count": len(mutants),
            "multi_mutants": sum(1 for mutant in mutants if mutant["n_sites"] > 1),
        },
    }


def container_name() -> str:
    """A name no other run can hold: concurrent wild types share one process.

    The timestamp and the pid alone collided as soon as two wild types started
    together, and docker refuses a duplicate name, so the run needs a suffix that
    is unique inside the process as well.
    """

    return f"gemme-{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def run_gemme(
    work: Path, image: str, alignment_name: str, mutation_name: str, nseqs: int, iterations: int, timeout: float
) -> dict[str, Any]:
    """The documented official invocation, from the download page's own text."""

    inner = (
        f"python2.7 $GEMME_PATH/gemme.py {alignment_name} -r input -f {alignment_name} "
        f"-m {mutation_name} -N {nseqs} -n {iterations}"
    )
    name = container_name()
    argv = [
        "docker", "run", "--rm", "--name", name, "--workdir", "/project",
        "--mount", f"type=bind,source={work},target=/project", image, "bash", "-c", inner,
    ]
    logs = work.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "run.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# " + " ".join(shlex.quote(part) for part in argv) + "\n")
        handle.flush()
        try:
            completed = subprocess.run(
                argv, cwd=work, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout
            )
            code = completed.returncode
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
            raise GemmeError(f"GEMME exceeded {timeout}s; the container was removed and the run log is {log_path}")
    record = {
        "argv": argv,
        "command": " ".join(shlex.quote(part) for part in argv),
        "inner_command": inner,
        "workdir": str(work),
        "log": str(log_path),
        "exit_code": code,
        "wall_seconds": round(time.time() - started, 1),
        "nseqs": nseqs,
        "iterations": iterations,
    }
    if code != 0:
        raise GemmeError(f"GEMME exited {code}; see {log_path}")
    return record


def parse_jet_selection(work: Path) -> dict[str, Any]:
    """Which sequences JET2 actually used, from its own log.

    The profile's depth is not GEMME's evidence depth: JET2 filters the supplied
    alignment (identity, redundancy) and then keeps a working subset, and it is
    the only place those counts are printed. Reading them from its log is how
    the ladder records the difference between its own homolog rows and the rows
    the conservation step consumed.
    """

    logs = sorted(work.glob("*.out"))
    if len(logs) != 1:
        return {"parsed": False, "reason": f"found {[path.name for path in logs]} instead of one JET log"}
    log = logs[0]
    selection: dict[str, Any] = {"source": log.name, "source_sha256": sha256_file(log), "parsed": False}
    iterations: list[dict[str, Any]] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        before = re.match(r"^Selected (\d+) sequences before filtering$", stripped)
        after = re.match(r"^Selected (\d+) after filtering$", stripped)
        removed = re.match(r"^-([A-Za-z ]+): Removed (\d+) seqs$", stripped)
        start = re.match(r"^Filtering sequences \(Iteration (\d+)\):$", stripped)
        trees = re.match(r"^Nb alignments:(\d+) Nb sequences:(\d+)$", stripped)
        if before:
            selection["submitted"] = int(before.group(1))
        elif after:
            selection["retained"] = int(after.group(1))
        elif start:
            iterations.append({"iteration": int(start.group(1)), "removed": {}})
        elif removed:
            if not iterations:
                iterations.append({"iteration": None, "removed": {}})
            iterations[-1]["removed"][removed.group(1).strip()] = int(removed.group(2))
        elif trees:
            selection["alignments"] = int(trees.group(1))
            selection["sequences_in_alignments"] = int(trees.group(2))
    if iterations:
        selection["filter_iterations"] = iterations
    selection["parsed"] = "submitted" in selection and "retained" in selection
    return selection


def parse_combined(path: Path, mutants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The `_normPred_evolCombi.txt` rows, in the order the mutant list asked for.

    `write.table` of the named vector the `-m` path builds writes the label and
    the value; the official code preserves input order, so position is the
    primary mapping and the label is checked rather than trusted.
    """

    rows: list[tuple[str, float]] = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, line in enumerate(lines, start=1):
        fields = next(csv.reader([line], delimiter=" ", quotechar='"', skipinitialspace=True))
        fields = [field for field in fields if field]
        if len(fields) == 2 and _is_number(fields[1]):
            rows.append((fields[0], float(fields[1])))
            continue
        if not rows and (len(fields) < 2 or fields[0] == ""):
            continue  # the column header write.table emits, e.g. "x"
        raise GemmeError(
            f"{path}:{number}: expected a label and a finite value, got {line!r}; NA would mean the "
            "official code could not place the substitution, which is a defect and not a score"
        )

    if len(rows) != len(mutants):
        raise GemmeError(f"{path}: {len(rows)} rows for {len(mutants)} mutants")
    scores: list[dict[str, Any]] = []
    for mutant, (label, value) in zip(mutants, rows):
        if label.strip() != mutant["gemme_mutant"]:
            raise GemmeError(
                f"{path}: row for {mutant['mutant']!r} is labelled {label!r} instead of "
                f"{mutant['gemme_mutant']!r}; the official code preserves input order, so this is drift"
            )
        scores.append({**mutant, "gemme_score": value})
    return scores


def _is_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def find_combined(work: Path) -> Path:
    found = sorted(work.glob(f"*{COMBINED_SUFFIX}"))
    if len(found) != 1:
        raise GemmeError(f"{work}: expected one *{COMBINED_SUFFIX}, found {[path.name for path in found]}")
    return found[0]


def write_scores(path: Path, scores: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        stream = csv.writer(handle, delimiter="\t", lineterminator="\n")
        stream.writerow(["mutant", "gemme_mutant", "positions", "n_sites", "gemme_score"])
        for score in scores:
            stream.writerow(
                [
                    score["mutant"],
                    score["gemme_mutant"],
                    ",".join(str(position) for position in score["positions"]),
                    score["n_sites"],
                    repr(score["gemme_score"]),
                ]
            )


def execute(alignment: Path, mutations: Path, out_dir: Path, image: str, nseqs: int, iterations: int, timeout: float) -> dict[str, Any]:
    """One official GEMME run, or a record saying exactly why there is none.

    The output directory must be new or empty. A driver that reuses a directory
    can silently adopt a previous run's official outputs -- `find_combined` would
    pick up a `_normPred_evolCombi.txt` this run never wrote -- so a non-empty
    directory is refused instead of cleaned: the caller owns those files, and a
    refusal leaves them exactly as they were.
    """

    out_dir = out_dir.resolve()
    if out_dir.exists():
        if not out_dir.is_dir():
            raise GemmeError(f"{out_dir} exists and is not a directory")
        occupied = sorted(entry.name for entry in out_dir.iterdir())
        if occupied:
            raise GemmeError(
                f"{out_dir} is not empty ({len(occupied)} entries, starting with {occupied[:3]}); "
                "this run needs a new or empty directory because reusing one can adopt another "
                "run's official outputs. Nothing was deleted"
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "distribution": OFFICIAL_DISTRIBUTION,
        "score_direction": SCORE_DIRECTION,
        "reproducibility": REPRODUCIBILITY,
        "matrix_indexing": (
            "positions are 1-based indices into the ungapped query sequence (the first "
            "alignment record), and a mutation string names the wild-type residue there plus "
            "the substitution, uppercase. The -m output this module reads is one row per "
            "requested mutant, in input order. The full-matrix run of the same release emits "
            "20 rows labelled by the lowercase mutant residue and len(query) columns carrying "
            "R's default names V1..Vn, so the column index is the position and the wild-type "
            "residue is not in the file; its cells are NA where the substitution equals the "
            "wild type"
        ),
        "multi_mutant_semantics": (
            "official: pred.R:normalizePredWithNbSeqsPCSelMult sums the combined-model "
            "normalised predicted effect over the mutated positions, weighted by JET2 trace; "
            "one -m entry per multi-mutant, never a sum of independently run singles. Because "
            "the value is a sum over sites, its magnitude grows with n_sites and the two are "
            "not interchangeable; the smoke run confirms a multi equals the sum of the same "
            "run's single-site values at those positions"
        ),
        "query_inclusion": (
            "the staged alignment is the profile's query plus its original homolog rows, unchanged "
            "apart from case and line wrapping, and nothing is added; JET2 then filters that "
            "alignment (identity, redundancy, and the driver's own sampling) and keeps a working "
            "subset, so the depth the conservation step used is not the profile depth. Its own counts "
            "are in jet_sequence_selection, read from the JET log rather than estimated here"
        ),
    }
    stage = "resources"
    try:
        record["resources"] = require_resources(out_dir)
        stage = "stage inputs"
        # The mount is work/ -> /project, so the two files the driver reads live
        # there and every official output lands beside them.
        work = out_dir / "work"
        staged = stage_inputs(alignment, mutations, work)
        record["alignment"] = staged["alignment"]
        record["mutations"] = staged["mutations"]
        mutants = staged["mutants"]

        stage = "image"
        record["image"] = image_identity(image)
        stage = "probe"
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        record["tool"] = probe_tool(image, out_dir / "logs" / "probe.log", timeout)

        stage = "run"
        record["run"] = run_gemme(
            work,
            image,
            staged["alignment_path"].name,
            staged["mutation_path"].name,
            nseqs,
            iterations,
            timeout,
        )

        stage = "parse"
        record["jet_sequence_selection"] = parse_jet_selection(work)
        combined = find_combined(work)
        scores = parse_combined(combined, mutants)
        write_scores(out_dir / SCORES, scores)
        record["outputs"] = {
            "combined_file": combined.name,
            "combined_sha256": sha256_file(combined),
            "official_files": {
                path.name: sha256_file(path) for path in sorted(work.iterdir()) if path.is_file()
            },
            "scores": SCORES,
            "scores_sha256": sha256_file(out_dir / SCORES),
        }
        record["score_summary"] = {
            "count": len(scores),
            "min": min(score["gemme_score"] for score in scores),
            "max": max(score["gemme_score"] for score in scores),
        }
        record["status"] = "completed"
    except Exception as error:  # recorded, re-raised: no score without the official run
        record["status"] = "failed"
        record["failed_at"] = stage
        record["error"] = f"{type(error).__name__}: {error}"
        write_json(out_dir / RECORD, record)
        raise
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(out_dir / RECORD, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alignment", required=True, type=Path, help="query-first aligned FASTA")
    parser.add_argument(
        "--mutations",
        required=True,
        type=Path,
        help="one mutant per line; multi-mutants joined by ':' (or ',') and ordered by position",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--nseqs", type=int, default=20000, help="GEMME -N, the driver's own default")
    parser.add_argument("--iterations", type=int, default=1, help="GEMME -n, the driver's own default")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    arguments = parser.parse_args(argv)
    try:
        record = execute(
            arguments.alignment,
            arguments.mutations,
            arguments.out_dir,
            arguments.image,
            arguments.nseqs,
            arguments.iterations,
            arguments.timeout_seconds,
        )
    except GemmeError as error:
        print(f"GEMME baseline failed: {error}", file=sys.stderr)
        return 1
    print(
        f"{record['score_summary']['count']} mutants scored into {arguments.out_dir / SCORES} "
        f"(image {record['image']['repo_digests'][0]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
