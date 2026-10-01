"""Shared protein annotations utilities required by capability measurements."""
from __future__ import annotations

import gzip
import re
import shutil
import subprocess
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .amino_acids import AA20
from .io import sha256_file
from .statistics import MINIMUM_BOOTSTRAP_UNITS, paired_group_bootstrap

@dataclass(frozen=True)
class HmmerTool:
    """A built HMMER 3.4 installation and the provenance to reproduce it."""

    hmmscan: Path
    hmmpress: Path
    version: str
    tarball: Path
    tarball_sha256: str
    hmmscan_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "hmmscan": str(self.hmmscan),
            "hmmpress": str(self.hmmpress),
            "version": self.version,
            "tarball": str(self.tarball),
            "tarball_sha256": self.tarball_sha256,
            "hmmscan_sha256": self.hmmscan_sha256,
            "checksum_note": "no publisher checksum is staged beside hmmer-3.4.tar.gz, "
            "unlike diamond-linux64-v2.1.24.tar.gz; the measured digest of the archive "
            "that was built and of the resulting binary are recorded instead of a "
            "digest nobody published being asserted",
        }


def prepare_hmmer(tarball: Path, destination: Path) -> HmmerTool:
    """Build HMMER from the staged source archive, or reuse an existing build.

    ``homology.prepare_diamond``'s shape, with the one difference the staged
    artefact forces: HMMER ships as source, so the tool is *built* rather than
    extracted and the version is read back from the built binary rather than from
    the archive's name. The build goes to a working location outside the
    repository, for the reason that module gives -- a binary and a pressed
    database never enter version control.
    """

    tarball = Path(tarball)
    destination = Path(destination)
    if not tarball.is_file():
        raise FileNotFoundError(f"{tarball} does not exist")
    hmmscan = destination / "bin" / "hmmscan"
    hmmpress = destination / "bin" / "hmmpress"
    if not (hmmscan.is_file() and hmmpress.is_file()):
        build = destination / "build"
        if build.exists():
            shutil.rmtree(build)
        build.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as archive:
            archive.extractall(path=build, filter="data")
        roots = [entry for entry in build.iterdir() if entry.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(
                f"{tarball} unpacked to {len(roots)} top-level directories; expected one"
            )
        for command in (
            ["./configure", f"--prefix={destination.resolve()}"],
            ["make", "-j", "8"],
            ["make", "install"],
        ):
            subprocess.run(
                command, cwd=roots[0], check=True, capture_output=True, text=True
            )
    if not (hmmscan.is_file() and hmmpress.is_file()):
        raise RuntimeError(f"the build did not produce {hmmscan} and {hmmpress}")
    completed = subprocess.run(
        [str(hmmscan), "-h"], capture_output=True, text=True, check=True
    )
    match = re.search(r"HMMER ([0-9][0-9.]*)", completed.stdout)
    if match is None:
        raise RuntimeError(f"cannot parse a HMMER version from {completed.stdout[:200]!r}")
    if not match.group(1).startswith("3.4"):
        raise RuntimeError(
            f"the built binary reports HMMER {match.group(1)}; this stage declares 3.4"
        )
    return HmmerTool(
        hmmscan=hmmscan,
        hmmpress=hmmpress,
        version=match.group(1),
        tarball=tarball,
        tarball_sha256=sha256_file(tarball),
        hmmscan_sha256=sha256_file(hmmscan),
    )


@dataclass(frozen=True)
class PfamDatabase:
    """A pressed Pfam-A profile database and what it covers."""

    path: Path
    source_gz: Path
    source_sha256: str
    n_profiles: int

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source": str(self.source_gz),
            "source_sha256": self.source_sha256,
            "n_profiles": int(self.n_profiles),
        }


def prepare_pfam(
    archive: Path, checksum_file: Path, destination: Path, *, tool: HmmerTool
) -> PfamDatabase:
    """Verify, decompress and press Pfam-A, or reuse an existing pressed database.

    The published digest is verified rather than assumed, which is what
    ``prepare_diamond`` does and what §0.05 records the cost of skipping: a search
    whose inputs were not the ones named produced a retraction, and the error
    worked in the direction that defeated the hypothesis under test.
    """

    archive = Path(archive)
    checksum_file = Path(checksum_file)
    destination = Path(destination)
    for path in (archive, checksum_file):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
    fields = checksum_file.read_text(encoding="utf-8").split()
    if not fields or len(fields[0]) != 64:
        raise ValueError(f"{checksum_file} does not begin with a sha256 digest")
    expected = fields[0].lower()
    observed = sha256_file(archive)
    if observed != expected:
        raise RuntimeError(
            f"{archive} sha256 {observed} does not match the published {expected}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    profile = destination / "Pfam-A.hmm"
    if not (profile.is_file() and profile.with_suffix(".hmm.h3i").is_file()):
        with gzip.open(archive, "rb") as source, profile.open("wb") as target:
            shutil.copyfileobj(source, target, length=1 << 24)
        subprocess.run(
            [str(tool.hmmpress), "-f", str(profile)],
            check=True,
            capture_output=True,
            text=True,
        )
    n_profiles = 0
    with profile.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("NAME "):
                n_profiles += 1
    if n_profiles < 1:
        raise RuntimeError(f"{profile} carries no profile")
    return PfamDatabase(
        path=profile, source_gz=archive, source_sha256=observed, n_profiles=n_profiles
    )


def write_fasta(path: Path, sequences: Mapping[str, str]) -> Path:
    """One FASTA of the generated sequences, wrapped at 60 residues."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in sequences.items():
            if not sequence:
                raise ValueError(f"{name}: refusing to write an empty sequence")
            handle.write(f">{name}\n")
            for start in range(0, len(sequence), 60):
                handle.write(sequence[start : start + 60] + "\n")
    return path


def run_hmmscan(
    tool: HmmerTool,
    database: PfamDatabase,
    query_fasta: Path,
    output_tbl: Path,
    *,
    evalue: float | None = None,
    threads: int,
    gathering_threshold: bool = False,
) -> tuple[list[str], str]:
    """Assign Pfam families to the query sequences; return the command and the log tail.

    The E-value threshold is a parameter rather than a literal because it is the
    one knob that decides how many families a generated sequence appears to carry,
    so it belongs in the artefact. ``gathering_threshold`` instead passes
    ``--cut_ga``, Pfam's own curated per-family cut, which is what makes "this
    sequence carries this family" a statement of the release rather than a
    threshold decision taken inside the measurement; the two are mutually
    exclusive and one of them must be given. No masking option is passed: HMMER's
    own null model handles composition bias, and §0.05 records what happened the
    last time a masking default silently truncated the evidence this programme
    reads.
    """

    query_fasta = Path(query_fasta)
    output_tbl = Path(output_tbl)
    if not query_fasta.is_file():
        raise FileNotFoundError(f"{query_fasta} does not exist")
    if threads < 1:
        raise ValueError("invalid hmmscan parameters")
    if gathering_threshold == (evalue is not None):
        raise ValueError(
            "pass exactly one of evalue= and gathering_threshold=True; a run that "
            "declared both would report a cut it did not apply"
        )
    if evalue is not None and evalue <= 0:
        raise ValueError("invalid hmmscan parameters")
    cut = ["--cut_ga"] if gathering_threshold else ["-E", repr(evalue)]
    command = [
        str(tool.hmmscan),
        "--tblout",
        str(output_tbl),
        "--noali",
        *cut,
        "--cpu",
        str(threads),
        str(database.path),
        str(query_fasta),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    if not output_tbl.is_file():
        raise RuntimeError("hmmscan produced no table")
    return command, completed.stdout[-2000:]


def parse_hmmscan_table(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Per-query family hits from ``--tblout``, best-scoring first.

    The accession is compared without its version downstream, because a Pfam
    release bump changes the version of a family that is otherwise the same family
    and a concept's declared referent cannot track that.
    """

    hits: dict[str, list[dict[str, Any]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"malformed hmmscan table row: {line!r}")
        accession = fields[1]
        hits.setdefault(fields[2], []).append(
            {
                "family": fields[0],
                "accession": accession,
                "accession_unversioned": accession.split(".", 1)[0],
                "evalue": float(fields[4]),
                "score": float(fields[5]),
            }
        )
    for entries in hits.values():
        entries.sort(key=lambda entry: entry["evalue"])
    return hits


def annotation_rates(
    hits: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str],
    accessions: Sequence[str],
) -> dict[str, Any]:
    """A36-6's two rates: any family at all, and a family the concept declares.

    ``any_family_rate`` is the attainability quantity and is read first: a
    generator whose unperturbed output no annotator recognises cannot show concept
    enrichment, and a null against a zero base rate is indistinguishable from an
    unreachable statistic.
    """

    if not names:
        raise ValueError("no generated sequence to score")
    wanted = {str(value).split(".", 1)[0] for value in accessions}
    annotated = [name for name in names if hits.get(name)]
    carrying = [
        name
        for name in names
        if any(hit["accession_unversioned"] in wanted for hit in hits.get(name, ()))
    ]
    return {
        "n_sequences": len(names),
        "n_with_any_family": len(annotated),
        "any_family_rate": len(annotated) / len(names),
        "n_with_concept_family": len(carrying),
        "concept_family_rate": len(carrying) / len(names),
        "declared_accessions": sorted(wanted),
        "per_sequence_concept_hit": {name: name in set(carrying) for name in names},
    }


def annotation_rate_contrast(
    injected: Mapping[str, Any], baseline: Mapping[str, Any], *, seed: int, n_bootstrap: int
) -> dict[str, Any]:
    """A36-6's interval on the injected minus the alpha=0 concept-family rate.

    The resampling unit is the generated sequence, and that is declared rather
    than assumed: generated output has no near-duplicate groups, so the group
    bootstrap this repository uses everywhere degenerates to a bootstrap over
    sequences here. ``paired_group_bootstrap`` is still the resampler -- one group
    per sequence -- so no second resampler enters the package.
    """

    labels = ["injected"] * int(injected["n_sequences"]) + ["baseline"] * int(
        baseline["n_sequences"]
    )
    values = list(injected["per_sequence_concept_hit"].values()) + list(
        baseline["per_sequence_concept_hit"].values()
    )
    if len(labels) != len(values):
        raise ValueError("the two conditions' sequence counts do not match their labels")
    condition = np.asarray([label == "injected" for label in labels], dtype=int)
    hits = np.asarray([1.0 if value else 0.0 for value in values], dtype=np.float64)
    groups = np.arange(len(values))
    if min(int(injected["n_sequences"]), int(baseline["n_sequences"])) < MINIMUM_BOOTSTRAP_UNITS:
        raise ValueError(
            f"fewer than {MINIMUM_BOOTSTRAP_UNITS} usable sequences on one side; the "
            "interval is not reported wider, it is not reported"
        )
    bootstrap = paired_group_bootstrap(
        condition,
        hits,
        np.zeros_like(hits),
        groups,
        _rate_difference_metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    return {
        "criterion": "A36-6",
        "rate_injected": float(injected["concept_family_rate"]),
        "rate_baseline": float(baseline["concept_family_rate"]),
        "rate_difference": bootstrap["difference"],
        "rate_difference_ci95": bootstrap["difference_ci95"],
        "excludes_zero": bool(bootstrap["difference_ci95"][0] > 0.0),
        "resampling_unit": "the generated sequence; generated output carries no "
        "near-duplicate groups, so the group bootstrap degenerates to one group per "
        "sequence and that is declared rather than left to a reader",
        "n_bootstrap": bootstrap["n_bootstrap_requested"],
    }


def _rate_difference_metric(truth: np.ndarray, predicted: np.ndarray) -> float:
    injected = np.asarray(truth).astype(bool)
    if injected.all() or (~injected).all():
        return float("nan")
    return float(predicted[injected].mean() - predicted[~injected].mean())


def pfam_referent(
    records: Sequence[Mapping[str, Any]],
    bearing: Sequence[bool],
    *,
    min_bearing_records: int,
) -> tuple[str, ...]:
    """The Pfam families a concept's bearing records carry, as its external referent.

    A36-6 needs a "concept-consistent" family set, and EXP-R2-213's concepts are
    GO terms and EC numbers, neither of which is a Pfam accession. The referent is
    therefore derived from the cohort's own ``pfam`` column, on the **fit** split
    only, so that the evaluation split never defines the target it is scored
    against. A family must appear on at least ``min_bearing_records`` bearing
    records to enter; a concept whose referent comes out empty has readout B
    refused with that reason rather than scored against a mapping invented here.
    """

    if min_bearing_records < 1:
        raise ValueError("a referent family must appear on at least one bearing record")
    flags = np.asarray(list(bearing), dtype=bool)
    if flags.size != len(records):
        raise ValueError("the bearing flags do not align with the records")
    counts: dict[str, int] = {}
    for record, carries in zip(records, flags):
        if not carries:
            continue
        for value in record["pfam"] or ():
            accession = str(value).split(".", 1)[0]
            counts[accession] = counts.get(accession, 0) + 1
    return tuple(
        sorted(
            accession
            for accession, count in counts.items()
            if count >= min_bearing_records
        )
    )


def clean_availability(root: Path) -> dict[str, Any]:
    """Whether CLEAN can be run here, stated rather than worked around.

    CLEAN is an optional second external instrument. Its inference path runs an
    ESM-1b encoder followed by CLEAN's own trained model, and neither weight file
    is one this host can fetch. This returns what is present and what is missing so
    that the artefact records a refusal with its reason instead of an EC prediction
    nothing produced.
    """

    root = Path(root)
    inference = root / "app" / "CLEAN_infer_fasta.py"
    weights = sorted(root.rglob("*.pth")) + sorted(root.rglob("*.pt"))
    cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    esm = sorted(cache.glob("esm1b*")) if cache.is_dir() else []
    missing: list[str] = []
    if not inference.is_file():
        missing.append("CLEAN inference entry point")
    if not weights:
        missing.append("CLEAN trained model weights (*.pt/*.pth)")
    if not esm:
        missing.append("ESM-1b encoder weights (esm1b_t33_650M_UR50S)")
    return {
        "runnable": not missing,
        "source_present": inference.is_file(),
        "missing": missing,
        "reason": "CLEAN's EC prediction runs an ESM-1b encoder followed by CLEAN's own "
        "trained model; the staged tree carries the source only and this host has no "
        "route to either weight host, so no EC prediction is produced rather than a "
        "placeholder being written",
    }


_RESIDUES = frozenset(AA20)


def extract_generated_sequence(text: str, *, end_delimiter: str) -> str:
    """The residue run a generated continuation spells, up to the end delimiter.

    Anything that is not a canonical residue ends the sequence, which is stricter
    than stripping the delimiter alone: a decoder that wandered back into prose
    should contribute a short sequence or none, never a sequence with prose folded
    into it.
    """

    residues: list[str] = []
    for character in text.split(end_delimiter, 1)[0]:
        if character in _RESIDUES:
            residues.append(character)
        elif character.isspace():
            continue
        else:
            break
    return "".join(residues)
