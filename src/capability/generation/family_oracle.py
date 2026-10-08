"""The restored Pfam/HMMER family-recognition oracle, and the gate that trusts it.

Why this module exists
======================

Every conditional generation endpoint in this programme is a *recognition* rate:
the share of generated sequences an external, non-neural oracle assigns to the
requested family. The oracle is therefore the instrument, and the one failure the
endpoint cannot absorb is a silently broken instrument. An ``hmmscan`` that
cannot find its pressed database, a Pfam release that is not the pinned one, or a
scan whose tables were written but never read all return the *same* artefact as a
model that generated nothing recognisable: an empty recognition set, which reads
as a true-negative yield of zero.

So nothing here is optional. The executable and all five Pfam files are checked
against the digests ``configs/generation_replication_manifest.json`` pinned
before any of this campaign's sequences existed; the call is the frozen one,
token for token; and :func:`run_positive_control` must recover the exact family
set the frozen 2026-09-05 oracle run recorded on a small set of retained
sequences, and must recover *no* family on retained sequences that run recorded
as carrying none, before :func:`recognise` will score anything.

What the oracle says and does not say
=====================================

``any_family`` is "some curated profile recognises this sequence at the
release's own gathering thresholds". ``complete_domain`` additionally requires
that one domain alignment covers at least
:data:`src.capability.generation.generative_control.COMPLETE_DOMAIN_COVERAGE` of
that profile's model length, which is what separates a complete domain from a
fragment of one. Neither is folding, function or novelty: a Pfam assignment is a
sequence-level homology statement and is read as nothing more.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core.protein_annotations import parse_hmmscan_table
from .conditioned_generation import PFAM_THRESHOLD
from .generative_control import (
    COMPLETE_DOMAIN_COVERAGE,
    best_profile_coverage,
    parse_domain_table,
)

SCHEMA_VERSION = "generation_family_oracle_v1"

#: The pinned oracle release, by basename. The campaign manifest keys these by
#: the authoring host's absolute paths on purpose -- that identity is what the
#: replication was digested against -- so the basename is the only portable join.
ORACLE_BASENAMES: tuple[str, ...] = (
    "hmmscan",
    "Pfam-A.hmm",
    "Pfam-A.hmm.h3f",
    "Pfam-A.hmm.h3i",
    "Pfam-A.hmm.h3m",
    "Pfam-A.hmm.h3p",
)

#: Where a restored oracle is expected to sit inside its own root: the layout of
#: ``results/R6/generation_inputs_20260905/oracle-inputs.tar.gz``. An extracted
#: archive is untrusted data, so the root is a parameter and never a default
#: inside a data directory.
HMMSCAN_RELATIVE = Path("hmmer/bin/hmmscan")
PFAM_RELATIVE = Path("pfam/Pfam-A.hmm")

#: The declared positive and negative recognition controls, with the family sets
#: the frozen oracle run recorded for them.
POSITIVE_CONTROLS = Path(__file__).with_name("family-oracle-positive-controls.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def pinned_digests(manifest: Path) -> dict[str, str]:
    """The oracle digests the replication manifest froze, keyed by basename.

    Refused if the manifest does not pin exactly the declared file set: a run
    against a release the manifest does not describe is not this measurement.
    """

    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    inputs = payload.get("oracle_inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError(f"{manifest} pins no oracle_inputs block")
    digests = {Path(key).name: str(value) for key, value in inputs.items()}
    if set(digests) != set(ORACLE_BASENAMES):
        raise ValueError(
            f"{manifest} pins {sorted(digests)} and this oracle is defined over "
            f"{list(ORACLE_BASENAMES)}; the two are not the same instrument"
        )
    return digests


@dataclass(frozen=True)
class Oracle:
    """One verified oracle: the executable, the pressed database, its digests."""

    hmmscan: Path
    pfam: Path
    digests: dict[str, str]
    root: Path

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "hmmscan": str(self.hmmscan),
            "pfam": str(self.pfam),
            "digests": dict(sorted(self.digests.items())),
            "threshold": PFAM_THRESHOLD,
            "threshold_note": (
                "Pfam-A's own curated per-family gathering thresholds (--cut_ga), so a "
                "family call is a statement of the release and not a cut chosen inside "
                "this measurement"
            ),
            "complete_domain_coverage_threshold": COMPLETE_DOMAIN_COVERAGE,
        }


def load_oracle(root: Path, *, pinned: Mapping[str, str]) -> Oracle:
    """Locate a restored oracle under ``root`` and refuse anything but the pinned bytes.

    Raised, never recorded. A recognition run that proceeded on a release whose
    bytes differ from the pinned ones would report rates the campaign's own
    comparison members were never measured against, and the difference would be
    invisible in the artefact.
    """

    root = Path(root)
    hmmscan = root / HMMSCAN_RELATIVE
    pfam = root / PFAM_RELATIVE
    expected = dict(pinned)
    paths = {hmmscan.name: hmmscan, pfam.name: pfam}
    for suffix in (".h3f", ".h3i", ".h3m", ".h3p"):
        companion = pfam.with_name(pfam.name + suffix)
        paths[companion.name] = companion
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"the family-recognition oracle is not restored under {root}: {missing} "
            f"absent. Restore it from results/R6/generation_inputs_20260905/"
            f"oracle-inputs.tar.gz, or build HMMER from external/tools/hmmer-3.4.tar.gz "
            f"and press the Pfam HMMs in external/references/. Recognition is refused "
            f"rather than returning an empty recognition set, which would read as a "
            f"true-negative yield of zero"
        )
    observed = {name: _sha256(path) for name, path in paths.items()}
    drifted = {name: (observed[name], expected[name]) for name in expected if observed[name] != expected[name]}
    if drifted:
        raise ValueError(
            "the restored oracle's bytes differ from the release the campaign pinned: "
            + ", ".join(f"{name} observed {got[:16]} pinned {want[:16]}" for name, (got, want) in sorted(drifted.items()))
        )
    return Oracle(hmmscan=hmmscan, pfam=pfam, digests=observed, root=root)


def scan_command(
    oracle: Oracle, fasta: Path, table: Path, domain_table: Path, *, threads: int
) -> list[str]:
    """The frozen oracle call, token for token.

    Identical to the call ``scripts/capability/generation/annotate_generation_replication.py``
    issues for the replication campaign's own attempts, so a rate computed here
    and a rate computed there are the same measurement.
    ``tests/generation/test_family_oracle.py`` asserts the two agree.
    """

    if threads < 1:
        raise ValueError("hmmscan needs a positive thread count")
    return [
        str(oracle.hmmscan),
        "--tblout",
        str(table),
        "--domtblout",
        str(domain_table),
        "--noali",
        "--cut_ga",
        "--cpu",
        str(threads),
        "-o",
        "/dev/null",
        str(oracle.pfam),
        str(fasta),
    ]


def _write_fasta(path: Path, sequences: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in sequences.items():
            if not sequence:
                raise ValueError(f"{name}: refusing to write an empty sequence")
            handle.write(f">{name}\n")
            for start in range(0, len(sequence), 60):
                handle.write(sequence[start : start + 60] + "\n")


def _scan_shard(task: tuple[Oracle, Path, Path, int]) -> dict[str, Any]:
    oracle, fasta, stem, threads = task
    table = stem.with_suffix(".tbl")
    domain_table = stem.with_suffix(".domtbl")
    command = scan_command(oracle, fasta, table, domain_table, threads=threads)
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"hmmscan exited {completed.returncode} on {fasta.name}: "
            f"{completed.stderr[-2000:]}"
        )
    if not (table.is_file() and domain_table.is_file()):
        raise RuntimeError(f"hmmscan wrote no table for {fasta.name}")
    return {
        "shard": fasta.name,
        "command": command,
        "elapsed_seconds": time.monotonic() - started,
        "tbl_sha256": _sha256(table),
        "domtbl_sha256": _sha256(domain_table),
    }


def _collect(tables: Sequence[Path]) -> dict[str, dict[str, Any]]:
    families: dict[str, set[str]] = {}
    coverage: dict[str, float] = {}
    domain_families: dict[str, set[str]] = {}
    for stem in tables:
        for name, entries in parse_hmmscan_table(stem.with_suffix(".tbl")).items():
            families.setdefault(name, set()).update(
                entry["accession_unversioned"] for entry in entries
            )
        for name, entries in parse_domain_table(stem.with_suffix(".domtbl")).items():
            best = best_profile_coverage(entries)
            if best is not None:
                coverage[name] = max(coverage.get(name, 0.0), best)
            domain_families.setdefault(name, set()).update(
                entry["accession_unversioned"] for entry in entries
            )
    # Containment is the invariant the finer read depends on: HMMER writes a
    # domain row only when a single domain clears GA2, so a family can sit on the
    # sequence table with no domain row, but a domain row for a family the
    # sequence table does not carry would mean the two tables describe different
    # scans.
    for name, assigned in domain_families.items():
        extra = assigned - families.get(name, set())
        if extra:
            raise RuntimeError(
                f"{name}: the domain table carries {sorted(extra)}, which the sequence "
                "table does not; the two tables are not the same scan"
            )
    return {
        name: {
            "families": sorted(assigned),
            "any_family": bool(assigned),
            "best_profile_coverage": coverage.get(name),
            "complete_domain": bool(assigned)
            and coverage.get(name) is not None
            and coverage[name] >= COMPLETE_DOMAIN_COVERAGE,
        }
        for name, assigned in families.items()
    }


def recognise(
    oracle: Oracle,
    sequences: Mapping[str, str],
    *,
    workspace: Path,
    shard_size: int = 200,
    workers: int = 8,
    threads: int = 1,
    label: str = "recognition",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Family set and best profile coverage per named sequence, plus the scan receipt.

    A sequence no profile recognises is absent from the returned mapping, which is
    what "no family" means; the receipt carries the count that was searched, so an
    absent sequence can never be confused with one that was never scanned.
    """

    if shard_size < 1 or workers < 1:
        raise ValueError("recognition needs a positive shard size and worker count")
    named = {name: sequence for name, sequence in sequences.items() if sequence}
    workspace = Path(workspace)
    (workspace / "shards").mkdir(parents=True, exist_ok=True)
    (workspace / "oracle").mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "label": label,
        "n_requested": len(sequences),
        "n_searched": len(named),
        "n_empty": len(sequences) - len(named),
        "shard_size": int(shard_size),
        "workers": int(workers),
        "threads_per_shard": int(threads),
        **oracle.record(),
    }
    if not named:
        raise ValueError(
            f"{label}: no non-empty sequence was supplied to the oracle. An empty "
            "recognition set is refused rather than reported as a zero yield"
        )
    order = sorted(named)
    shards: list[Path] = []
    tasks: list[tuple[Oracle, Path, Path, int]] = []
    for index, start in enumerate(range(0, len(order), shard_size)):
        chunk = order[start : start + shard_size]
        fasta = workspace / "shards" / f"{label}_{index:05d}.fasta"
        _write_fasta(fasta, {name: named[name] for name in chunk})
        stem = workspace / "oracle" / fasta.stem
        shards.append(stem)
        tasks.append((oracle, fasta, stem, threads))
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        receipt["shards"] = list(pool.map(_scan_shard, tasks))
    hits = _collect(shards)
    unknown = sorted(set(hits) - set(named))
    if unknown:
        raise RuntimeError(f"the oracle returned names that were never submitted: {unknown[:5]}")
    receipt["n_with_any_family"] = len(hits)
    receipt["n_with_complete_domain"] = sum(1 for value in hits.values() if value["complete_domain"])
    return hits, receipt


def load_positive_controls(path: Path = POSITIVE_CONTROLS) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    controls = payload.get("controls")
    if not controls:
        raise ValueError(f"{path} declares no recognition control")
    positives = [entry for entry in controls if entry["required_families"]]
    negatives = [entry for entry in controls if not entry["required_families"]]
    if not positives or not negatives:
        raise ValueError(
            f"{path} must declare both positives and negatives: a positives-only "
            "control passes an oracle that assigns every family to everything, and a "
            "negatives-only control passes an oracle that assigns nothing"
        )
    return list(controls)


def run_positive_control(
    oracle: Oracle,
    *,
    workspace: Path,
    controls: Sequence[Mapping[str, Any]] | None = None,
    threads: int = 4,
) -> dict[str, Any]:
    """Refuse a restored oracle that does not reproduce the frozen recognitions.

    The declared positives carry the exact family set the frozen 2026-09-05 run
    recorded; the declared negatives carried none. An oracle that misses a family
    on a positive, invents one on a negative, or cannot run at all is refused here
    rather than producing a recognition set that reads as a measured zero.
    """

    entries = list(controls) if controls is not None else load_positive_controls()
    sequences = {str(entry["id"]): str(entry["sequence"]) for entry in entries}
    if len(sequences) != len(entries):
        raise ValueError("the recognition controls carry duplicate identifiers")
    hits, receipt = recognise(
        oracle,
        sequences,
        workspace=Path(workspace) / "positive_control",
        shard_size=max(1, len(sequences)),
        workers=1,
        threads=threads,
        label="positive_control",
    )
    failures: list[dict[str, Any]] = []
    per_control: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry["id"])
        required = sorted(entry["required_families"])
        observed = sorted(hits.get(name, {}).get("families", []))
        missing = sorted(set(required) - set(observed))
        # A positive control declares the exact frozen family set, so an extra
        # family on a positive is drift in the instrument and not a richer read.
        # A negative control declares none, so any family at all is a failure.
        spurious = sorted(set(observed) - set(required))
        record = {
            "id": name,
            "kind": "positive" if required else "negative",
            "required_families": required,
            "observed_families": observed,
            "missing_families": missing,
            "spurious_families": spurious,
            "passed": not missing and not spurious,
        }
        per_control.append(record)
        if not record["passed"]:
            failures.append(record)
    result = {
        "schema_version": SCHEMA_VERSION,
        "n_controls": len(entries),
        "n_positive": sum(1 for entry in entries if entry["required_families"]),
        "n_negative": sum(1 for entry in entries if not entry["required_families"]),
        "controls": per_control,
        "scan_receipt": receipt,
        "passed": not failures,
    }
    if failures:
        raise RuntimeError(
            "the restored family-recognition oracle failed its own controls and "
            "nothing is scored with it: "
            + "; ".join(
                f"{record['id']} missing {record['missing_families']} spurious "
                f"{record['spurious_families']}"
                for record in failures
            )
        )
    return result


def target_hit(families: Iterable[str], referent: Iterable[str]) -> bool:
    """Whether a recognised family set intersects a class's declared referent.

    The referent is the Pfam family set the class's own referent draw carried, as
    frozen in the campaign's anchor artefact. Neither an EC number nor an InterPro
    superfamily is a Pfam accession, so the class-to-profile map comes from that
    artefact and is never reconstructed here.
    """

    wanted = {str(value).split(".", 1)[0] for value in referent}
    if not wanted:
        raise ValueError(
            "a class with an empty referent has no profile set the oracle could "
            "assign to; it is an unmeasurable class and must not enter a rate"
        )
    return any(str(value).split(".", 1)[0] in wanted for value in families)


def class_referents(anchors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The frozen class-to-Pfam-referent map of one arm's anchor artefact.

    Reads ``classes`` when present -- ``anchors_<arm>.json`` carries every drawn
    class there, including the ones the instrument gate declared unmeasurable --
    and otherwise ``instrument_anchors``, which the campaign report keeps for the
    admitted classes only. The ``admitted`` flag travels with the referent so a
    caller can state which support it aggregated over instead of inferring it.
    """

    block = anchors.get("classes") or anchors.get("instrument_anchors") or anchors
    referents: dict[str, dict[str, Any]] = {}
    for class_key, record in block.items():
        if not isinstance(record, Mapping) or "referent" not in record:
            continue
        referents[str(class_key)] = {
            "referent": tuple(str(value) for value in record["referent"]),
            "admitted": bool(record.get("admitted", True)),
            "label": str(record.get("label", class_key)),
            "real_rate": record.get("real_rate"),
            "random_rate": record.get("random_rate"),
        }
    if not referents:
        raise ValueError("the anchor artefact carries no class referent")
    return referents
