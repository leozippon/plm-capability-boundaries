"""Sequence search, homology strata and database inputs for capability measurements."""
from __future__ import annotations

import math
import re
import subprocess
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from ..core.arms import Cohort
from ..core.io import sha256_file

STRATUM_EDGES: tuple[float, ...] = (0.0, 30.0, 70.0, 95.0, 100.000001)


STRATUM_NAMES: tuple[str, ...] = (
    "lt30_no_detectable_homology",
    "id30_to_70_remote_homology",
    "id70_to_95_close_homology",
    "ge95_near_duplicate",
)


DIAMOND_FIELDS: tuple[str, ...] = (
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "nident",
    "qstart",
    "qend",
    "qlen",
    "slen",
    "evalue",
    "bitscore",
)


ALIGNMENT_FIELDS: tuple[str, ...] = (*DIAMOND_FIELDS, "qseq_gapped", "sseq_gapped")


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


@dataclass(frozen=True)
class DiamondTool:
    """A verified DIAMOND binary and the provenance needed to reproduce it."""

    executable: Path
    version: str
    binary_sha256: str
    tarball: Path
    tarball_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "executable": str(self.executable),
            "version": self.version,
            "binary_sha256": self.binary_sha256,
            "tarball": str(self.tarball),
            "tarball_sha256": self.tarball_sha256,
        }


def prepare_diamond(tarball: Path, checksum_file: Path, destination: Path) -> DiamondTool:
    """Verify the staged tarball against its checksum and extract it.

    The checksum is verified rather than assumed because two DIAMOND tarballs are
    staged side by side and only one has a published digest; extracting the wrong
    one would change the aligner without changing anything visible in the output.
    Extraction goes to a working location outside the repository so that a 25 GB
    database and a binary never enter version control.
    """

    tarball = Path(tarball)
    checksum_file = Path(checksum_file)
    destination = Path(destination)
    for path in (tarball, checksum_file):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")

    fields = checksum_file.read_text(encoding="utf-8").split()
    if len(fields) < 1 or len(fields[0]) != 64:
        raise ValueError(f"{checksum_file} does not begin with a sha256 digest")
    expected = fields[0].lower()
    observed = sha256_file(tarball)
    if observed != expected:
        raise RuntimeError(
            f"{tarball} sha256 {observed} does not match the published {expected}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        binaries = [member for member in members if Path(member.name).name == "diamond"]
        if len(binaries) != 1:
            raise RuntimeError(
                f"{tarball} holds {len(binaries)} files named 'diamond'; expected exactly one"
            )
        archive.extractall(path=destination, members=binaries, filter="data")
    executable = destination / binaries[0].name
    if not executable.is_file():
        raise RuntimeError(f"extraction did not produce {executable}")
    executable.chmod(0o755)

    completed = subprocess.run(
        [str(executable), "version"], capture_output=True, text=True, check=True
    )
    match = re.search(r"diamond version ([0-9][0-9.]*)", completed.stdout)
    if match is None:
        raise RuntimeError(f"cannot parse a version from {completed.stdout!r}")
    return DiamondTool(
        executable=executable,
        version=match.group(1),
        binary_sha256=sha256_file(executable),
        tarball=tarball,
        tarball_sha256=observed,
    )


@dataclass(frozen=True)
class DiamondDatabase:
    """Exactly what was searched, and how much of the source it covers.

    ``coverage_fraction`` is the whole point of this record.  A partial database
    can only ever miss homology, and the direction that error pushes the reading
    is asymmetric (see the module docstring), so a report that does not state
    coverage cannot be interpreted at all.
    """

    path: Path
    source_fasta: Path
    source_records: int
    sequences: int
    letters: int
    makedb_command: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source_records < 1 or self.sequences < 1 or self.letters < 1:
            raise ValueError("database record, sequence and letter counts must be positive")
        if self.sequences > self.source_records:
            raise ValueError(
                f"database holds {self.sequences} sequences but the source FASTA has "
                f"{self.source_records} records"
            )

    @property
    def coverage_fraction(self) -> float:
        return self.sequences / self.source_records

    def record(self) -> dict[str, Any]:
        source = Path(self.source_fasta)
        return {
            "database_path": str(self.path),
            "source_fasta": str(source),
            "source_fasta_bytes": source.stat().st_size,
            "source_fasta_records": self.source_records,
            "indexed_sequences": self.sequences,
            "indexed_letters": self.letters,
            "coverage_fraction": _finite(self.coverage_fraction, "database coverage"),
            "is_complete": self.sequences == self.source_records,
            "makedb_command": list(self.makedb_command),
        }


def count_fasta_records(path: Path, *, chunk: int = 1 << 24) -> tuple[int, int]:
    """``(records, residues)`` for a FASTA, in one pass and without decoding it.

    Both halves are returned because :func:`build_database` needs both to decide
    whether an existing index was built from *this* file. The record count alone
    cannot tell a half-built index from an index of a different corpus that
    happens to hold the same number of entries -- including the case that
    actually occurs, a source FASTA edited in place while its entry count stays
    put.

    ``residues`` counts every non-newline byte on a non-header line, which is
    what DIAMOND reports as ``Letters``. A sequence DIAMOND refuses to index
    also disappears from its ``Sequences`` count, so the two checks together
    admit an index only when it covers the same entries *and* the same residues
    as the file named beside it.

    Line state is carried across block boundaries, so a header whose newline
    ends one block and whose ``>`` starts the next is still counted once. On a
    24 GB corpus that boundary case happens a handful of times, which is small
    enough to look like a rounding difference and large enough to make the
    coverage fraction wrong.

    That invariant was *claimed* here and not delivered until EXP-R2-068. The
    inner scan ran ``while position <= len(block)``, so a block ending exactly
    on a newline took one extra iteration over an empty trailing segment, found
    no newline in it, and cleared ``at_line_start``. The next block's ``>`` then
    read as a continuation line: its record went uncounted and its header bytes
    were added to ``residues``. On ``'>r0\\nAAAA\\n>r1\\nCCCC\\n>r2\\nGGGG\\n'``
    -- true answer ``(3, 12)`` -- chunk 6 returned ``(2, 15)``, chunk 9 ``(1, 18)``
    and chunk 18 ``(2, 15)``. Read the loop bound as the invariant it enforces:
    a block is a sequence of complete lines plus at most one partial tail, and
    only a partial tail may clear ``at_line_start``.

    **No published number moved.** The shipped
    ``homology_control_unmasked/homology_assignment.json`` records
    ``source_fasta_records == indexed_sequences == 60315044`` at
    ``coverage_fraction 1.0``, so the 16 MiB default chunk never landed on a
    newline in the real UniRef50 file and EXP-R2-064's stratification stands.
    The defect was reachable, not reached.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    records = 0
    residues = 0
    in_header = False
    at_line_start = True
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            position = 0
            while position < len(block):
                newline = block.find(b"\n", position)
                segment = block[position:newline] if newline != -1 else block[position:]
                if at_line_start and segment[:1] == b">":
                    in_header = True
                    records += 1
                if not in_header:
                    residues += len(segment) - segment.count(b"\r")
                if newline == -1:
                    at_line_start = False
                    break
                in_header = False
                at_line_start = True
                position = newline + 1
    if records < 1:
        raise RuntimeError(f"{path} contains no FASTA records")
    return records, residues


def _dbinfo(tool: DiamondTool, database: Path) -> tuple[int, int]:
    completed = subprocess.run(
        [str(tool.executable), "dbinfo", "--db", str(database)],
        capture_output=True,
        text=True,
        check=True,
    )
    values: dict[str, int] = {}
    for key in ("Sequences", "Letters"):
        match = re.search(rf"^\s*{key}\s+(\d+)\s*$", completed.stdout, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(f"cannot parse {key} from `diamond dbinfo` output")
        values[key] = int(match.group(1))
    return values["Sequences"], values["Letters"]


def database_counts(tool: DiamondTool, database: Path) -> tuple[int, int]:
    """``(sequences, letters)`` an existing index reports about itself.

    Public because an index built elsewhere cannot always be checked the way
    :func:`build_database` checks one: a corpus staged as ``.fasta.gz`` has no
    residue count readable by :func:`count_fasta_records`, so the only thing an
    adopting caller can compare the index against is the published release
    record that was digest-verified when the corpus was staged.
    """

    return _dbinfo(tool, database)


def build_database(
    tool: DiamondTool,
    source_fasta: Path,
    database: Path,
    *,
    threads: int,
    tmpdir: Path,
    rebuild: bool = False,
) -> DiamondDatabase:
    """Index the corpus, or adopt an existing index after checking it.

    An existing index is adopted only if its own ``dbinfo`` counts match the
    source FASTA on **both** axes: sequences against records, and letters
    against residues.  A half-built or differently-built database would
    otherwise be searched silently and would answer a different question from
    the one the JSON claims was asked.

    The letter check was added by EXP-R2-067.  ``_dbinfo`` has always returned
    ``letters`` and ``DiamondDatabase.record`` has always published it as
    ``indexed_letters``, but nothing compared it, so adoption turned on the
    record count alone -- a key that cannot separate "this index was built from
    this file" from "this index was built from a different corpus with the same
    number of entries", which is what a source FASTA edited in place looks like.
    That is the shape this programme has already paid for once: a resume verdict
    that reported complete against inputs that no longer matched and still
    passed its own checksum.  Here the artefact would have named the new FASTA
    while every stratum came from the old index.
    """

    source_fasta = Path(source_fasta)
    database = Path(database)
    tmpdir = Path(tmpdir)
    if threads < 1:
        raise ValueError("threads must be positive")
    if not source_fasta.is_file():
        raise FileNotFoundError(f"{source_fasta} does not exist")
    source_records, source_residues = count_fasta_records(source_fasta)

    command = (
        str(tool.executable),
        "makedb",
        "--in",
        str(source_fasta),
        "--db",
        str(database),
        "--threads",
        str(threads),
        "--tmpdir",
        str(tmpdir),
    )
    if rebuild or not database.is_file():
        database.parent.mkdir(parents=True, exist_ok=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(list(command), check=True, capture_output=True, text=True)

    sequences, letters = _dbinfo(tool, database)
    if sequences != source_records:
        raise RuntimeError(
            f"{database} indexes {sequences} of {source_records} source records; it was "
            "not built from this FASTA in full. Rebuild it, or record the subset "
            "explicitly -- an unrecorded partial database makes the strata "
            "uninterpretable."
        )
    if letters != source_residues:
        raise RuntimeError(
            f"{database} indexes {sequences} sequences holding {letters} residues, but "
            f"{source_fasta} holds {source_residues} residues across the same "
            f"{source_records} records. The record counts agree and the contents do "
            "not, so this index was built from a different corpus than the one named "
            "beside it. Rebuild it with --rebuild-db; adopting it would attribute "
            "every homology stratum to a file that did not produce it."
        )
    return DiamondDatabase(
        path=database,
        source_fasta=source_fasta,
        source_records=source_records,
        sequences=sequences,
        letters=letters,
        makedb_command=command,
    )


def write_query_fasta(cohort: Cohort, path: Path) -> list[str]:
    """One FASTA record per cohort record, named by its cohort index.

    The identifier is positional because the stratification has to be joined back
    onto the cohort by position; accession-based naming would make that join
    depend on a header format the cohort does not promise to preserve.
    """

    if cohort.kind != "protein":
        raise ValueError(f"cohort {cohort.name!r} is not a protein cohort")
    if not cohort.records:
        raise ValueError(f"cohort {cohort.name!r} is empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identifiers = [f"q{index:05d}" for index in range(len(cohort.records))]
    with path.open("w", encoding="utf-8") as handle:
        for identifier, record in zip(identifiers, cohort.records):
            handle.write(f">{identifier}\n{record}\n")
    return identifiers


@dataclass(frozen=True)
class Hit:
    """One DIAMOND HSP, with identity expressed over the query rather than the HSP."""

    query: str
    subject: str
    pident: float
    length: int
    nident: int
    qstart: int
    qend: int
    qlen: int
    slen: int
    evalue: float
    bitscore: float
    #: The aligned query and subject strings *including* gap characters, so the
    #: two are the same length and can be walked column by column. Present only
    #: when the search was asked for :data:`ALIGNMENT_FIELDS`; optional because
    #: every existing caller stratifies on the counts above and pays nothing for
    #: an alignment it does not read, and a consumer that needs them must check.
    qseq_gapped: str | None = None
    sseq_gapped: str | None = None

    @property
    def identity_over_query(self) -> float:
        """Percent of the *query* that is identically matched.

        ``pident`` is identity within the aligned region, so a corpus entry that
        is a 60%-length fragment of the query scores 100 on ``pident`` while
        being nothing like a stored copy of the query.  Stratifying on ``pident``
        would put such a record in the near-duplicate bin and destroy the
        contrast this control depends on.
        """

        return 100.0 * self.nident / self.qlen


def run_diamond_blastp(
    tool: DiamondTool,
    database: DiamondDatabase,
    query_fasta: Path,
    output_tsv: Path,
    *,
    threads: int,
    sensitivity: str,
    evalue: float,
    max_target_seqs: int,
    fields: Sequence[str] = DIAMOND_FIELDS,
) -> tuple[list[str], str]:
    """Search the cohort against the corpus; return the command and the log tail.

    ``--very-sensitive`` is the default at the call site because the claim that
    matters most is a *negative* one -- that a record has no close relative in the
    corpus -- and a fast search cannot support a negative.  The query set is tens
    of sequences against tens of millions, so runtime is set by the database scan
    and sensitivity is nearly free.

    ``--masking 0`` is not a tuning choice; it is the difference between this
    control measuring what it claims to and measuring the opposite.  DIAMOND
    masks low-complexity and *repetitive* query regions by default, and this
    cohort is selected for containing internal tandem repeats.  With masking on,
    the HSP stops at the repeat: cohort record 0 of the 2026-07-28 run is
    byte-identical to ``UniRef50_Q3E8Z8`` over all 732 residues, and DIAMOND
    reported ``pident 100`` over 607 aligned residues, giving
    ``identity_over_query`` 82.9 and placing a verbatim member of ProtGPT2's
    pretraining corpus in the *close homology* bin rather than the near-duplicate
    one.  Five of forty-eight exact-cohort records were mis-binned that way, all
    in the same direction, and all into the bin with the highest measured
    induction -- which is exactly the pattern that would manufacture the
    "memorisation does not explain induction" reading this control exists to
    test.  The bias is also the reverse of the upward bias the module docstring
    declares, and it grows with repeat content, the one property the cohort is
    selected on.

    ``fields`` selects the tabular columns and defaults to
    :data:`DIAMOND_FIELDS`, which is what a stratification consumes. A caller
    that needs the alignments themselves -- a position-specific profile cannot
    be rebuilt from the counts -- passes :data:`ALIGNMENT_FIELDS`. The same list
    has to reach :func:`parse_hits`, so it is a parameter of both rather than a
    literal in either.
    """

    query_fasta = Path(query_fasta)
    output_tsv = Path(output_tsv)
    if not query_fasta.is_file():
        raise FileNotFoundError(f"{query_fasta} does not exist")
    if threads < 1 or evalue <= 0 or max_target_seqs < 1:
        raise ValueError("invalid DIAMOND search parameters")
    _checked_fields(fields)
    allowed = {"fast", "default", "sensitive", "mid-sensitive", "more-sensitive",
               "very-sensitive", "ultra-sensitive"}
    if sensitivity not in allowed:
        raise ValueError(f"unknown sensitivity {sensitivity!r}; known: {sorted(allowed)}")

    command = [
        str(tool.executable),
        "blastp",
        "--db",
        str(database.path),
        "--query",
        str(query_fasta),
        "--out",
        str(output_tsv),
        "--outfmt",
        "6",
        *fields,
        "--evalue",
        repr(evalue),
        "--max-target-seqs",
        str(max_target_seqs),
        "--threads",
        str(threads),
        # See the docstring: repeat masking truncates the alignment of exactly
        # the records this cohort is built from.
        "--masking",
        "0",
    ]
    if sensitivity != "default":
        command.append(f"--{sensitivity}")
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    log = (completed.stdout + completed.stderr).strip().splitlines()
    return command, "\n".join(log[-12:])


_FIELD_ATTRIBUTES: dict[str, tuple[str, Any]] = {
    "qseqid": ("query", str),
    "sseqid": ("subject", str),
    "pident": ("pident", float),
    "length": ("length", int),
    "nident": ("nident", int),
    "qstart": ("qstart", int),
    "qend": ("qend", int),
    "qlen": ("qlen", int),
    "slen": ("slen", int),
    "evalue": ("evalue", float),
    "bitscore": ("bitscore", float),
    "qseq_gapped": ("qseq_gapped", str),
    "sseq_gapped": ("sseq_gapped", str),
}


def _checked_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Refuse a field list this module cannot parse into a complete ``Hit``.

    Every column in :data:`DIAMOND_FIELDS` is required because every consumer
    reads them; anything outside :data:`_FIELD_ATTRIBUTES` has no home on the
    record and would be silently discarded, which is the shape that lets a
    search be asked for a column nothing ever reads.
    """

    requested = tuple(str(field) for field in fields)
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate DIAMOND output fields in {requested}")
    unknown = [field for field in requested if field not in _FIELD_ATTRIBUTES]
    if unknown:
        raise ValueError(
            f"DIAMOND fields {unknown} have no place on a Hit; known fields are "
            f"{sorted(_FIELD_ATTRIBUTES)}"
        )
    missing = [field for field in DIAMOND_FIELDS if field not in requested]
    if missing:
        raise ValueError(
            f"DIAMOND fields {missing} are required by every consumer of a Hit and "
            "are not in the requested output"
        )
    return requested


def parse_hits(output_tsv: Path, *, fields: Sequence[str] = DIAMOND_FIELDS) -> list[Hit]:
    """Read the tabular output, failing on any row that is not the declared shape.

    ``fields`` must be the list the search was run under; it defaults to
    :data:`DIAMOND_FIELDS`, so an existing caller reads exactly what it did
    before.
    """

    requested = _checked_fields(fields)
    rows: list[Hit] = []
    with Path(output_tsv).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != len(requested):
                raise ValueError(
                    f"{output_tsv}:{number} has {len(parts)} fields, expected "
                    f"{len(requested)}"
                )
            values = {}
            for field, part in zip(requested, parts):
                attribute, cast = _FIELD_ATTRIBUTES[field]
                values[attribute] = cast(part)
            rows.append(Hit(**values))
    return rows


def assign_stratum(identity: float) -> str:
    """Band a percent identity, using the boundaries fixed in this module."""

    value = _finite(identity, "percent identity")
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"percent identity {value} is outside [0, 100]")
    for index, name in enumerate(STRATUM_NAMES):
        if STRATUM_EDGES[index] <= value < STRATUM_EDGES[index + 1]:
            return name
    raise RuntimeError(f"percent identity {value} fell through every stratum")


@dataclass(frozen=True)
class HomologyAssignment:
    """One cohort record's closest relative in the corpus, and its stratum."""

    record_index: int
    query_id: str
    query_length: int
    n_hits: int
    max_identity_over_query: float
    max_pident: float
    best_subject: str | None
    best_bitscore: float | None
    best_qstart: int | None
    best_qend: int | None
    best_hit_spans_repeat: bool | None
    stratum: str
    #: The best hit is essentially exact over a length-matched subject yet
    #: covers well under the whole query. See :func:`truncated_alignment`.
    best_hit_looks_truncated: bool | None = None
    #: The hit list for this query reached ``--max-target-seqs``, so
    #: ``max_identity_over_query`` is a maximum over the reported hits and not
    #: over the corpus.
    hit_list_saturated: bool | None = None

    def record(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "query_id": self.query_id,
            "query_length": self.query_length,
            "n_hits": self.n_hits,
            "max_identity_over_query": _finite(
                self.max_identity_over_query, "max identity"
            ),
            "max_pident": _finite(self.max_pident, "max pident"),
            "best_subject": self.best_subject,
            "best_bitscore": (
                None if self.best_bitscore is None else _finite(self.best_bitscore, "bitscore")
            ),
            "best_qstart": self.best_qstart,
            "best_qend": self.best_qend,
            "best_hit_spans_repeat": self.best_hit_spans_repeat,
            "best_hit_looks_truncated": self.best_hit_looks_truncated,
            "hit_list_saturated": self.hit_list_saturated,
            "stratum": self.stratum,
        }


TRUNCATION_COVERAGE_LIMIT = 95.0


TRUNCATION_LENGTH_TOLERANCE = 0.02


TRUNCATION_PIDENT_FLOOR = 99.0


TRUNCATION_RULES = ("any", "stratum_changing")


def potential_identity_over_query(hit: Hit) -> float:
    """Identity over the query this alignment would reach if it were not truncated.

    An upper bound, and deliberately the most generous one: every query residue
    the alignment does not cover is assumed to have matched. That is what makes
    it safe to compare against a stratum boundary -- a truncation that cannot
    reach a higher stratum even under the most favourable repair cannot have
    caused a mis-binning.
    """

    if hit.qlen < 1:
        raise ValueError("a hit against a zero-length query has no identity")
    aligned = hit.qend - hit.qstart + 1
    if aligned < 1 or aligned > hit.qlen:
        raise ValueError(
            f"alignment covers {aligned} of a {hit.qlen}-residue query, which is not "
            "a query span"
        )
    return 100.0 * (hit.nident + hit.qlen - aligned) / hit.qlen


def truncation_raises_stratum(hit: Hit, observed_identity: float) -> bool:
    """Could repairing this truncation move its record into a higher stratum?"""

    potential = potential_identity_over_query(hit)
    if potential <= observed_identity:
        return False
    return assign_stratum(potential) != assign_stratum(observed_identity)


def truncated_alignment(hit: Hit) -> bool:
    """Does this alignment look truncated rather than partial?

    The signature is: the aligned region is essentially exact, the subject is
    essentially the same length as the query, and yet the alignment covers well
    under the whole query. A real partial homologue whose subject happens to
    match the query's length would have to be identical over a fragment and
    unalignable over the rest, which does not describe any biological
    relationship. It does describe an aligner that stopped at a masked region --
    the failure that put five verbatim corpus members into the close-homology bin
    and left no trace in any artefact.

    Reported rather than corrected: the repair is to search without masking, and
    silently re-binning a truncated hit would be inventing an alignment.
    """

    if hit.qlen < 1 or hit.slen < 1:
        return False
    length_ratio = abs(hit.slen - hit.qlen) / hit.qlen
    return bool(
        hit.pident >= TRUNCATION_PIDENT_FLOOR
        and length_ratio <= TRUNCATION_LENGTH_TOLERANCE
        and hit.identity_over_query < TRUNCATION_COVERAGE_LIMIT
    )


def assign_homology(
    cohort: Cohort,
    identifiers: Sequence[str],
    hits: Sequence[Hit],
    *,
    max_target_seqs: int | None = None,
    truncation_rule: str = "any",
) -> list[HomologyAssignment]:
    """Join hits back onto the cohort and band every record.

    A record with no hit is assigned identity 0 and lands in the lowest stratum.
    That is the correct reading of an ``e``-value-filtered miss, but it is also
    the only place where an incomplete database could put a memorised sequence in
    the wrong bin, so ``n_hits`` is carried through to the report rather than
    collapsed into the identity.

    A best hit that is essentially exact over a length-matched subject but covers
    well under the whole query stops the run. That combination is not a partial
    homologue, it is a truncated alignment, and it puts a verbatim member of the
    pretraining corpus into a lower stratum -- the direction that makes
    memorisation look like a weaker explanation than it is, which is the reading
    this control exists to test. It is refused rather than flagged because a
    stratification built on truncated alignments is not a weaker measurement, it
    is a different one.

    ``truncation_rule`` selects which flagged alignments stop the run; see
    :data:`TRUNCATION_RULES`. The default is the original behaviour, so no
    existing caller changes. ``"stratum_changing"`` is for a caller that searches
    thousands of targets per query, where the flag's measured false-positive
    class -- a hyper-conserved protein whose relatives are exact over a
    terminally-offset span -- makes the strict rule unrunnable without weakening
    what it protects.
    """

    if truncation_rule not in TRUNCATION_RULES:
        raise ValueError(
            f"unknown truncation rule {truncation_rule!r}; rules are {list(TRUNCATION_RULES)}"
        )
    if len(identifiers) != len(cohort.records):
        raise ValueError("identifier list does not match the cohort length")
    repeats = cohort.metadata.get("repeats")
    if repeats is None or len(repeats) != len(cohort.records):
        raise ValueError(f"cohort {cohort.name!r} carries no per-record repeat coordinates")

    by_query: dict[str, list[Hit]] = {identifier: [] for identifier in identifiers}
    for hit in hits:
        if hit.query not in by_query:
            raise ValueError(f"hit for unknown query {hit.query!r}")
        by_query[hit.query].append(hit)

    assignments: list[HomologyAssignment] = []
    for index, identifier in enumerate(identifiers):
        record = cohort.records[index]
        found = by_query[identifier]
        for hit in found:
            if hit.qlen != len(record):
                raise ValueError(
                    f"{identifier}: DIAMOND reports qlen {hit.qlen} for a "
                    f"{len(record)}-residue record"
                )
        if not found:
            assignments.append(
                HomologyAssignment(
                    record_index=index,
                    query_id=identifier,
                    query_length=len(record),
                    n_hits=0,
                    max_identity_over_query=0.0,
                    max_pident=0.0,
                    best_subject=None,
                    best_bitscore=None,
                    best_qstart=None,
                    best_qend=None,
                    best_hit_spans_repeat=None,
                    best_hit_looks_truncated=None,
                    hit_list_saturated=False,
                    stratum=assign_stratum(0.0),
                )
            )
            continue
        best = max(found, key=lambda hit: (hit.identity_over_query, hit.bitscore))
        first, second, span = (int(value) for value in repeats[index])
        # DIAMOND coordinates are 1-based inclusive; the repeat coordinates are
        # 0-based half-open over the record.
        spans_repeat = best.qstart - 1 <= first and best.qend >= second + span
        assignments.append(
            HomologyAssignment(
                record_index=index,
                query_id=identifier,
                query_length=len(record),
                n_hits=len(found),
                max_identity_over_query=best.identity_over_query,
                max_pident=max(hit.pident for hit in found),
                best_subject=best.subject,
                best_bitscore=best.bitscore,
                best_qstart=best.qstart,
                best_qend=best.qend,
                best_hit_spans_repeat=bool(spans_repeat),
                best_hit_looks_truncated=truncated_alignment(best),
                # ``n_hits`` counts reported HSP rows and ``--max-target-seqs``
                # caps *subject sequences*, so comparing one against the other
                # compares two different things: a subject reported under three
                # HSPs contributes three rows and one sequence, and the list then
                # reads as saturated at a third of the cap. Counted over distinct
                # subjects, which is the unit the cap is expressed in. On the
                # 2026-07-29 unmasked run every saturated record has exactly one
                # HSP per subject, so the two agree there and no published
                # saturation flag moves; the fix is for the search that does not.
                hit_list_saturated=(
                    None
                    if max_target_seqs is None
                    else len({hit.subject for hit in found}) >= max_target_seqs
                ),
                stratum=assign_stratum(best.identity_over_query),
            )
        )
    # Every hit, not only the best one. A truncated near-duplicate is a hit whose
    # alignment stopped at a masked region, and masking is exactly what can push
    # it *below* an untruncated but genuinely more distant relative -- so the
    # record whose stratification is wrong is precisely the record whose best hit
    # is not the truncated one. Inspecting only the best hit therefore looked
    # hardest at the cases that need it least. The 2026-07-29 unmasked run has no
    # truncated alignment at any rank, so no published stratification moves.
    observed = {assignment.query_id: assignment.max_identity_over_query
                for assignment in assignments}
    truncated = [
        (identifier, hit.subject, hit.identity_over_query)
        for identifier in identifiers
        for hit in by_query[identifier]
        if truncated_alignment(hit)
        and (
            truncation_rule == "any"
            or truncation_raises_stratum(hit, observed[identifier])
        )
    ]
    if truncated:
        raise RuntimeError(
            f"{len(truncated)} alignments over {len(assignments)} records are "
            f"exact over a length-matched subject yet cover under "
            f"{TRUNCATION_COVERAGE_LIMIT}% of the query, e.g. {truncated[:3]}. That is "
            "a truncated alignment, not a partial homologue, and it under-bins a "
            "verbatim corpus member. Re-run the search with --masking 0: DIAMOND masks "
            "repetitive query regions by default and this cohort is selected for "
            f"internal tandem repeats (truncation_rule={truncation_rule!r})"
        )
    return assignments
