"""Natural continuous HSP fragments against column-permutation nulls.

The retained R1 support is a query-coordinate projection: every subject insertion
is dropped, so its rows are not natural proteins and cannot answer a question
about the sequence distribution of real proteins. The original DIAMOND hits still
carry both gapped strings, so each HSP's continuous subject fragment is recovered verbatim
(``native``) together with the query position, if any, of every native index.

A causal model scores those native strings. The primary null permutes, at each
query column, the amino acid among the rows that carry a residue there; inserted
residues, row lengths, gap masks and the query-column fragment structure are
untouched. That null therefore destroys the association between a query column
and the insertions and mask beside it, which is why the resulting contrast is a
column-permutation contrast and not a calibrated probability or a pure
query-query higher-order term.

Nothing here reads a model at prepare time, and nothing here fits a phenotype.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..context.homology import ALIGNMENT_FIELDS, Hit, parse_hits
from ..core.amino_acids import AA20
from ..core.statistics import columns_with_spread

SCHEMA_QUEUE = "family_distribution_queue"
SCHEMA_AUDIT = "family_distribution_audit"
SCHEMA_SCORES = "family_distribution_scores"
SCHEMA_ANALYSIS = "family_distribution_analysis"
INPUT_SNAPSHOT = "scientific-inputs.json"

AA_INDEX = {residue: index for index, residue in enumerate(AA20)}
#: A native index aligned to no query position: a subject insertion.
INSERTION = -1


class NativeError(ValueError):
    """A hit that cannot become a lossless native row, with a counted reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class NativeRow:
    """One subject's residues verbatim, with its query-coordinate map.

    ``query_index`` has one entry per native character: the 0-based wild-type
    position the residue is aligned to, or :data:`INSERTION`. ``fragments`` are
    the half-open native ranges between query residues absent from the subject.
    These mark query-map structure, not breaks in the native chain: removing
    gaps from one HSP's sseq recovers one continuous natural subject fragment.
    Missing subject coordinates prevent locating it in the source record.
    """

    row_id: str
    subject: str
    bitscore: float
    qstart: int
    qend: int
    native: str
    query_index: np.ndarray
    fragments: tuple[tuple[int, int], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "subject": self.subject,
            "bitscore": self.bitscore,
            "qstart": self.qstart,
            "qend": self.qend,
            "native": self.native,
            "query_index": [int(value) for value in self.query_index],
            "fragments": [[int(start), int(stop)] for start, stop in self.fragments],
        }

    @classmethod
    def of(cls, payload: dict[str, Any]) -> "NativeRow":
        row = cls(
            row_id=str(payload["row_id"]),
            subject=str(payload["subject"]),
            bitscore=float(payload["bitscore"]),
            qstart=int(payload["qstart"]),
            qend=int(payload["qend"]),
            native=str(payload["native"]),
            query_index=np.asarray(payload["query_index"], dtype=np.int64),
            fragments=tuple((int(start), int(stop)) for start, stop in payload["fragments"]),
        )
        row.check()
        return row

    def check(self) -> None:
        if len(self.query_index) != len(self.native):
            raise ValueError(f"{self.row_id}: native length and query map differ")
        if not self.native or set(self.native) - set(AA20):
            raise ValueError(f"{self.row_id}: native string is not AA20")
        mapped = [int(value) for value in self.query_index if value != INSERTION]
        if any(later <= earlier for earlier, later in zip(mapped, mapped[1:])):
            raise ValueError(f"{self.row_id}: query positions are not strictly increasing")
        if any(int(value) < 0 and int(value) != INSERTION for value in self.query_index):
            raise ValueError(f"{self.row_id}: invalid query index")
        if mapped and not (self.qstart - 1 <= mapped[0] and mapped[-1] < self.qend):
            raise ValueError(f"{self.row_id}: query map leaves the reported HSP span")
        covered = [False] * len(self.native)
        for start, stop in self.fragments:
            if not 0 <= start < stop <= len(self.native) or any(covered[start:stop]):
                raise ValueError(f"{self.row_id}: invalid native fragments")
            covered[start:stop] = [True] * (stop - start)
        if not all(covered):
            raise ValueError(f"{self.row_id}: native fragments do not cover the row")


# ------------------------------------------------------------- configuration


REQUIRED_CONFIG = (
    "declaration", "paths", "native", "null", "clustering", "queue", "statistics",
    "scoring", "output",
)


def read_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    missing = [key for key in REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError(f"{path}: incomplete protocol, missing {missing}")
    if config["null"]["column_shuffles"] != len(config["null"]["column_shuffle_seeds"]):
        raise ValueError("the declared column shuffle count and seed list differ")
    if not 0.0 < config["clustering"]["coverage_of_shorter_row_floor"] <= 1.0:
        raise ValueError("a coverage floor lies in (0, 1]")
    if not 0.0 < config["clustering"]["overlap_identity_floor"] <= 1.0:
        raise ValueError("an identity floor lies in (0, 1]")
    if config["queue"]["pilot_units"] < 1 or config["queue"]["max_evaluation_units"] < 1:
        raise ValueError("the declared partition sizes are positive")
    return config


def load_json(path: Path) -> Any:
    """Read a JSON document, naming the file that failed instead of a bare decode error."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: unreadable JSON document: {error}") from error


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if name is None:
            return
        sequence = "".join(chunks)
        if not sequence:
            raise ValueError(f"{path}: empty record {name}")
        if set(sequence) - set(AA20):
            raise ValueError(f"{path}: record {name} is not AA20")
        if name in sequences:
            raise ValueError(f"{path}: duplicate record {name}")
        sequences[name] = sequence

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            flush()
            name, chunks = line[1:].strip().split()[0], []
        elif line.strip():
            if name is None:
                raise ValueError(f"{path}: sequence before the first header")
            chunks.append(line.strip())
    flush()
    return sequences


def wildtype_families(root: Path, config: dict[str, Any]) -> dict[str, int]:
    """The retained original-family label of every wild type, read only."""

    catalogue = load_json(root / config["paths"]["wildtypes_catalogue"])
    assay_to_wildtype = catalogue["assay_to_wildtype"]
    samples = load_json(root / config["paths"]["samples"])
    families: dict[str, int] = {}
    for row in samples:
        assay = str(row["assay"])
        if assay not in assay_to_wildtype:
            raise ValueError(f"{assay} is not in the retained assay-to-wildtype map")
        wildtype = str(assay_to_wildtype[assay])
        cluster = int(row["cluster"])
        if families.setdefault(wildtype, cluster) != cluster:
            raise ValueError(f"{wildtype} carries two original family clusters")
    return families


# ------------------------------------------------------------ native strings


def rebuild_gapped(row: NativeRow, wildtype: str) -> tuple[str, str]:
    """The DIAMOND column order implied by the row's native string and map."""

    query: list[str] = []
    subject: list[str] = []
    position = row.qstart - 1
    for index, mapped in enumerate(row.query_index):
        if mapped == INSERTION:
            query.append("-")
            subject.append(row.native[index])
            continue
        while position < mapped:
            query.append(wildtype[position])
            subject.append("-")
            position += 1
        query.append(wildtype[position])
        subject.append(row.native[index])
        position += 1
    while position < row.qend:
        query.append(wildtype[position])
        subject.append("-")
        position += 1
    return "".join(query), "".join(subject)


def native_row(row_id: str, wildtype: str, hit: Hit) -> NativeRow:
    """One subject's residues, refusing every lossy or inconsistent hit."""

    if hit.qlen != len(wildtype):
        raise NativeError("qlen_mismatch", f"{hit.query}/{hit.subject}: {hit.qlen} vs {len(wildtype)}")
    query_gapped, subject_gapped = hit.qseq_gapped, hit.sseq_gapped
    if query_gapped is None or subject_gapped is None:
        raise NativeError("no_alignment", f"{hit.query}/{hit.subject}")
    if len(query_gapped) != len(subject_gapped):
        raise NativeError("gapped_length_mismatch", f"{hit.query}/{hit.subject}")
    if query_gapped.replace("-", "") != wildtype[hit.qstart - 1 : hit.qend]:
        raise NativeError("query_residues_not_wildtype", f"{hit.query}/{hit.subject}")

    residues: list[str] = []
    mapping: list[int] = []
    fragments: list[tuple[int, int]] = []
    fragment_start = 0
    position = hit.qstart - 1
    for column, (query, subject) in enumerate(zip(query_gapped, subject_gapped)):
        if query == "-":
            if subject == "-":
                raise NativeError("both_gapped", f"{hit.query}/{hit.subject}:{column}")
            residues.append(subject)
            mapping.append(INSERTION)
            continue
        position += 1
        if subject == "-":
            if fragment_start < len(residues):
                fragments.append((fragment_start, len(residues)))
            fragment_start = len(residues)
            continue
        residues.append(subject)
        mapping.append(position - 1)
    if position != hit.qend:
        raise NativeError("qend_mismatch", f"{hit.query}/{hit.subject}: {position} vs {hit.qend}")
    if fragment_start < len(residues):
        fragments.append((fragment_start, len(residues)))

    native = "".join(residues)
    noncanonical = sorted(set(native) - set(AA20))
    if noncanonical:
        raise NativeError("noncanonical_subject", f"{hit.subject}: {noncanonical}")
    if not 40 <= len(native) <= 512:
        raise NativeError("length_out_of_range", f"{hit.subject}: {len(native)}")

    row = NativeRow(
        row_id=row_id,
        subject=hit.subject,
        bitscore=float(hit.bitscore),
        qstart=int(hit.qstart),
        qend=int(hit.qend),
        native=native,
        query_index=np.asarray(mapping, dtype=np.int64),
        fragments=tuple(fragments),
    )
    row.check()
    if rebuild_gapped(row, wildtype) != (query_gapped, subject_gapped):
        raise NativeError(
            "roundtrip_mismatch",
            f"{hit.query}/{hit.subject}: the native string does not rebuild the alignment "
            "verbatim (adjacent gap columns in the other order)",
        )
    return row


def select_rows(
    wildtype_id: str,
    wildtype: str,
    hits: Sequence[Hit],
    config: dict[str, Any],
) -> tuple[list[NativeRow], dict[str, Any]]:
    """Best hit per subject, coverage floor, exact dedup, seeded queue cap."""

    native = config["native"]
    reasons: Counter[str] = Counter()
    best: dict[str, Hit] = {}
    for hit in hits:
        if hit.qlen != len(wildtype):
            reasons["qlen_mismatch"] += 1
            continue
        span = hit.qend - hit.qstart + 1
        if 100.0 * span / len(wildtype) < native["coverage_floor_percent"]:
            reasons["coverage_below_floor"] += 1
            continue
        previous = best.get(hit.subject)
        if previous is None or hit.bitscore > previous.bitscore:
            best[hit.subject] = hit
    ordered = sorted(best.values(), key=lambda hit: (-hit.bitscore, hit.subject))
    ordered = ordered[: native["max_hits_per_wildtype"]]

    rows: list[NativeRow] = []
    seen: set[str] = set()
    for hit in ordered:
        try:
            row = native_row(f"{wildtype_id}#candidate", wildtype, hit)
        except NativeError as error:
            reasons[error.reason] += 1
            continue
        if row.native in seen:
            reasons["duplicate_native"] += 1
            continue
        seen.add(row.native)
        rows.append(row)

    parsed = len(rows)
    if len(rows) > native["queue_cap"]:
        rng = np.random.default_rng(config["queue"]["seed"])
        keep = np.sort(rng.choice(len(rows), size=native["queue_cap"], replace=False))
        rows = [rows[index] for index in keep]
        reasons["beyond_queue_cap"] = parsed - len(rows)
    rows = [replace(row, row_id=f"{wildtype_id}#{index:04d}") for index, row in enumerate(rows)]
    statistics = {
        "wildtype_id": wildtype_id,
        "length": len(wildtype),
        "hits_retrieved": len(hits),
        "subjects_after_coverage": len(best),
        "rows_parsed": parsed,
        "queue_rows": len(rows),
        "rejected": dict(sorted(reasons.items())),
    }
    return rows, statistics


# --------------------------------------------------------------------- nulls


def column_null(rows: Sequence[NativeRow], seed: int) -> list[str]:
    """Permute each query column's residues among the rows that carry one."""

    characters = [list(row.native) for row in rows]
    holders: dict[int, list[tuple[int, int]]] = {}
    for index, row in enumerate(rows):
        for native_index, position in enumerate(row.query_index):
            if position != INSERTION:
                holders.setdefault(int(position), []).append((index, native_index))
    rng = np.random.default_rng(seed)
    for position in sorted(holders):
        slots = holders[position]
        if len(slots) < 2:
            continue
        values = [characters[row][native] for row, native in slots]
        for target, source in zip(slots, rng.permutation(len(slots))):
            characters[target[0]][target[1]] = values[source]
    return ["".join(value) for value in characters]


def composition_null(row: NativeRow, seed: int) -> str:
    rng = np.random.default_rng(seed)
    characters = list(row.native)
    return "".join(characters[index] for index in rng.permutation(len(characters)))


def verify_column_null(rows: Sequence[NativeRow], shuffled: Sequence[str]) -> None:
    """Require the declared structure to survive the permutation exactly."""

    if len(rows) != len(shuffled):
        raise RuntimeError("a column null changed the number of rows")
    observed: dict[int, list[str]] = {}
    expected: dict[int, list[str]] = {}
    for row, value in zip(rows, shuffled):
        if len(value) != len(row.native):
            raise RuntimeError(f"{row.row_id}: a column null changed the native length")
        for index, position in enumerate(row.query_index):
            if position == INSERTION:
                if value[index] != row.native[index]:
                    raise RuntimeError(f"{row.row_id}: a column null moved an inserted residue")
                continue
            observed.setdefault(int(position), []).append(value[index])
            expected.setdefault(int(position), []).append(row.native[index])
    for position, values in expected.items():
        if Counter(values) != Counter(observed[position]):
            raise RuntimeError(f"column {position}: a column null changed the residue multiset")


# ---------------------------------------------------------------- clustering


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def cluster_rows(rows: Sequence[NativeRow], wildtype_length: int, config: dict[str, Any]) -> list[int]:
    """Connect rows that carry the same region under the declared thresholds."""

    rules = config["clustering"]
    count = len(rows)
    if count == 0:
        return []
    presence = np.zeros((count, wildtype_length), dtype=np.float32)
    codes = np.full((count, wildtype_length), -1, dtype=np.int8)
    for index, row in enumerate(rows):
        for native_index, position in enumerate(row.query_index):
            if position != INSERTION:
                presence[index, int(position)] = 1.0
                codes[index, int(position)] = AA_INDEX[row.native[native_index]]
    shared = presence @ presence.T
    matches = np.zeros((count, count), dtype=np.float32)
    for residue in range(len(AA20)):
        block = (codes == residue).astype(np.float32)
        matches += block @ block.T
    shorter = presence.sum(axis=1)
    span = np.minimum(shorter[:, None], shorter[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(shared > 0, shared / np.maximum(span, 1.0), 0.0)
        identity = np.where(shared > 0, matches / np.maximum(shared, 1.0), 0.0)
    connected = (
        (shared >= rules["shared_query_positions_floor"])
        & (coverage >= rules["coverage_of_shorter_row_floor"])
        & (identity >= rules["overlap_identity_floor"])
    )
    np.fill_diagonal(connected, False)
    union = _UnionFind(count)
    for left, right in np.argwhere(connected):
        union.union(int(left), int(right))
    labels: list[int] = []
    names: dict[int, int] = {}
    for index in range(count):
        labels.append(names.setdefault(union.find(index), len(names)))
    return labels


def representatives(rows: Sequence[NativeRow], labels: Sequence[int]) -> dict[int, int]:
    """Highest bitscore per cluster, subject id breaking ties."""

    chosen: dict[int, int] = {}
    for index, label in enumerate(labels):
        current = chosen.get(label)
        if current is None:
            chosen[label] = index
            continue
        row, best = rows[index], rows[current]
        if row.bitscore > best.bitscore or (row.bitscore == best.bitscore and row.subject < best.subject):
            chosen[label] = index
    return chosen


# ------------------------------------------------- repeated native fragments


def _twelve_mers(sequence: str) -> np.ndarray:
    codes = np.fromiter((AA_INDEX[residue] for residue in sequence), dtype=np.int64, count=len(sequence))
    window = 12
    if len(codes) < window:
        return np.zeros(0, dtype=np.int64)
    value = np.zeros(len(codes) - window + 1, dtype=np.int64)
    for offset in range(window):
        value = value * len(AA20) + codes[offset : offset + len(value)]
    return np.unique(value)


def _containment(left: Sequence[str], right: Sequence[str]) -> tuple[str, str] | None:
    """The first contained row in a stable order, so the audit is reproducible.

    ``set`` iteration order over strings changes between processes, so the order
    is fixed by length and then by the strings themselves; otherwise two runs
    would merge the same units and record different evidence.
    """

    for value in sorted(set(left), key=lambda text: (len(text), text)):
        for other in right:
            if len(value) <= len(other) and value in other:
                return value, other
    return None


def duplicate_fragment_groups(
    natives: dict[str, list[str]], *, config: dict[str, Any]
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Merge units that repeat a native sequence or contain one verbatim.

    A contained row (at least 40 residues) necessarily shares ONE 12-mer,
    even for A*40, which has only one distinct 12-mer. The inverted index
    generates candidates without a distinct-kmer-count threshold; substring
    confirmation, not fingerprint overlap, decides every edge.
    """

    names = sorted(natives)
    fingerprints = {
        name: (
            np.unique(np.concatenate([_twelve_mers(value) for value in natives[name]]))
            if natives[name]
            else np.zeros(0, dtype=np.int64)
        )
        for name in names
    }
    holders: dict[int, list[int]] = {}
    candidates: set[tuple[int, int]] = set()
    for right, name in enumerate(names):
        for fingerprint in fingerprints[name]:
            previous = holders.setdefault(int(fingerprint), [])
            candidates.update((left, right) for left in previous)
            previous.append(right)
    union = _UnionFind(len(names))
    evidence: list[dict[str, Any]] = []
    for left, right in sorted(candidates):
        a, b = names[left], names[right]
        found = None
        for first, second in ((a, b), (b, a)):
            hit = _containment(natives[first], natives[second])
            if hit is not None:
                found = {"units": sorted([a, b]), "kind": "contained_native",
                         "contained_length": len(hit[0]), "container_length": len(hit[1]),
                         "contained": hit[0]}
                break
        if found is None:
            continue
        evidence.append(found)
        union.union(left, right)
    grouped: dict[int, list[str]] = {}
    for index, name in enumerate(names):
        grouped.setdefault(union.find(index), []).append(name)
    return sorted(grouped.values(), key=lambda members: sorted(members)), evidence


# --------------------------------------------------------------- preparation


def qualify(cluster_count: int, config: dict[str, Any]) -> tuple[bool, str]:
    """Whether one wild type may become an inference unit at all."""

    required = config["clustering"]["min_clusters_per_wildtype"]
    if cluster_count >= required:
        return True, "qualified"
    return False, f"fewer_than_{required}_clusters"


def split_clusters(cluster_count: int, config: dict[str, Any], rng) -> dict[str, Any]:
    """The pre-declared held-out cluster count, drawn from the declared stream."""

    rules = config["clustering"]
    test_count = min(rules["max_test_clusters"], max(rules["min_test_clusters"], cluster_count // 2))
    train_count = cluster_count - test_count
    if train_count < rules["min_train_clusters"]:
        raise ValueError(
            f"{cluster_count} clusters leave {train_count} training clusters, below the "
            f"{rules['min_train_clusters']} declared minimum")
    order = rng.permutation(cluster_count)
    held = {int(label) for label in order[:test_count]}
    return {
        "test_clusters": test_count,
        "train_clusters": train_count,
        "test": sorted(f"c{label:03d}" for label in held),
        "train": sorted(f"c{label:03d}" for label in set(range(cluster_count)) - held),
    }


def select_family_wildtypes(
    by_family: dict[int, list[str]], clusters: dict[str, tuple[int, bool]]
) -> tuple[list[str], list[dict[str, Any]]]:
    """One qualified wild type per original family cluster, or an explicit exclusion."""

    selected: list[str] = []
    selection: list[dict[str, Any]] = []
    for cluster in sorted(by_family):
        candidates = sorted(by_family[cluster])
        usable = [name for name in candidates if clusters[name][1]]
        entry: dict[str, Any] = {
            "family_cluster": cluster,
            "candidates": [{"wildtype_id": name, "clusters": clusters[name][0],
                            "qualified": clusters[name][1]} for name in candidates],
        }
        if usable:
            chosen = sorted(usable, key=lambda name: (-clusters[name][0], name))[0]
            selected.append(chosen)
            entry["selected"] = chosen
            entry["reason"] = ("most_qualified_clusters" if len(usable) > 1
                               else "sole_qualified_wildtype")
        else:
            entry["selected"] = None
            entry["reason"] = "no_qualified_wildtype_in_family"
        selection.append(entry)
    return selected, selection


def evaluation_rows(wildtype: dict[str, Any]) -> list[NativeRow]:
    """The ONLY evaluation/donor batch: held-out near-duplicate representatives."""

    rows = [NativeRow.of(payload) for payload in wildtype["rows"]
            if payload["split"] == "test" and payload["representative"]]
    if not rows or len(rows) > 32:
        raise ValueError(f"{wildtype['wildtype_id']}: expected 1..32 held-out representatives")
    return rows


def scientific_inputs(root: Path, config: dict[str, Any], queue: dict[str, Any],
                      *, create: bool = False) -> str:
    """One immutable, complete JSON snapshot, shared by every score document.

    Direct content comparison deliberately replaces partial ID/length gates and
    digest frameworks. An old score without its snapshot is never adopted.
    """

    output = root / config["output"]
    path = output / INPUT_SNAPSHOT
    expected = {"protocol": config, "queue": queue}
    if queue.get("protocol") != config:
        raise ValueError("queue protocol differs from the current scientific inputs")
    if path.exists():
        if load_json(path) != expected:
            raise ValueError("scientific input snapshot differs; use a fresh output directory")
    elif any((output / "scores").glob("*/*.json")):
        raise ValueError("scores exist but the scientific input snapshot is missing")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(expected, indent=1, allow_nan=False) + "\n")
    else:
        raise ValueError("scientific input snapshot is missing; run prepare first")
    return INPUT_SNAPSHOT


def prepare(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the native queue, its cluster splits and the inference units."""

    support = load_json(root / config["paths"]["support"])
    catalogue = read_fasta(root / config["paths"]["wildtype_fasta"])
    sequences = {name: catalogue[name] for name in support["wildtypes"]
                 if name in catalogue}
    missing = sorted(set(support["wildtypes"]) - set(sequences))
    if missing:
        raise ValueError(f"the retained wild-type FASTA lacks supported wild types: {missing}")
    families = wildtype_families(root, config)
    unknown = sorted(set(sequences) - set(families))
    if unknown:
        raise ValueError(f"wild types without an original family label: {unknown}")

    hits: dict[str, list[Hit]] = {}
    wanted = set(sequences)
    for hit in parse_hits(root / config["paths"]["corpus_hits"], fields=ALIGNMENT_FIELDS):
        if hit.query in wanted:
            hits.setdefault(hit.query, []).append(hit)
    if set(hits) != wanted:
        raise ValueError("a wild type has no corpus hit")

    summaries: dict[str, dict[str, Any]] = {}
    labels_by_wildtype: dict[str, list[int]] = {}
    picks_by_wildtype: dict[str, dict[int, int]] = {}
    for wildtype_id in sorted(sequences):
        rows, statistics = select_rows(wildtype_id, sequences[wildtype_id], hits[wildtype_id], config)
        labels = cluster_rows(rows, len(sequences[wildtype_id]), config)
        cluster_count = len(set(labels))
        picks = representatives(rows, labels)
        qualified, reason = qualify(cluster_count, config)
        statistics.update(
            family_cluster=families[wildtype_id],
            clusters=cluster_count,
            qualified=qualified,
            reason=reason,
            representatives={f"c{label:03d}": rows[index].row_id for label, index in sorted(picks.items())},
        )
        summaries[wildtype_id] = statistics
        labels_by_wildtype[wildtype_id] = labels
        picks_by_wildtype[wildtype_id] = picks

    split_rng = np.random.default_rng(config["clustering"]["split_seed"])
    for wildtype_id in sorted(sequences):
        if not summaries[wildtype_id]["qualified"]:
            continue
        summaries[wildtype_id]["split"] = split_clusters(
            summaries[wildtype_id]["clusters"], config, split_rng)
    qualified = [name for name in sorted(sequences) if summaries[name]["qualified"]]

    by_family: dict[int, list[str]] = {}
    for wildtype_id in sorted(sequences):
        by_family.setdefault(families[wildtype_id], []).append(wildtype_id)
    selected, family_selection = select_family_wildtypes(
        by_family, {name: (summaries[name]["clusters"], summaries[name]["qualified"])
                    for name in summaries})
    for name in summaries:
        if name not in selected and summaries[name]["reason"] == "qualified":
            summaries[name]["reason"] = "not_selected_by_family_rule"

    natives: dict[str, list[str]] = {}
    for wildtype_id in selected:
        rows, _ = select_rows(wildtype_id, sequences[wildtype_id], hits[wildtype_id], config)
        natives[wildtype_id] = [row.native for row in rows]
    groups, merge_evidence = duplicate_fragment_groups(natives, config=config)
    del natives

    units = []
    for members in groups:
        label = "_".join(f"{families[name]:03d}" for name in sorted(members))
        chosen = min(members, key=lambda name: (-summaries[name]["clusters"], name))
        for name in members:
            summaries[name]["merged_unit_representative"] = chosen
            if name != chosen:
                summaries[name]["reason"] = "merged_source_not_evaluation_query"
        units.append({"unit_id": f"u{label}", "wildtype_ids": [chosen],
                      "source_wildtype_ids": sorted(members),
                      "family_clusters": sorted({families[name] for name in members})})
    units.sort(key=lambda entry: entry["unit_id"])
    order = np.random.default_rng(config["queue"]["seed"]).permutation(len(units))
    pilot = int(config["queue"]["pilot_units"])
    evaluation = int(config["queue"]["max_evaluation_units"])
    for rank, position in enumerate(order):
        entry = units[int(position)]
        entry["partition"] = (
            "pilot" if rank < pilot else "evaluation" if rank < pilot + evaluation else "unused"
        )
        entry["selection_rank"] = rank

    queue_units = []
    for entry in units:
        if entry["partition"] == "unused":
            continue
        wildtypes = []
        for wildtype_id in entry["wildtype_ids"]:
            rows, _ = select_rows(wildtype_id, sequences[wildtype_id], hits[wildtype_id], config)
            labels = labels_by_wildtype[wildtype_id]
            if len(labels) != len(rows):
                raise RuntimeError(f"{wildtype_id}: the queue changed between passes")
            representative = set(picks_by_wildtype[wildtype_id].values())
            held = set(summaries[wildtype_id]["split"]["test"])
            payloads = []
            for index, row in enumerate(rows):
                label = f"c{labels[index]:03d}"
                payload = row.payload()
                payload.update(cluster=label, split="test" if label in held else "train",
                               representative=index in representative)
                payloads.append(payload)
            wildtypes.append({"wildtype_id": wildtype_id,
                              "wildtype": sequences[wildtype_id],
                              "rows": payloads})
        queue_units.append({**entry, "wildtypes": wildtypes})

    queue = {
        "schema": SCHEMA_QUEUE,
        "protocol": config,
        "units": queue_units,
    }
    audit = {
        "schema": SCHEMA_AUDIT,
        "wildtypes": summaries,
        "family_selection": family_selection,
        "duplicate_merges": merge_evidence,
        "units": [
            {key: value for key, value in entry.items() if key != "wildtypes"} for entry in units
        ],
        "counts": {
            "wildtypes": len(sequences),
            "original_family_clusters": len(by_family),
            "qualified_wildtypes": len(qualified),
            "selected_wildtypes": len(selected),
            "independent_units": len(units),
            "pilot_units": sum(entry["partition"] == "pilot" for entry in units),
            "evaluation_units": sum(entry["partition"] == "evaluation" for entry in units),
            "unused_units": sum(entry["partition"] == "unused" for entry in units),
            "merged_units": sum(len(entry["source_wildtype_ids"]) > 1 for entry in units),
            "retained_queue_rows": sum(len(wt["rows"]) for unit in queue_units for wt in unit["wildtypes"]),
            "heldout_rows_by_partition": {
                partition: sum(len(evaluation_rows(wt)) for unit in queue_units
                               if unit["partition"] == partition for wt in unit["wildtypes"])
                for partition in ("pilot", "evaluation")},
            "null_sequences_by_partition": {
                partition: sum(len(evaluation_rows(wt)) for unit in queue_units
                               if unit["partition"] == partition for wt in unit["wildtypes"])
                * (config["null"]["column_shuffles"] + config["null"]["composition_shuffles"])
                for partition in ("pilot", "evaluation")},
        },
    }
    return queue, audit


# ------------------------------------------------------------------ statistics


def profile_frequencies(rows: Iterable[NativeRow], *, pseudocount: float) -> dict[int, np.ndarray]:
    counts: dict[int, np.ndarray] = {}
    for row in rows:
        for index, position in enumerate(row.query_index):
            if position == INSERTION:
                continue
            counts.setdefault(int(position), np.zeros(len(AA20)))[AA_INDEX[row.native[index]]] += 1.0
    return {
        position: (value + pseudocount) / (value.sum() + pseudocount * len(AA20))
        for position, value in counts.items()
    }


def profile_score(row: NativeRow, frequencies: dict[int, np.ndarray]) -> float:
    """Additive over the query columns the profile carries, and over nothing else."""

    total = 0.0
    for index, position in enumerate(row.query_index):
        distribution = frequencies.get(int(position))
        if distribution is not None:
            total += math.log(float(distribution[AA_INDEX[row.native[index]]]))
    return total


def profile_batch_audit(rows: Sequence[NativeRow], frequencies: dict[int, np.ndarray]) -> float:
    """Largest difference between a whole-batch sum and a split-batch sum."""

    total = sum(profile_score(row, frequencies) for row in rows)
    cut = len(rows) // 2
    split = sum(profile_score(row, frequencies) for row in rows[:cut]) + sum(
        profile_score(row, frequencies) for row in rows[cut:]
    )
    return abs(total - split)


def markov_model(rows: Iterable[NativeRow], *, pseudocount: float) -> dict[str, np.ndarray]:
    start = np.zeros(len(AA20))
    transitions = np.zeros((len(AA20), len(AA20)))
    for row in rows:
        start[AA_INDEX[row.native[0]]] += 1.0
        for left, right in zip(row.native, row.native[1:]):
            transitions[AA_INDEX[left], AA_INDEX[right]] += 1.0
    return {
        "start": (start + pseudocount) / (start.sum() + pseudocount * len(AA20)),
        "transitions": (transitions + pseudocount)
        / (transitions.sum(axis=1, keepdims=True) + pseudocount * len(AA20)),
    }


def markov_score(row: NativeRow, model: dict[str, np.ndarray]) -> float:
    total = math.log(float(model["start"][AA_INDEX[row.native[0]]]))
    for left, right in zip(row.native, row.native[1:]):
        total += math.log(float(model["transitions"][AA_INDEX[left], AA_INDEX[right]]))
    return total


def contrast(
    native_nll: Sequence[float] | np.ndarray,
    null_nll: Sequence[Sequence[float]] | np.ndarray,
    lengths: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """(sum log w_native - mean_k sum log w_null_k) / sum length, nats per residue.

    The sums run over the whole group and the denominator is the group's total
    length, which is the weighting the objective itself implies. Dividing each
    sequence first and averaging would break the Profile zero invariant below.
    """

    native = np.asarray(native_nll, dtype=float)
    nulls = np.asarray(null_nll, dtype=float)
    length = np.asarray(lengths, dtype=float)
    if nulls.ndim == 1:
        # One declared null draw, as the composition control has, is still a draw set.
        nulls = nulls[None, :]
    if native.ndim != 1 or nulls.ndim != 2 or nulls.shape[1] != native.size or length.shape != native.shape:
        raise ValueError("a contrast needs one native vector, its nulls and matching lengths")
    if not (np.isfinite(native).all() and np.isfinite(nulls).all() and np.isfinite(length).all()):
        raise ValueError("a contrast needs finite likelihoods and lengths")
    if native.size == 0 or length.sum() <= 0:
        raise ValueError("a contrast needs at least one scored sequence and its length")
    total_null = float(nulls.sum(axis=1).mean())
    numerator = total_null - float(native.sum())
    return {
        "nats_per_residue": numerator / float(length.sum()),
        "numerator_nats": numerator,
        "total_length": float(length.sum()),
        "native_nats": float(native.sum()),
        "null_mean_nats": total_null,
        "nulls": int(nulls.shape[0]),
    }


def per_residue_spearman(
    model_nll: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    lengths: Sequence[float] | np.ndarray,
) -> float | None:
    """Family-internal rank correlation of per-residue values; not a calibration."""

    from scipy.stats import spearmanr

    values = -np.asarray(model_nll, dtype=float) / np.asarray(lengths, dtype=float)
    reference_values = np.asarray(reference, dtype=float) / np.asarray(lengths, dtype=float)
    if values.size < 5 or np.std(values) == 0 or np.std(reference_values) == 0:
        return None
    result = spearmanr(values, reference_values).statistic
    return None if not np.isfinite(result) else float(result)


# --------------------------------------------------------------- model scoring


def native_stage():
    """Import the established native loader/forward, without running its CLI."""

    import importlib.util

    path = Path(__file__).resolve().parents[3] / "scripts/capability/stages/context_homologue.py"
    spec = importlib.util.spec_from_file_location("family_native_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("native likelihood interface is unavailable")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    return stage


def load_causal(name: str, *, device: str | None) -> Any:
    """All 33 native Readout doors; None loads declarations/tokenizers only."""

    from ..interactions.pairwise_epistasis import ARM_DTYPE, ROSTER
    from ..readouts.readout_extraction import load_readout_arm

    if name not in ROSTER:
        raise ValueError(f"{name} is not one of the retained {len(ROSTER)} roster arms")
    return load_readout_arm(name, native_stage(), device=device,
                            dtype=ARM_DTYPE.get(name, "float32"))


def checkpoint_identity(handle) -> dict[str, Any]:
    """Real checkpoint metadata, not a directory nickname or a new digest.

    Config/tokenizer declarations and tensor inventory are compared directly.
    This is operational identity, not cryptographic attestation of weight bytes.
    """

    path = Path(handle.spec.path)
    config = load_json(path / "config.json")
    tensors = [{"name": item.name, "bytes": item.stat().st_size,
                "modified_ns": item.stat().st_mtime_ns}
               for item in sorted(path.iterdir())
               if item.is_file() and item.suffix in (".safetensors", ".bin", ".pt")]
    if not tensors or not config.get("model_type"):
        raise ValueError(f"{handle.name}: checkpoint has no tensor inventory or model type")
    return {"arm": handle.name, "config": config, "tensor_files": tensors,
            "tokenizer_metadata": {name: load_json(path / name) for name in
                                   ("tokenizer_config.json", "special_tokens_map.json")
                                   if (path / name).is_file()}}


def verify_loaded(handle, declared: dict[str, Any]) -> None:
    """Refuse another loaded architecture/checkpoint or parameter precision."""

    from ..interactions.pairwise_epistasis import ARM_DTYPE

    if checkpoint_identity(handle) != declared:
        raise ValueError("loaded checkpoint differs from the predeclared checkpoint identity")
    actual = handle.model.config.to_dict()
    shape_keys = ("model_type", "vocab_size", "n_layer", "n_embd", "num_hidden_layers",
                  "hidden_size", "intermediate_size", "num_attention_heads", "n_head",
                  "n_positions", "max_position_embeddings", "seq_length", "max_seq_len")
    for key in shape_keys:
        if key in declared["config"] and actual.get(key) != declared["config"][key]:
            raise ValueError(f"{handle.name}: loaded model config differs at {key}")
    if realised_dtype(handle) != "torch." + ARM_DTYPE.get(handle.name, "float32"):
        raise ValueError(f"{handle.name}: loaded parameter dtype differs from declared policy")


def realised_dtype(handle) -> str:
    """The precision the loaded model actually carries, measured not assumed."""

    return "/".join(sorted({str(parameter.dtype) for parameter in handle.model.parameters()}))


def _nll_and_tokens(
    handle, sequences: Sequence[str], *, batch_size: int, max_len: int
) -> tuple[list[float], list[int]]:
    """Native summed likelihood only: no representation hooks or generic mask.

    Singleton forward is the retained whole-panel production setting: RITA
    refuses unequal padding and ProGen3's expert mixture is batch-dependent.
    """

    from ..readouts.readout_extraction import pack_sequence, forward_readout_rows

    if batch_size != 1:
        raise ValueError("native whole-panel likelihood requires batch_size=1")
    stage = native_stage()
    values, tokens = [], []
    for sequence in sequences:
        ids, (start, end), _ = pack_sequence(handle, sequence)
        if len(ids) > max_len or not 1 <= start < end <= len(ids):
            raise ValueError("native packed target exceeds the budget or has an invalid scored span")
        logits, tensor = forward_readout_rows(handle, [ids], stage)
        record = stage._target_nll(logits, tensor, start, end)
        value, count = float(record["nll_sum"]), int(record["scored_tokens"])
        if not math.isfinite(value) or count != end - start:
            raise RuntimeError("native likelihood is nonfinite or changed its scored span")
        values.append(value)
        tokens.append(count)
    return values, tokens


def validate_arm_partition(arm: str, partition: str, config: dict[str, Any]) -> None:
    """Refuse an arm the declaration does not place on this partition."""

    roster = [str(name) for name in config["scoring"]["arms"]]
    if arm not in roster:
        raise ValueError(f"{arm} is outside the declared {len(roster)}-arm roster")
    if partition == "pilot":
        declared = [str(name) for name in config["queue"]["pilot_models"]]
        if arm not in declared:
            raise ValueError(f"{arm} is not one of the declared pilot models {declared}")
    elif partition != "evaluation":
        raise ValueError(f"unknown partition {partition!r}")


def score_config(config: dict[str, Any], *, max_len: int, partition: str, arm: str) -> dict[str, Any]:
    """The declared fields a stored score document must still agree with."""

    return {
        "arm": arm,
        "partition": partition,
        "max_len": int(max_len),
        "column_shuffle_seeds": [int(seed) for seed in config["null"]["column_shuffle_seeds"]],
        "composition_shuffles": int(config["null"]["composition_shuffles"]),
        "score_function": config["scoring"]["score_function"],
        "normalisation": config["scoring"]["normalisation"],
    }


def validate_score_document(document: dict[str, Any], config: dict[str, Any], *,
                            partition: str, arm: str) -> None:
    """The same snapshot, native score definition and loaded-identity gate in both readers."""

    from ..interactions.pairwise_epistasis import ARM_DTYPE

    expected = score_config(config, max_len=config["scoring"]["max_len"], partition=partition, arm=arm)
    if (document.get("schema") != SCHEMA_SCORES or document.get("config") != expected
            or document.get("scientific_inputs") != INPUT_SNAPSHOT):
        raise ValueError(f"{arm}/{partition}: score document was written under another declaration")
    checkpoint = document.get("checkpoint", {})
    if (checkpoint.get("arm") != arm or not checkpoint.get("config", {}).get("model_type")
            or not checkpoint.get("tensor_files")
            or document.get("dtype") != "torch." + ARM_DTYPE.get(arm, "float32")):
        raise ValueError(f"{arm}/{partition}: missing or inconsistent loaded checkpoint identity/dtype")


def score_partition(
    root: Path,
    config: dict[str, Any],
    queue: dict[str, Any],
    *,
    arm: str,
    partition: str,
    device: str,
    batch_size: int,
    max_len: int,
    log=lambda message: None,
) -> dict[str, Any]:
    """Score one arm on one declared partition, resuming completed units."""

    validate_arm_partition(arm, partition, config)
    if batch_size != 1 or max_len != config["scoring"]["max_len"]:
        raise ValueError("scoring requires singleton batches and the declared max_len")
    snapshot = scientific_inputs(root, config, queue, create=True)
    declared = score_config(config, max_len=max_len, partition=partition, arm=arm)
    checkpoint = checkpoint_identity(load_causal(arm, device=None))
    path = root / config["output"] / "scores" / arm / f"{partition}.json"
    document: dict[str, Any] = {"schema": SCHEMA_SCORES, "config": declared,
                                "scientific_inputs": snapshot, "checkpoint": checkpoint, "units": {}}
    if path.exists():
        existing = load_json(path)
        validate_score_document(existing, config, partition=partition, arm=arm)
        if existing["checkpoint"] != checkpoint:
            raise ValueError(f"{path}: checkpoint identity differs; use a fresh output directory")
        document = existing

    units = [unit for unit in queue["units"] if unit["partition"] == partition]
    if not units:
        raise ValueError(f"the queue declares no {partition} unit")
    by_unit = {unit["unit_id"]: unit for unit in units}
    for unit_id, stored in document["units"].items():
        unit = by_unit.get(unit_id)
        if unit is None:
            raise RuntimeError(f"{path}: unit {unit_id} is not in the frozen {partition} partition")
        for wildtype in unit["wildtypes"]:
            record = stored.get("wildtypes", {}).get(wildtype["wildtype_id"])
            if record is None:
                raise RuntimeError(f"{path}: {wildtype['wildtype_id']} was not scored in {unit_id}")
            _stored_scores(record, evaluation_rows(wildtype), config, str(path))
    pending = [unit for unit in units if unit["unit_id"] not in document["units"]]
    if not pending:
        log(f"{arm}/{partition}: nothing to score, {len(document['units'])} units retained")
        return document

    handle = load_causal(arm, device=device)
    verify_loaded(handle, checkpoint)
    handle.model.eval().requires_grad_(False)
    document["dtype"] = realised_dtype(handle)
    seeds = [int(seed) for seed in config["null"]["column_shuffle_seeds"]]
    for unit in pending:
        for wildtype in unit["wildtypes"]:
            rows = evaluation_rows(wildtype)
            nulls = [column_null(rows, seed) for seed in seeds]
            for shuffled in nulls:
                verify_column_null(rows, shuffled)
            compositions = [
                composition_null(row, config["queue"]["seed"] + index)
                for index, row in enumerate(rows)
            ]
            native_values, native_tokens = _nll_and_tokens(
                handle, [row.native for row in rows], batch_size=batch_size, max_len=max_len)
            null_scores = [
                _nll_and_tokens(handle, shuffled, batch_size=batch_size, max_len=max_len)
                for shuffled in nulls
            ]
            composition_values, composition_tokens = _nll_and_tokens(
                handle, compositions, batch_size=batch_size, max_len=max_len)
            document["units"].setdefault(unit["unit_id"], {"wildtypes": {}})
            document["units"][unit["unit_id"]]["wildtypes"][wildtype["wildtype_id"]] = {
                "row_ids": [row.row_id for row in rows],
                "lengths": [len(row.native) for row in rows],
                "native_nll": native_values,
                "column_null_nll": [value for value, _ in null_scores],
                "composition_null_nll": composition_values,
                "scored_tokens": {
                    "native": native_tokens,
                    "column_null": [tokens for _, tokens in null_scores],
                    "composition_null": composition_tokens,
                },
            }
            log(f"{arm}/{partition}/{unit['unit_id']}/{wildtype['wildtype_id']}: "
                f"{len(rows)} rows scored")
        _atomic_json(path, document)
    return document


# ------------------------------------------------------------------- analysis


def _wildtype_statistics(
    wildtype: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Model-free scores of one frozen wild type, on its own training clusters."""

    statistics = config["statistics"]
    rows = evaluation_rows(wildtype)
    training = [NativeRow.of(payload) for payload in wildtype["rows"]
                if payload["split"] == "train" and payload["representative"]]
    if not training:
        raise ValueError(f"{wildtype['wildtype_id']}: no training cluster")
    profile = profile_frequencies(training, pseudocount=statistics["profile_pseudocount"])
    markov = markov_model(training, pseudocount=statistics["markov_pseudocount"])
    seeds = [int(seed) for seed in config["null"]["column_shuffle_seeds"]]
    permuted = []
    for seed in seeds:
        shuffled = column_null(rows, seed)
        verify_column_null(rows, shuffled)
        permuted.append([replace(row, native=value) for row, value in zip(rows, shuffled)])
    profile_native = np.asarray([profile_score(row, profile) for row in rows])
    profile_null = np.asarray([[profile_score(row, profile) for row in shifted] for shifted in permuted])
    markov_native = np.asarray([markov_score(row, markov) for row in rows])
    markov_null = np.asarray([[markov_score(row, markov) for row in shifted] for shifted in permuted])
    lengths = np.asarray([len(row.native) for row in rows], dtype=float)
    return {
        "rows": rows,
        "lengths": lengths,
        "profile_native": profile_native,
        "markov_native": markov_native,
        # A log-score plays the role of log w, so its negation is the nll argument.
        "profile": contrast(-profile_native, -profile_null, lengths),
        "markov": contrast(-markov_native, -markov_null, lengths),
        "profile_batch_audit": profile_batch_audit(rows, profile),
    }


def token_compression(
    native_tokens: Sequence[int] | np.ndarray,
    null_tokens: Sequence[Sequence[int]] | np.ndarray,
    lengths: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Scored-target difference between the native path and the column nulls.

    A tokenizer may spell a native sequence and its column-permuted null with a
    different number of scored targets, so their log-likelihood sums cover
    different supports. This figure is reported beside the contrast, in targets
    per native residue, and it is deliberately not a test: a large value does not
    rank models by how real their probabilities are, and it is not evidence of
    parametrised memory.
    """

    native = np.asarray(native_tokens, dtype=float)
    nulls = np.asarray(null_tokens, dtype=float)
    length = np.asarray(lengths, dtype=float)
    if nulls.ndim == 1:
        nulls = nulls[None, :]
    if native.ndim != 1 or nulls.ndim != 2 or nulls.shape[1] != native.size or length.shape != native.shape:
        raise ValueError("a token diagnostic needs one native vector, its nulls and matching lengths")
    if not (np.isfinite(native).all() and np.isfinite(nulls).all() and length.sum() > 0):
        raise ValueError("a token diagnostic needs finite token counts and a positive length")
    native_total = float(native.sum())
    null_mean = float(nulls.sum(axis=1).mean())
    return {
        "tokens_per_residue": (null_mean - native_total) / float(length.sum()),
        "native_tokens": native_total,
        "null_mean_tokens": null_mean,
        "total_residues": float(length.sum()),
        "nulls": int(nulls.shape[0]),
    }


def _stored_scores(stored: dict[str, Any], rows: Sequence[NativeRow], config: dict[str, Any],
                   label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    seeds = [int(seed) for seed in config["null"]["column_shuffle_seeds"]]
    if len(set(stored["row_ids"])) != len(stored["row_ids"]):
        raise ValueError(f"{label}: the score record repeats a row id")
    if list(stored["row_ids"]) != [row.row_id for row in rows]:
        raise ValueError(f"{label}: the score record does not match the frozen queue rows")
    lengths = np.asarray([len(row.native) for row in rows], dtype=float)
    if [int(value) for value in stored["lengths"]] != [int(value) for value in lengths]:
        raise ValueError(f"{label}: the stored lengths do not match the frozen queue")
    native = np.asarray(stored["native_nll"], dtype=float)
    nulls = np.asarray(stored["column_null_nll"], dtype=float)
    compositions = np.asarray(stored["composition_null_nll"], dtype=float)
    if nulls.shape != (len(seeds), len(rows)):
        raise ValueError(f"{label}: the column null record has the wrong shape")
    if native.shape != (len(rows),) or compositions.shape != native.shape:
        raise ValueError(f"{label}: the native/composition record has the wrong shape")
    for name, value in (("native", native), ("column", nulls), ("composition", compositions)):
        if not np.isfinite(value).all():
            raise ValueError(f"{label}: the {name} scores are not finite")
    counter = stored.get("scored_tokens")
    if not isinstance(counter, dict):
        raise ValueError(f"{label}: the score record has no scored-token counts")
    tokens = {
        "native": np.asarray(counter["native"], dtype=float),
        "column": np.asarray(counter["column_null"], dtype=float),
        "composition": np.asarray(counter["composition_null"], dtype=float),
    }
    if tokens["native"].shape != native.shape or tokens["column"].shape != nulls.shape:
        raise ValueError(f"{label}: the scored-token counts have the wrong shape")
    if tokens["composition"].shape != compositions.shape:
        raise ValueError(f"{label}: the composition scored-token counts have the wrong shape")
    if any((value < 1).any() or not np.isfinite(value).all() for value in tokens.values()):
        raise ValueError(f"{label}: a scored-token count is missing or invalid")
    return native, nulls, compositions, tokens


def analyse(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Family-equal contrasts, the declared maxT family and the descriptive layer."""

    output = root / config["output"]
    queue = load_json(output / "queue.json")
    if queue.get("schema") != SCHEMA_QUEUE:
        raise ValueError("the frozen queue is not a family-distribution queue")
    scientific_inputs(root, config, queue)
    statistics = config["statistics"]
    tolerance = float(statistics["profile_null_audit_tolerance"])
    summary: dict[str, Any] = {"schema": SCHEMA_ANALYSIS, "partitions": {}, "units": {},
                               "unscored_partitions": {}, "arms": []}
    for partition in ("pilot", "evaluation"):
        units = [unit for unit in queue["units"] if unit["partition"] == partition]
        if not units:
            continue
        documents: dict[str, dict[str, Any]] = {}
        for path in sorted((output / "scores").glob(f"*/{partition}.json")):
            document = load_json(path)
            arm = str(document["config"]["arm"])
            validate_score_document(document, config, partition=partition, arm=arm)
            if arm in documents:
                raise ValueError(f"{partition}: repeated score document for {arm}")
            documents[arm] = document
        if not documents:
            # Recorded, not analysed: a partition with no score document is an
            # unfinished stage, and the pilot is meant to be readable alone.
            summary["unscored_partitions"][partition] = (
                f"the queue declares {len(units)} {partition} units and no score document exists")
            continue
        arms = sorted(documents)

        unit_records: dict[str, Any] = {}
        model_delta: dict[str, dict[str, float]] = {arm: {} for arm in arms}
        composition_delta: dict[str, dict[str, float]] = {arm: {} for arm in arms}
        profile_max = 0.0
        profile_batch_max = 0.0
        control_deltas: dict[str, list[dict[str, Any]]] = {"profile": [], "markov": []}
        spearman: dict[str, list[float]] = {arm: [] for arm in arms}
        markov_spearman: dict[str, list[float]] = {arm: [] for arm in arms}
        token_delta: dict[str, dict[str, float]] = {arm: {} for arm in arms}
        for unit in units:
            per_arm: dict[str, dict[str, Any]] = {}
            for wildtype in unit["wildtypes"]:
                wildtype_id = wildtype["wildtype_id"]
                frozen = _wildtype_statistics(wildtype, config)
                rows, lengths = frozen["rows"], frozen["lengths"]
                record = {
                    "profile": frozen["profile"],
                    "markov": frozen["markov"],
                    "profile_batch_audit": frozen["profile_batch_audit"],
                }
                profile_max = max(profile_max, abs(frozen["profile"]["nats_per_residue"]))
                profile_batch_max = max(profile_batch_max, frozen["profile_batch_audit"])
                control_deltas["profile"].append(frozen["profile"])
                control_deltas["markov"].append(frozen["markov"])
                for arm in arms:
                    stored = documents[arm]["units"].get(unit["unit_id"], {}).get("wildtypes", {})
                    if wildtype_id not in stored:
                        raise ValueError(f"{arm}/{partition}: {wildtype_id} is unscored")
                    native, nulls, compositions, tokens = _stored_scores(
                        stored[wildtype_id], rows, config, f"{arm}/{wildtype_id}")
                    record[arm] = {
                        "column": contrast(native, nulls, lengths),
                        "composition": contrast(native, compositions, lengths),
                        "scored_token_compression": token_compression(
                            tokens["native"], tokens["column"], lengths),
                    }
                    record[arm]["spearman_profile"] = per_residue_spearman(
                        native, frozen["profile_native"], lengths)
                    record[arm]["spearman_markov"] = per_residue_spearman(
                        native, frozen["markov_native"], lengths)
                    per_arm.setdefault(arm, {})[wildtype_id] = record[arm]
                unit_records.setdefault(unit["unit_id"], {})[wildtype_id] = record
            for arm in arms:
                model_delta[arm][unit["unit_id"]] = float(
                    np.mean([per_arm[arm][name]["column"]["nats_per_residue"] for name in per_arm[arm]]))
                composition_delta[arm][unit["unit_id"]] = float(
                    np.mean([per_arm[arm][name]["composition"]["nats_per_residue"]
                             for name in per_arm[arm]]))
                token_delta[arm][unit["unit_id"]] = float(
                    np.mean([per_arm[arm][name]["scored_token_compression"]["tokens_per_residue"]
                             for name in per_arm[arm]]))
                values = [per_arm[arm][name]["spearman_profile"] for name in per_arm[arm]]
                markov_values = [per_arm[arm][name]["spearman_markov"] for name in per_arm[arm]]
                spearman[arm].extend(value for value in values if value is not None)
                markov_spearman[arm].extend(value for value in markov_values if value is not None)
        if profile_max > tolerance:
            raise RuntimeError(
                f"the Profile-zero invariant failed: the largest column-null Profile contrast is "
                f"{profile_max:.3e} > {tolerance:.0e}")
        # The gated invariant is the contrast, which cancels exactly. The raw-nats
        # batch figure measures float summation order over a total of order 1e6
        # nats, so it is reported and not gated; a real normalisation defect would
        # show up in the contrast itself.

        groups = sorted(unit_records)
        matrix = np.column_stack([[model_delta[arm][group] for group in groups] for arm in arms])
        compositions = np.column_stack([[composition_delta[arm][group] for group in groups]
                                        for arm in arms])
        declared_roster = [str(arm) for arm in config["scoring"]["arms"]]
        missing = sorted(set(declared_roster) - set(arms))
        unlisted = sorted(set(arms) - set(declared_roster))
        if unlisted:
            raise ValueError(f"{partition}: arms {unlisted} are outside the declared roster")
        record: dict[str, Any] = {
            "units": len(groups),
            "hypothesis_family": len(arms),
            "arms": arms,
            "roster": {"declared": len(declared_roster), "scored": len(arms), "missing": missing},
            "groups": groups,
        }
        if partition == "pilot":
            record.update(status="descriptive_only", reason="the pilot stage produces no formal conclusion")
            record["model_delta"] = model_delta
            record["composition_delta"] = composition_delta
        else:
            from ..core.statistics import bootstrap_unit_floor

            floor = bootstrap_unit_floor(len(groups), minimum_units=statistics["min_inference_units"])
            record["unit_floor"] = floor
            if missing:
                record.update(
                    status="descriptive_only",
                    reason=f"the declared {len(declared_roster)}-column family is incomplete: "
                           f"{len(missing)} arms are unscored, and a narrower family would be a "
                           "different, undeclared one",
                    model_delta={"mean": matrix.mean(axis=0).tolist()},
                    composition_delta={"mean": compositions.mean(axis=0).tolist()},
                )
            elif floor["degenerate"]:
                record.update(
                    status="descriptive_only",
                    reason=f"{len(groups)} family groups is below the "
                           f"{statistics['min_inference_units']}-unit inference floor",
                    model_delta={"mean": matrix.mean(axis=0).tolist()},
                    composition_delta={"mean": compositions.mean(axis=0).tolist()},
                )
            else:
                record.update(
                    status="complete",
                    inference=_family_inference(
                        matrix, arms, draws=statistics["bootstrap_draws"],
                        seed=statistics["bootstrap_seed"]),
                    composition_inference=_family_inference(
                        compositions, arms, draws=statistics["bootstrap_draws"],
                        seed=statistics["bootstrap_seed"]),
                )
        record["scored_token_diagnostic"] = {
            "definition": "(mean over the declared column nulls of the summed scored targets "
                          "- the summed scored targets of the native sequences) / total native "
                          "residues, in targets per residue",
            "per_arm": {arm: float(np.mean(list(token_delta[arm].values()))) for arm in arms},
            "per_group": token_delta,
            "status": "descriptive",
            "role": "reported beside the main contrast; no interval, no significance family",
            "reading": "a nonzero value means the native and null paths are not scored over the "
                       "same number of canonical targets, so part of the log-likelihood "
                       "difference can come from tokenizer compression rather than from residue "
                       "probabilities. It does not rank models by the reality of their "
                       "probabilities and it is not evidence of parametrised memory.",
        }
        record["controls"] = {
            "profile_zero_audit_max_abs_nats_per_residue": profile_max,
            "profile_batch_additivity_max_abs": profile_batch_max,
            "markov_mean_nats_per_residue": float(
                np.mean([entry["nats_per_residue"] for entry in control_deltas["markov"]])),
            "markov_units": len(control_deltas["markov"]),
        }
        record["spearman"] = {
            arm: {
                "profile_mean": float(np.mean(spearman[arm])) if spearman[arm] else None,
                "profile_units": len(spearman[arm]),
                "markov_mean": float(np.mean(markov_spearman[arm])) if markov_spearman[arm] else None,
                "markov_units": len(markov_spearman[arm]),
                "per_residue": True,
                "length_confounded": True,
                "probability_calibration": False,
            }
            for arm in arms
        }
        summary["partitions"][partition] = record
        summary["units"][partition] = unit_records
    summary["arms"] = sorted({arm for record in summary["partitions"].values()
                              for arm in record["arms"]})
    summary["notes"] = [
        "The contrast is a query-column permutation contrast; the null breaks the association "
        "between a query column and its inserted residues and mask, so it is not a calibrated "
        "probability and not a pure query-query higher-order term.",
        "The Profile contrast is exactly zero by construction and is the gated audit; the "
        "raw-nats batch-additivity figure beside it measures float summation order and is "
        "reported, not gated.",
        "A larger model contrast than the Markov contrast is not evidence of beating Markov; "
        "the Markov value is descriptive on the same samples.",
        "Spearman values are per-residue rank correlations within a wild type; length is a "
        "confound and they are not a probability calibration.",
        "The scored-target diagnostic is reported beside the main contrast and is not a test: "
        "tokenizer compression can contribute to the canonical-path log-likelihood difference, "
        "and neither its size nor its sign ranks models by probability truthfulness or "
        "establishes parametrised memory.",
    ]
    return summary


def _family_inference(matrix: np.ndarray, arms: Sequence[str], *, draws: int, seed: int) -> dict[str, Any]:
    """Standardized maxT as declared primary, pointwise quantiles as auxiliary."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(arms) or len(matrix) < 2 or draws < 2:
        raise ValueError("inference needs groups-by-labelled-columns and at least two draws/groups")
    if not np.isfinite(matrix).all():
        raise ValueError("inference needs a finite matrix")
    point = matrix.mean(axis=0)
    se = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
    # A column with no spread is degenerate, not infinitely precise, and `se > 0`
    # did not say so: a constant that is not exactly representable leaves a
    # rounding residue in both the standard error and the resampled deviation, and
    # their ratio then took the family maximum. The test is the shared one, so
    # `varying` keeps carrying the per-arm `status` and `degenerate_arms` below.
    varying = columns_with_spread(matrix, se)
    # Generate ONE matrix of resampled family means. Both intervals below use
    # these exact rows, not two different RNG algorithms with the same seed.
    counts = np.random.default_rng(seed).multinomial(
        len(matrix), np.full(len(matrix), 1 / len(matrix)), size=draws)
    resampled = counts @ matrix / len(matrix)
    # This panel already declares degeneracy per arm, in `status` and
    # `degenerate_arms` below, so it takes the mask rather than
    # `studentisable_family`'s refusal of a family with no spread anywhere.
    maximum = (np.max(np.abs((resampled[:, varying] - point[varying]) / se[varying]), axis=1)
               if varying.any() else np.zeros(draws))
    critical = float(np.quantile(maximum, 0.95))
    pointwise = np.quantile(resampled, [0.025, 0.975], axis=0)
    return {
        "primary": {
            "method": "shared family multinomial bootstrap; maximum absolute centered statistic/fixed SE",
            "draws": int(draws), "seed": int(seed), "critical_value": critical,
            "family_size": len(arms), "groups": int(matrix.shape[0]),
            "hypotheses": [
                {"arm": arm, "point": float(point[index]),
                 "simultaneous": [float(point[index] - critical * se[index]),
                                  float(point[index] + critical * se[index])],
                 "status": "estimated" if varying[index] else "empirical_bootstrap_degenerate"}
                for index, arm in enumerate(arms)
            ],
        },
        "auxiliary_pointwise_95": {
            "arm": list(arms), "interval": pointwise.T.tolist(),
            "note": "same resampled family means as primary; auxiliary, not the declared primary",
        },
        "degenerate_arms": [arm for arm, varies in zip(arms, varying) if not varies],
        "degeneracy_note": "Zero empirical SE is bootstrap degeneracy, not certain capability or equivalence.",
    }