"""One machine-readable table of every quantity the manuscript cites.

A quantity enters the table only by being read out of a file on disk: an admitted
analysis artifact, or --- where no artifact is retained locally --- a per-gate
record document. Nothing is typed in by hand except the locator (path, pointer,
and the semantic fields the artifact does not carry: support description, level,
verdict). A locator that does not resolve produces a recorded gap and no row, so
a quantity the manuscript needs and no file supports is absent and reported
rather than filled in.

The refusals enforced here are table-level conditions that must always hold:
every quantity carries a unit; every interval carries its resampling unit and
draw count; an estimate without an interval carries the reason it has none; an
effective (Kish) count carries the weighting convention it was computed under;
and every verdict comes from the four-outcome vocabulary, which never merges
``unresolved`` with ``not_detected``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The four outcomes the manuscript keeps apart. ``unresolved`` is a statement
#: about an instrument and ``not_detected`` a statement about a measurement that
#: resolved against zero; merging them manufactures a result in either
#: direction, so the vocabulary is closed and a fifth name is a refusal.
VERDICTS = frozenset({"supported", "not_detected", "unresolved", "measurement_limited"})

#: What a representation-level result is conditional on, and whether that
#: condition has been settled. A row is provisional while its condition is open;
#: when the experiment that decides it reports, the condition moves here and the
#: rows that carried it stop being provisional. A condition is settled by an
#: artifact, so the record names the experiment that closed it.
CONDITIONS = {
    "readout_selection": "the readout class and the extraction depth, decided by the readout-class "
                         "sweep over 132 admitted fit cells and the depth sweep over 54 fitted "
                         "(arm, panel) cells",
    "gate_representation_recomputation": "the class-axis representation recomputation across the "
                                         "gates, closed on folding stability, external confirmation, "
                                         "residue interactions and remote homology with every verdict "
                                         "standing and no resolved sign change",
    "local_context_gate_recomputation": "the local-context gate's own class-axis representation "
                                        "recomputation, closed on its full panel of 99 cells with "
                                        "every verdict standing and no resolved sign change",
}
SETTLED_CONDITIONS = frozenset({"readout_selection", "gate_representation_recomputation",
                                "local_context_gate_recomputation"})

#: Likelihood-level results are final. A representation-level result is
#: provisional only while the condition it names is open.
LEVELS = frozenset({"likelihood", "representation", "measurement", "control", "generation"})

#: ``estimate`` is a resampled point estimate and must carry an interval or a
#: reason it cannot. ``range_bound`` is one end of a min/max over a panel, which
#: is not a point estimate of one unit. ``census`` is a complete enumeration,
#: which has no resampling unit. ``effective_count`` is a Kish count and must
#: name its weighting convention. ``support_count`` sizes a support and
#: ``constant`` is a declared design value.
#: ``artifact_leaf`` is a scalar field an artifact records with no interval
#: beside it: an accounting count, a declared threshold or a fitted magnitude
#: the artifact states alone. It says what it is rather than claiming to be a
#: resampled estimate.
KINDS = frozenset(
    {"estimate", "range_bound", "census", "count", "support_count", "effective_count", "constant",
     "ratio", "artifact_leaf"}
)

SOURCE_KINDS = frozenset({"artifact", "gate_record"})

#: How a control count came to exist. A control computed while the run happened
#: and one recovered afterwards from what the run persisted are different
#: quantities, and a count of the first is not interchangeable with a count of
#: the second. Carrying this on the row is what makes a later sum across them
#: visibly wrong rather than quietly plausible.
CONTROL_PROVENANCE = frozenset({"computed_during_the_run", "derived_after_the_fact"})


class LedgerError(RuntimeError):
    """A declaration that cannot be admitted: the table would carry a hole."""


@dataclass(frozen=True)
class Support:
    """A measured support. Two quantities on different supports are not
    comparable without saying so, which is why the identifier is carried on
    every row rather than inferred from the counts."""

    support_id: str
    description: str


@dataclass
class Quantity:
    """One derived number, with everything needed to quote it."""

    id: str
    claim: str
    family: str
    value: float
    unit: str
    kind: str
    support_id: str
    interval: tuple[float, float] | None = None
    interval_kind: str | None = None
    resampling_unit: str | None = None
    resampling_draws: int | None = None
    no_interval_reason: str | None = None
    seed_set: tuple[int, ...] = ()
    level: str = "measurement"
    verdict: str | None = None
    weighting_convention: str | None = None
    conditioning: str | None = None
    control_provenance: str | None = None
    source_kind: str = "artifact"
    source_path: str = ""
    source_sha256: str = ""
    source_pointer: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.unit:
            raise LedgerError(f"{self.id}: every quantity carries a unit")
        if self.kind not in KINDS:
            raise LedgerError(f"{self.id}: unknown kind {self.kind!r}")
        if self.level not in LEVELS:
            raise LedgerError(f"{self.id}: unknown level {self.level!r}")
        if self.verdict is not None and self.verdict not in VERDICTS:
            raise LedgerError(f"{self.id}: verdict {self.verdict!r} is outside the four-outcome vocabulary")
        if self.source_kind not in SOURCE_KINDS:
            raise LedgerError(f"{self.id}: unknown source kind {self.source_kind!r}")
        if self.interval is not None:
            low, high = self.interval
            if not low <= self.value <= high:
                raise LedgerError(f"{self.id}: interval {self.interval} does not bracket {self.value}")
            if not self.resampling_unit:
                raise LedgerError(f"{self.id}: an interval carries its resampling unit")
            if not self.resampling_draws:
                raise LedgerError(f"{self.id}: an interval carries its resampling draw count")
            if not self.interval_kind:
                raise LedgerError(f"{self.id}: an interval names what kind of interval it is")
        elif self.kind == "estimate" and not self.no_interval_reason:
            raise LedgerError(f"{self.id}: a point estimate with no interval states why it has none")
        if self.control_provenance is not None and self.control_provenance not in CONTROL_PROVENANCE:
            raise LedgerError(
                f"{self.id}: unknown control provenance {self.control_provenance!r}")
        if self.conditioning is not None and self.conditioning not in CONDITIONS:
            raise LedgerError(f"{self.id}: unknown conditioning {self.conditioning!r}")
        if self.kind == "effective_count" and not self.weighting_convention:
            raise LedgerError(f"{self.id}: an effective (Kish) count names the weighting it was computed under")
        if not self.source_path or not self.source_sha256:
            raise LedgerError(f"{self.id}: a quantity carries the digest of the file it came from")

    @property
    def provisional(self) -> bool:
        """A representation-level outcome whose condition is still open.

        Defaulting to the open condition keeps a row provisional unless it says
        which settled experiment released it, so a row cannot lose the marking
        by omission.
        """
        if self.level != "representation":
            return False
        return (self.conditioning or "gate_representation_recomputation") not in SETTLED_CONDITIONS

    def as_row(self) -> dict[str, object]:
        return {
            "id": self.id,
            "claim": self.claim,
            "family": self.family,
            "value": self.value,
            "unit": self.unit,
            "kind": self.kind,
            "support_id": self.support_id,
            "interval_low": None if self.interval is None else self.interval[0],
            "interval_high": None if self.interval is None else self.interval[1],
            "interval_kind": self.interval_kind,
            "resampling_unit": self.resampling_unit,
            "resampling_draws": self.resampling_draws,
            "no_interval_reason": self.no_interval_reason,
            "seed_set": list(self.seed_set),
            "level": self.level,
            "provisional": self.provisional,
            "verdict": self.verdict,
            "weighting_convention": self.weighting_convention,
            "conditioning": self.conditioning,
            "control_provenance": self.control_provenance,
            "conditioning_settled": None if self.conditioning is None
                                    else self.conditioning in SETTLED_CONDITIONS,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_pointer": "/".join(self.source_pointer),
        }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Artifacts:
    """Reads and digests the files a quantity is resolved from, once each."""

    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.root = root.resolve()
        self._payloads: dict[str, object] = {}
        self._digests: dict[str, str] = {}
        self._texts: dict[str, str] = {}

    def _path(self, relative: str) -> Path:
        """Locate one source, which has to be a path inside the repository.

        A source outside the tree is not distributed with it, so a locator that
        leaves the repository makes the table unbuildable from a clone while
        still resolving wherever that neighbouring directory happens to exist.
        """
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise LedgerError(f"{relative}: a source outside the repository is not admissible")
        return path

    def sha256(self, relative: str) -> str:
        if relative not in self._digests:
            self._digests[relative] = digest(self._path(relative))
        return self._digests[relative]

    def json(self, relative: str) -> object:
        if relative not in self._payloads:
            content = self._path(relative).read_bytes()
            self._digests[relative] = hashlib.sha256(content).hexdigest()
            self._payloads[relative] = json.loads(content)
        return self._payloads[relative]

    def text(self, relative: str) -> str:
        if relative not in self._texts:
            content = self._path(relative).read_bytes()
            self._digests[relative] = hashlib.sha256(content).hexdigest()
            self._texts[relative] = content.decode("utf-8")
        return self._texts[relative]

    def exists(self, relative: str) -> bool:
        return self._path(relative).is_file()


def resolve(payload: object, pointer: tuple[str, ...] | list[str]) -> object:
    """Walk a JSON payload by literal key segments.

    Segments are literal keys, or ``[i]`` for a list index. Keys are taken
    whole, because arm names such as ``qwen2.5-7b`` and contrast names such as
    ``C_G_T_M1|C_G_T`` contain characters a dotted path would split on.
    """
    node = payload
    for segment in pointer:
        if segment.startswith("[") and segment.endswith("]"):
            index = int(segment[1:-1])
            if not isinstance(node, list) or not -len(node) <= index < len(node):
                raise KeyError("/".join(map(str, pointer)))
            node = node[index]
            continue
        if not isinstance(node, dict) or segment not in node:
            raise KeyError("/".join(map(str, pointer)))
        node = node[segment]
    return node


#: Why a quantity is absent. The distinction decides what it constrains: a
#: quantity that exists in the cluster store and is not staged here is a
#: retrieval problem and constrains nothing; a quantity no run computed, or that
#: no run recorded, constrains what the paper may claim.
REACHABILITY = frozenset({"cluster_only", "not_recorded", "not_measured"})


@dataclass
class Gap:
    """A quantity the manuscript needs that no retained file supports."""

    id: str
    claim: str
    family: str
    reason: str
    looked_in: str
    reachability: str = "not_recorded"
    #: How deep the search went. "No fitted estimate of any kind" and "no fitted
    #: estimate among the top-level keys" are different claims, and only one of
    #: them was true when this field was added: a gap recorded from a shallow
    #: enumeration read as an absence in the evidence. A gap that does not say
    #: how deep it looked is counted in the table's own summary.
    searched: str = ""

    def __post_init__(self) -> None:
        if self.reachability not in REACHABILITY:
            raise LedgerError(f"{self.id}: unknown reachability {self.reachability!r}")

    def as_row(self) -> dict[str, str]:
        return {
            "id": self.id,
            "claim": self.claim,
            "family": self.family,
            "reason": self.reason,
            "looked_in": self.looked_in,
            "reachability": self.reachability,
            "searched": self.searched,
        }


class Ledger:
    """The table under construction, with its supports and its gaps."""

    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts
        self.supports: dict[str, Support] = {}
        self.quantities: list[Quantity] = []
        self.gaps: list[Gap] = []
        self._ids: set[str] = set()

    def declare_support(self, support_id: str, description: str) -> str:
        existing = self.supports.get(support_id)
        if existing is not None and existing.description != description:
            raise LedgerError(f"support {support_id!r} declared twice with different descriptions")
        self.supports[support_id] = Support(support_id, description)
        return support_id

    def add(self, quantity: Quantity) -> Quantity:
        if quantity.id in self._ids:
            raise LedgerError(f"duplicate quantity id {quantity.id!r}")
        if quantity.support_id not in self.supports:
            raise LedgerError(f"{quantity.id}: support {quantity.support_id!r} is not declared")
        self._ids.add(quantity.id)
        self.quantities.append(quantity)
        return quantity

    def gap(self, *, id: str, claim: str, family: str, reason: str, looked_in: str,
            reachability: str = "not_recorded", searched: str = "") -> None:
        self.gaps.append(Gap(id=id, claim=claim, family=family, reason=reason,
                             looked_in=looked_in, reachability=reachability, searched=searched))

    def payload(self, *, scope: str) -> dict[str, object]:
        by_family: dict[str, int] = {}
        for quantity in self.quantities:
            by_family[quantity.family] = by_family.get(quantity.family, 0) + 1
        return {
            "schema": "manuscript_derived_numbers_v1",
            "scope": scope,
            "verdict_vocabulary": sorted(VERDICTS),
            "conditions": {name: {"statement": statement,
                                  "settled": name in SETTLED_CONDITIONS}
                           for name, statement in sorted(CONDITIONS.items())},
            "supports": {key: value.description for key, value in sorted(self.supports.items())},
            "counts": {
                "quantities": len(self.quantities),
                "by_family": dict(sorted(by_family.items())),
                "artifact_backed": sum(q.source_kind == "artifact" for q in self.quantities),
                "gate_record_backed": sum(q.source_kind == "gate_record" for q in self.quantities),
                "provisional": sum(q.provisional for q in self.quantities),
                "gaps": len(self.gaps),
                "gaps_without_a_stated_search_depth":
                    sum(not gap.searched for gap in self.gaps),
                "gaps_by_reachability": {
                    key: sum(gap.reachability == key for gap in self.gaps)
                    for key in sorted(REACHABILITY)
                },
            },
            "source_sha256": {
                path: self.artifacts.sha256(path)
                for path in sorted({q.source_path for q in self.quantities})
            },
            "quantities": [q.as_row() for q in sorted(self.quantities, key=lambda q: q.id)],
            "gaps": [g.as_row() for g in sorted(self.gaps, key=lambda g: g.id)],
        }

    def write(self, directory: Path, *, scope: str) -> dict[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.payload(scope=scope)
        (directory / "derived-numbers.json").write_text(json.dumps(payload, indent=2) + "\n")
        rows = payload["quantities"]
        fields = list(rows[0]) if rows else []
        with (directory / "derived-numbers.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: (";".join(map(str, value)) if isinstance(value, list) else value)
                                 for key, value in row.items()})
        (directory / "evidence-gaps.json").write_text(
            json.dumps({"schema": "manuscript_evidence_gaps_v1",
                        "rule": "a quantity the manuscript needs and no retained file supports is "
                                "absent from the table and listed here; it is never filled in from "
                                "prose. Reachability separates the two kinds: cluster_only is staged "
                                "elsewhere and constrains nothing, while not_recorded and "
                                "not_measured constrain what may be claimed",
                        "counts": payload["counts"]["gaps_by_reachability"],
                        "gaps": payload["gaps"]}, indent=2) + "\n"
        )
        return payload


def checked_sum(quantities: list[Quantity], *, field: str) -> float:
    """Sum quantities that agree on one declared property, and refuse otherwise.

    A control several units appear to share is several controls unless one
    implementation computed it. Summing their counts hides which was measured
    how, which is the aggregation form of the same confusion a denominator or a
    population hides. The refusal is the point: the caller records why the total
    does not exist rather than publishing one that averages two kinds of thing.
    """
    values = {getattr(quantity, field) for quantity in quantities}
    if len(values) > 1:
        raise LedgerError(
            "refusing to sum across rows that disagree on " + field + ": " +
            ", ".join(sorted(str(value) for value in values)) +
            "; these are several quantities and their total would hide which is which"
        )
    return sum(quantity.value for quantity in quantities)


def load_table(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "manuscript_derived_numbers_v1":
        raise LedgerError(f"{path}: not a derived-number table")
    return payload
