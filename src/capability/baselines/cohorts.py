"""Wild-type cohorts of the explicit-statistics ladder.

The R1 mutation-effect lane publishes two explicit evolutionary statistics per
mutant -- a plmc Potts energy difference and an official GEMME score -- over its
own 174 wild types. The phenotype and interaction lanes need the same two
statistics over their own reference sequences. What they share is not a result
but a *procedure*, and this module is that procedure's input half: it turns a
lane's frozen cohort file into the one roster every later stage reads, so that
R1 is one instance of the procedure rather than its hard-wired default.

A descriptor states, for one cohort:

``references``
    The distinct wild-type sequences to model, each with a stable identifier, the
    identifier its homolog search listed it under, and every cohort-native name
    that carries the same sequence. Two measured backgrounds with identical
    sequences are one reference and one Potts fit, named by the alphabetically
    first of them, which is the rule the stability cohort already uses.

``rows``
    One row per (background, mutation) the lane asks about, in the lane's own
    order, carrying the parsed substitutions. A mutation named by two backgrounds
    of the same reference is two rows and one computation: collapsing them would
    delete measurements rather than duplicates.

``hits`` and ``max_sequences``
    The retained DIAMOND hit table the alignment is rebuilt from, and the depth
    cap it is rebuilt at. The cap is declared per cohort and is R1's 5000
    everywhere, because comparability with the R1 ladder is the purpose of the
    work; it is a descriptor field so that a cohort can never inherit a depth it
    does not state.

``profile_store``
    A retained site-independent profile store, used as a *cross-check* and never
    as an input. The alignment is rebuilt from the raw hit table, so a store
    built at a different cap changes nothing in the produced features; the
    rebuild is compared against the store only for a wild type where neither cap
    binds, and the comparison is recorded as skipped where one does.

``reuse``
    References whose sequence is byte-identical to an R1 wild type that already
    carries a fitted Potts model. Those take R1's frozen alignment, so a retained
    parameter file remains exactly the model of the alignment it was fitted on.

Two limits are declared rather than repaired. One belongs to the retained
searches: R1 searched at ``--max-target-seqs 5000`` and the later cohorts at
10000, so for a wild type with more than 5000 coverage-passing subjects the two
candidate pools differ even at one depth cap, and re-running the searches is the
only fix. The other belongs to the support itself: a reference with fewer than
:data:`~.msa_baselines.MIN_POTTS_ROWS` retained homolog rows carries no model at
all, and is enumerated as a refusal with its reason rather than imputed.

No phenotype, measured value or model output is read anywhere in this module.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np

from .msa_baselines import (
    build_variant_table,
    check_assay_blocks,
    parse_substitutions,
    path_digest,
    read_fasta,
    substitution_counts,
)

#: The configuration's own shape label.
SCHEMA_CONFIG = "explicit_statistics_cohorts_v1"

#: A reference identifier is the cohort's own name for that sequence, and it
#: names a directory entry and a plmc parameter file. Retained names are kept
#: verbatim -- one stability background is spelled ``v2R31S|R32S_2N5D.pdb`` --
#: because rewriting a measured background's identity to tidy a path is exactly
#: the drift these artefacts exist to prevent. What is refused is a name that
#: cannot be one path component: a separator, whitespace or a leading dot. Every
#: stage passes the name as an argument and never through a shell.
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+|-]*$")

#: ``sample_id`` of a cohort that has no frozen identifier of its own. The pair
#: is already unique, and the separator is absent from every cohort-native name.
SAMPLE_SEPARATOR = "::"

#: The pairwise and phenotype feature stores retain Neff to four decimals, so a
#: rebuild can be compared with them to half of the last retained place and no
#: further. It is a serialization precision, not a modelling tolerance.
PROFILE_FEATURES_NEFF_PRECISION = 5e-5


@dataclass(frozen=True)
class Reference:
    """One wild-type sequence to model, and the names that ask for it."""

    wildtype_id: str
    sequence: str
    query_id: str
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not IDENTIFIER.match(self.wildtype_id):
            raise ValueError(
                f"{self.wildtype_id!r} is not usable as a reference identifier; it names an "
                "alignment file and a parameter file")
        if not self.sequence:
            raise ValueError(f"{self.wildtype_id}: empty reference sequence")


@dataclass(frozen=True)
class ProfileStore:
    """A retained profile store, read back for the rebuild's cross-check only."""

    reader: str
    paths: tuple[Path, ...]
    max_sequences: int
    _expectations: dict[str, dict]

    def expectation(self, name: str) -> dict | None:
        return self._expectations.get(name)


@dataclass(frozen=True)
class Cohort:
    """One cohort's complete input half, resolved from the configuration."""

    name: str
    title: str
    reader: str
    root: Path
    out: Path
    log_dir: Path
    fit_subdir: str
    hits: Path
    max_sequences: int
    search_max_target_seqs: int
    variant_source_label: str
    inputs: dict[str, str]
    references: tuple[Reference, ...]
    rows: tuple[dict, ...]
    profile_store: ProfileStore | None
    reuse: dict[str, dict]
    expect: dict
    protocol: dict
    profile_delta: str | None

    @property
    def by_reference(self) -> dict[str, list[dict]]:
        """The variant rows of each reference, in table order. Built on access, so
        a caller that needs it per reference binds it once."""

        grouped: dict[str, list[dict]] = {entry.wildtype_id: [] for entry in self.references}
        for row in self.rows:
            grouped[row["wildtype_id"]].append(row)
        return grouped

    @property
    def sequences(self) -> dict[str, str]:
        return {entry.wildtype_id: entry.sequence for entry in self.references}

    def counts(self) -> dict:
        counts = substitution_counts(self.rows)
        return counts | {
            "assays": len({row["assay"] for row in self.rows}),
            "wildtypes": len(self.references),
            "reused_wildtypes": len(self.reuse),
        }


# --------------------------------------------------------------- profile stores


def _retrieval_bound_store(paths: dict[str, Path], max_sequences: int) -> ProfileStore:
    """The R5 retrieval-bound store: one npz array and one record per query."""

    frequencies = np.load(paths["frequencies"], allow_pickle=False)
    records = json.loads(Path(paths["records"]).read_text())["profiles"]
    expectations = {
        name: {
            "frequencies": np.asarray(frequencies[name], dtype=np.float64),
            "n_profile_sequences": int(record["n_profile_sequences"]),
            "neff": float(record["neff"]),
            "neff_tolerance": 1e-9,
            "source": str(paths["frequencies"]),
        }
        for name, record in records.items()
        if name in frequencies
    }
    return ProfileStore("retrieval_bound", tuple(paths.values()), max_sequences, expectations)


def _profile_features_store(paths: Sequence[Path], max_sequences: int) -> ProfileStore:
    """The pairwise/phenotype feature store: ``<name>|frequencies`` and a metadata blob.

    These stores carry neither ``n_hits`` nor the saturation flags the retrieval-
    bound store carries, which is why the cross-check is expressed as the three
    quantities a rebuild can actually be compared on. They also write Neff to
    four decimals, so the comparison carries that retained precision with it
    rather than asserting a precision the file does not hold.
    """

    expectations: dict[str, dict] = {}
    declared: set[int] = set()
    for path in paths:
        arrays = np.load(path, allow_pickle=False)
        metadata = json.loads(str(arrays["metadata"]))
        declared.add(int(metadata["max_sequences"]))
        for record in metadata["backgrounds"]:
            if record["status"] != "present":
                continue
            key = f"{record['name']}|frequencies"
            if key not in arrays:
                raise ValueError(f"{path}: {record['name']} is present but carries no frequencies")
            expectations.setdefault(record["name"], {
                "frequencies": np.asarray(arrays[key], dtype=np.float64),
                "n_profile_sequences": int(record["n_profile_sequences"]),
                "neff": float(record["neff"]),
                "neff_tolerance": PROFILE_FEATURES_NEFF_PRECISION,
                "source": str(path),
            })
    if declared != {max_sequences}:
        raise ValueError(
            f"the retained stores declare max_sequences {sorted(declared)} where the descriptor "
            f"says {max_sequences}; the cross-check's validity depends on that number")
    return ProfileStore("profile_features", tuple(paths), max_sequences, expectations)


def _load_store(declaration: dict, *, root: Path) -> ProfileStore:
    reader = declaration["reader"]
    if reader == "retrieval_bound":
        return _retrieval_bound_store(
            {"frequencies": root / declaration["frequencies"],
             "records": root / declaration["records"]},
            int(declaration["max_sequences"]))
    if reader == "profile_features":
        paths = [root / declaration["frequencies"]]
        if declaration.get("also"):
            paths.append(root / declaration["also"])
        return _profile_features_store(paths, int(declaration["max_sequences"]))
    raise ValueError(f"unknown profile store reader {reader!r}")


# --------------------------------------------------------------- row assembly


def rows_of(references: Iterable[Reference],
            variants: dict[str, list[tuple[str, str]]]) -> list[dict]:
    """Variant rows for a cohort with no frozen identifier table of its own.

    ``variants`` maps a reference identifier to its ``(assay, mutation)`` pairs in
    the lane's own order. The row schema is the one the R1 export already
    publishes, so the fit, the GEMME stage and the assembly read one shape.
    """

    rows: list[dict] = []
    sequences = {entry.wildtype_id: entry.sequence for entry in references}
    for entry in references:
        pairs = variants.get(entry.wildtype_id) or []
        if not pairs:
            raise ValueError(f"{entry.wildtype_id}: a reference with no variant is not modelled")
        index_of: dict[str, int] = {}
        for assay, mutation in pairs:
            substitutions = parse_substitutions(mutation, sequences[entry.wildtype_id])
            position = index_of.get(assay, 0)
            index_of[assay] = position + 1
            rows.append({
                "sample_id": f"{assay}{SAMPLE_SEPARATOR}{mutation}",
                "assay": assay,
                "wildtype_id": entry.wildtype_id,
                "mutation": mutation,
                "variant_index": position,
                "original_index": len(rows),
                "substitutions": [[wild, site, mutated] for wild, site, mutated in substitutions],
                "n_substitutions": len(substitutions),
            })
    return rows


def validate_rows(rows: Sequence[dict], references: Sequence[Reference]) -> None:
    """Invariants every cohort's table must hold, whatever produced it."""

    sequences = {entry.wildtype_id: entry.sequence for entry in references}
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if row["original_index"] != index:
            raise ValueError(f"{row['sample_id']}: original_index is not row {index}")
        if row["sample_id"] in seen_ids:
            raise ValueError(f"duplicate sample_id {row['sample_id']}")
        seen_ids.add(row["sample_id"])
        pair = (row["assay"], row["mutation"])
        if pair in seen_pairs:
            raise ValueError(f"{row['assay']}: duplicate variant {row['mutation']!r}")
        seen_pairs.add(pair)
        sequence = sequences.get(row["wildtype_id"])
        if sequence is None:
            raise ValueError(f"{row['sample_id']}: {row['wildtype_id']} is not a cohort reference")
        stated = tuple(tuple(token) for token in row["substitutions"])
        if stated != parse_substitutions(row["mutation"], sequence):
            raise ValueError(
                f"{row['sample_id']}: the stated substitutions are not {row['mutation']!r} on "
                f"{row['wildtype_id']}")
    check_assay_blocks(rows)
    named = {row["wildtype_id"] for row in rows}
    silent = [entry.wildtype_id for entry in references if entry.wildtype_id not in named]
    if silent:
        raise ValueError(f"{len(silent)} references carry no variant row: {silent[:5]}")


def _union_references(members: Sequence[tuple[str, str]]) -> tuple[Reference, ...]:
    """One reference per distinct sequence, named by its alphabetically first member."""

    grouped: dict[str, list[str]] = {}
    for name, sequence in members:
        grouped.setdefault(sequence, []).append(name)
    references = []
    for sequence, names in grouped.items():
        aliases = tuple(sorted(set(names)))
        references.append(Reference(wildtype_id=aliases[0], sequence=sequence,
                                    query_id=aliases[0], aliases=aliases))
    return tuple(sorted(references, key=lambda entry: entry.wildtype_id))


# -------------------------------------------------------------------- readers


def _read_r1(descriptor: dict, *, root: Path,
             sample_id_of: Callable[[str, str], str] | None) -> tuple[tuple[Reference, ...], list[dict]]:
    """The frozen R1 table, validated against the measured cohort that produced it."""

    if sample_id_of is None:
        raise ValueError(
            "the r1 cohort declares the frozen cohort-sha sample_id and no identifier function "
            "was supplied; the entry point must inject it rather than re-deriving it here")
    inputs = descriptor["inputs"]
    samples = json.loads((root / inputs["samples"]).read_text())
    catalogue = json.loads((root / inputs["wildtypes"]).read_text())
    measured = json.loads((root / inputs["cohort"]).read_text())
    assays = {record["assay"]: record for record in measured["assays"]}
    if len(assays) != len(measured["assays"]):
        raise ValueError("duplicate cohort assays")
    sequences: dict[str, str] = {}
    for name, sequence in read_fasta(root / inputs["wildtypes_fasta"]):
        if name in sequences:
            raise ValueError(f"duplicate wild-type record {name}")
        sequences[name] = sequence
    rows = build_variant_table(
        samples, wildtype_of_assay=catalogue["assay_to_wildtype"], sequences=sequences,
        cohort=measured | {"assays": assays}, sample_id_of=sample_id_of)
    used = sorted({row["wildtype_id"] for row in rows})
    references = tuple(
        Reference(wildtype_id=name, sequence=sequences[name], query_id=name, aliases=(name,))
        for name in used)
    for name in used:
        for assay in {row["assay"] for row in rows if row["wildtype_id"] == name}:
            if assays[assay]["wildtype"] != sequences[name]:
                raise ValueError(f"{assay}: cohort wild type differs from the retained FASTA")
    return references, rows


def _read_domainome(descriptor: dict, *, root: Path, **_: object
                    ) -> tuple[tuple[Reference, ...], list[dict]]:
    """The Domainome abundance cohort: one assay per domain, singles only."""

    domains = json.loads((root / descriptor["inputs"]["cohort"]).read_text())["domains"]
    references = _union_references([(entry["name"], entry["wildtype"]) for entry in domains])
    owner = {alias: entry.wildtype_id for entry in references for alias in entry.aliases}
    variants: dict[str, list[tuple[str, str]]] = {}
    for domain in sorted(domains, key=lambda entry: entry["name"]):
        wildtype = domain["wildtype"]
        for variant in domain["variants"]:
            site = int(variant["position"])
            mutation = f"{wildtype[site - 1]}{site}{variant['mutant']}"
            variants.setdefault(owner[domain["name"]], []).append((domain["name"], mutation))
    return references, rows_of(references, variants)


def _read_pairwise(descriptor: dict, *, root: Path, **_: object
                   ) -> tuple[tuple[Reference, ...], list[dict]]:
    """Stability singles and interaction cycle corners over one union of references.

    An interaction cycle names its four corners as indices into the background's
    retained mutant sequences; the three non-wild-type corners are recovered by
    difference against the wild type, because a cycle contrast needs every corner
    of the same model.
    """

    inputs = descriptor["inputs"]
    stability = json.loads((root / inputs["stability_cohort"]).read_text())["backgrounds"]
    interactions = json.loads((root / inputs["interaction_plan"]).read_text())["backgrounds"]
    references = _union_references(
        [(entry["name"], entry["wildtype"]) for entry in stability]
        + [(entry["name"], entry["wildtype"]) for entry in interactions])
    owner = {alias: entry.wildtype_id for entry in references for alias in entry.aliases}
    variants: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()

    def add(name: str, mutation: str) -> None:
        # A background may be measured in both lanes under one name, and a cycle
        # names each corner once per cycle. One (background, mutation) is one
        # feature, so the second request for it is the same row, not a new one.
        if (name, mutation) in seen:
            return
        seen.add((name, mutation))
        variants.setdefault(owner[name], []).append((name, mutation))

    for background in sorted(stability, key=lambda entry: entry["name"]):
        wildtype = background["wildtype"]
        for variant in background["variants"]:
            site = int(variant["position"])
            add(background["name"], f"{wildtype[site - 1]}{site}{variant['mutant']}")
    for background in sorted(interactions, key=lambda entry: entry["name"]):
        wildtype = background["wildtype"]
        for cycle in background["cycles"]:
            for state in cycle["states"]:
                mutation = _difference(wildtype, background["sequences"][int(state)])
                if mutation is not None:
                    add(background["name"], mutation)
    return references, rows_of(references, variants)


def _read_phenotype(descriptor: dict, *, root: Path, **_: object
                    ) -> tuple[tuple[Reference, ...], list[dict]]:
    """The phenotype-breadth catalogue, joined to its admitted measured rows."""

    inputs = descriptor["inputs"]
    catalogue = json.loads((root / inputs["catalogue"]).read_text())
    references = _union_references([(entry["WT_name"], entry["sequence"]) for entry in catalogue])
    sequences = {entry.wildtype_id: entry.sequence for entry in references}
    variants: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for path in sorted(Path(root / inputs["rows"]).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            name = f"{row['cohort']}:{row['wildtype_sha256'][:16]}"
            if name not in sequences:
                continue
            site, wild, mutant = int(row["position"]), row["wt_aa"], row["mt_aa"]
            if sequences[name][site - 1] != wild:
                raise ValueError(
                    f"{row['row_id']}: {name} carries {sequences[name][site - 1]} at {site}, the "
                    f"measured row says {wild}")
            mutation = f"{wild}{site}{mutant}"
            key = (row["background"], mutation)
            if key in seen:
                continue
            seen.add(key)
            variants.setdefault(name, []).append((row["background"], mutation))
    for name in variants:
        variants[name].sort()
    return references, rows_of(references, variants)


def _difference(wildtype: str, mutant: str) -> str | None:
    """The colon-joined mutation string of one mutant sequence, or None for the wild type."""

    if len(wildtype) != len(mutant):
        raise ValueError("a cohort mutant sequence is not the wild type's length")
    sites = [f"{wild}{index + 1}{changed}"
             for index, (wild, changed) in enumerate(zip(wildtype, mutant)) if wild != changed]
    return ":".join(sites) if sites else None


READERS: dict[str, Callable[..., tuple[tuple[Reference, ...], list[dict]]]] = {
    "r1_frozen_samples": _read_r1,
    "domainome_abundance": _read_domainome,
    "pairwise_union": _read_pairwise,
    "phenotype_breadth": _read_phenotype,
}


# ---------------------------------------------------------------------- reuse


def _reuse_map(declaration: dict, references: Sequence[Reference], *, root: Path) -> dict[str, dict]:
    """References byte-identical to an R1 wild type that already carries a fitted model.

    Reuse is keyed on the sequence, not on a name: the two cohorts were drawn
    from different sources and share nothing else. A reused reference takes R1's
    frozen alignment, which is what makes a retained parameter file still the
    model of the alignment it was fitted on.
    """

    scores = json.loads((root / declaration["scores"]).read_text())
    exported = json.loads(
        (root / declaration["export"] / "inputs" / "support.json").read_text())
    support = exported["wildtypes"]
    sequences = dict(read_fasta(root / declaration["wildtypes_fasta"]))
    fitted = {sequences[name]: name for name in sorted(scores["backgrounds"]) if name in sequences}
    reuse: dict[str, dict] = {}
    for entry in references:
        name = fitted.get(entry.sequence)
        if name is None:
            continue
        record = scores["backgrounds"][name]
        reuse[entry.wildtype_id] = {
            "cohort": declaration["cohort"],
            "wildtype_id": name,
            "max_sequences": int(exported["protocol"]["max_sequences"]),
            "support": support[name],
            "status": record["status"],
            "selected_lambda_e": record["selected_lambda_e"],
            "regularization_policy": record["regularization_policy"],
            "params": record.get("params"),
            "scores": str(declaration["scores"]),
        }
    return reuse


# ------------------------------------------------------------------- loading


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text())
    if config.get("schema") != SCHEMA_CONFIG:
        raise ValueError(f"{path}: schema {config.get('schema')!r} is not {SCHEMA_CONFIG!r}")
    return config


def load_cohort(name: str, *, config_path: Path, root: Path,
                sample_id_of: Callable[[str, str], str] | None = None) -> Cohort:
    """Resolve one cohort descriptor, refusing anything it cannot mean."""

    root = Path(root)
    config = load_config(config_path)
    if name not in config["cohorts"]:
        raise SystemExit(
            f"unknown cohort {name!r}; {Path(config_path).name} declares "
            f"{sorted(config['cohorts'])}")
    descriptor = config["cohorts"][name]
    reader = descriptor["reader"]
    if reader not in READERS:
        raise ValueError(f"{name}: unknown reader {reader!r}")
    for label, path in descriptor["inputs"].items():
        if not (root / path).exists():
            raise FileNotFoundError(f"{name}: input {label} is missing at {root / path}")
    cap = int(descriptor["max_sequences"])
    if cap < 1:
        raise ValueError(f"{name}: max_sequences must be positive")
    references, rows = READERS[reader](descriptor, root=root, sample_id_of=sample_id_of)
    validate_rows(rows, references)
    store = _load_store(descriptor["profile_store"], root=root) if descriptor.get(
        "profile_store") else None
    reuse = _reuse_map(descriptor["reuse"], references, root=root) if descriptor.get(
        "reuse") else {}
    cohort = Cohort(
        name=name,
        title=descriptor["title"],
        reader=reader,
        root=root,
        out=root / descriptor["out"],
        log_dir=root / descriptor["log_dir"],
        fit_subdir=descriptor["fit_subdir"],
        hits=root / descriptor["hits"],
        max_sequences=cap,
        search_max_target_seqs=int(descriptor["search_max_target_seqs"]),
        variant_source_label=descriptor["variant_source_label"],
        inputs=dict(descriptor["inputs"]),
        references=references,
        rows=tuple(rows),
        profile_store=store,
        reuse=reuse,
        expect=dict(descriptor.get("expect") or {}),
        protocol=dict(config["protocol"]),
        profile_delta=descriptor.get("profile_delta"),
    )
    _check_expectations(cohort)
    return cohort


#: Declared quantities the roster alone cannot measure. ``refusals`` depends on
#: the retained homolog rows, so the export stage checks it against the same
#: descriptor once it has read them.
DEFERRED_EXPECTATIONS = ("refusals",)


def _check_expectations(cohort: Cohort) -> None:
    """A descriptor's declared shape is a refusal, not a comment."""

    counts = cohort.counts()
    measured = {
        "samples": counts["samples"],
        "single_substitution": counts["single_substitution"],
        "multiple_substitution": counts["multiple_substitution"],
        "assays": counts["assays"],
        "wildtypes": counts["wildtypes"],
        "references": counts["wildtypes"],
        "reusable": counts["reused_wildtypes"],
    }
    drift = {key: (value, measured[key]) for key, value in cohort.expect.items()
             if key in measured and measured[key] != value}
    unknown = sorted(set(cohort.expect) - set(measured) - set(DEFERRED_EXPECTATIONS))
    if unknown:
        raise ValueError(f"{cohort.name}: the descriptor expects unknown quantities {unknown}")
    if drift:
        raise ValueError(
            f"{cohort.name}: the cohort no longer has the shape its descriptor declares -- "
            + "; ".join(f"{key} declared {want}, measured {got}" for key, (want, got) in drift.items()))


def input_record(cohort: Cohort, *, extra: dict[str, str] | None = None) -> dict:
    """The manifest's input block: every byte the export depends on, by digest."""

    paths = dict(cohort.inputs) | dict(extra or {})
    record = {}
    for label, path in paths.items():
        resolved = cohort.root / path
        record[label] = {"path": str(path), "sha256": path_digest(resolved)}
        if resolved.is_dir():
            record[label]["directory_files"] = sorted(
                entry.name for entry in resolved.iterdir() if entry.is_file())
            record[label]["digest_rule"] = "sha256 of the sorted 'name sha256' listing"
    return record
