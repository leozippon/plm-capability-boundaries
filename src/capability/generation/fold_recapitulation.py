"""E20 fold recapitulation: input contract, per-sequence controls and measurement.

Fixed-length generation in each domain's own wild-type frame means generated
residue *i* is wild-type residue *i* by construction. Every comparison here is
therefore at a **known** correspondence: the primary endpoint is
``fold_geometry.tm_score`` between the ESMFold2 structure of the generated
sequence and the ESMFold2 structure of the wild type, and every identity is a
column-for-column match. No sequence or structure alignment is inferred.

Input contract ``e20_generated_sequence_v1``
============================================

One flat JSON object per line. All keys are required; a missing key, an unknown
key, a wrong type or an inconsistent record is refused, never skipped.

``schema_version``  the literal contract name.
``id``              unique non-empty string.
``domain``          one of the twelve domains (``1E0L`` or ``1E0L.pdb``).
``model``           non-empty string: the PLM, except that arms P and H are model-independent and
                    carry exactly ``potts-joint`` and ``natural`` (``MODEL_FREE_ARMS``), labels no
                    other arm may use.
``arm``             one of ``N M C F Pr Cx Cd P H``. Factorial: N native, M Potts
                    fields only, C couplings only, F fields and couplings. Outside it:
                    Pr independent-site profile guidance, Cx permuted-column couplings
                    (primary incorrect-constraint control), Cd cross-domain couplings
                    (secondary), P Potts-only sampling, H held-out natural rows.
``lambda``          finite number; exactly 0 for N, P, H and > 0 for M, C, F, Pr, Cx, Cd.
``seed``            integer.
``sequence``        canonical 20-letter residues, length equal to the wild type.
``native_loglik``   finite number: full-sequence native log-likelihood; null only for P, H
                    (they have no language model, so also no ``lambda_fields`` /
                    ``lambda_coupling``: both null).
``guidance_score``  finite number; null only for N, P, H.
``mean_step_entropy`` finite number >= 0: mean per-step entropy; null only for P, H.
``attempt_budget``  integer >= 1; identical for every record of a (model, domain, arm).

Optional strength pair ``lambda_fields`` / ``lambda_coupling`` (both or neither): the two
doses of a guided setting, finite and >= 0. ``lambda`` is then their nominal single number
(the coupling dose if the arm carries couplings, else the fields dose, else 0) and the pair
must be consistent with the arm. The development sweep runs one arm at several strengths,
so a **cell** is ``(model, domain, arm, lambda_fields, lambda_coupling)``; ``lambda`` and
``attempt_budget`` must be constant within it. Without a pair the key carries ``None``s.
The evaluation phase has one frozen strength per (model, arm), declared by the protocol's
``strengths`` list (``parse_strength_declaration``): ``frozen_strength`` is the only pair the primary
contrasts read; ``dose_response_strengths`` are extra pairs that are descriptive only.
``require_frozen_strength`` refuses any strength the protocol did not declare, because reading sealed
domains at an undeclared strength is selection, and ``frozen_rows`` is the only way to the primary rows.

Measured quantities are three distinct endpoints and are never combined:
``tm_score`` (primary), ``plddt_mean`` and ``ptm`` (secondary).
Nothing here is stability, function or experimental validation.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.io import sha256_file
from ..ladder import fold_geometry
from . import structure_evidence as se

SCHEMA_VERSION = "e20_generated_sequence_v1"
#: Composition vectors use this order (ARND..., the generated_phenotype and BLOSUM62
#: order). core/amino_acids.AA20 (ACDE...) is a different order used by profiles and
#: Potts; tests assert both, and no vector built here may be mixed with those.
AA = se.AA_ORDER
assert AA == "ARNDCQEGHILKMFPSTWYV"
ARMS = ("N", "M", "C", "F", "Pr", "Cx", "Cd", "P", "H")
#: Model-independent reference arms and the one ``model`` label each must carry. They have no
#: language model, hence no native score, entropy or strength pair; ``NULL_OK`` and
#: ``validate_records`` derive every null exemption from this mapping and from nothing else.
MODEL_FREE_ARMS = {"P": "potts-joint", "H": "natural"}
GUIDED_ARMS = ("M", "C", "F", "Pr", "Cx", "Cd")
FIELDS = (
    "schema_version", "id", "domain", "model", "arm", "lambda", "seed", "sequence",
    "native_loglik", "guidance_score", "mean_step_entropy", "attempt_budget",
)
#: Arms for which a field is legitimately null (no PLM, or no guidance).
NULL_OK = {
    "native_loglik": set(MODEL_FREE_ARMS),
    "mean_step_entropy": set(MODEL_FREE_ARMS),
    "guidance_score": {"N", *MODEL_FREE_ARMS},
}

#: A MSA row is a candidate homolog for novelty only if it covers this fraction
#: of the domain; a short fragment matching exactly says little about copying and
#: would otherwise dominate the maximum.
MIN_MSA_COVERAGE = 0.5
#: Declared novelty ceiling: percent identity to every MSA row at or below this.
NOVELTY_CEILING = 70.0
#: Declared primary-endpoint stratum: candidates with identity to the wild type strictly below this
#: percent. Sequence recovery against the wild type is essentially complete by about 64% identity, so
#: the 70% MSA ceiling above admits copies of the target and does not decontaminate the endpoint.
IDENTITY_TO_WT_CEILING = 20.0
#: A domain-cell with fewer admissible candidates than this has no value in that stratum: it is
#: reported missing, never imputed and never silently included.
MIN_ADMISSIBLE_PER_CELL = 25
#: Two sequences of one cell are near-duplicates at or above this identity.
NEAR_DUPLICATE_IDENTITY = 90.0

#: Repo-relative defaults for local runs only. data/ is not in the H200 code
#: snapshot, so the pod must pass these explicitly; library functions never
#: resolve a repo root themselves.
DEFAULT_WT_FASTA = Path("data/pairwise_assets/same12_provenance_20260924/query_12.faa")
DEFAULT_MSA_DIR = Path("data/pairwise_assets/same12_provenance_20260924/msa")
DEFAULT_PROTOCOL = Path("data/r6_constraint_response_20261013/diagnostic_protocol.json")
REFERENCE_SCHEMA = "e20_wild_type_references_v1"


OPTIONAL_FIELDS = ("lambda_fields", "lambda_coupling")
#: (fields dose positive?, coupling dose positive?) each arm must carry.
STRENGTH_SHAPE = {
    "N": (False, False), "P": (False, False), "H": (False, False),
    "M": (True, False), "Pr": (True, False),
    "C": (False, True), "Cx": (False, True), "Cd": (False, True),
    "F": (True, True),
}


class ContractError(ValueError):
    """An input record violates the declared contract."""


def strength_pair(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    return (row.get("lambda_fields"), row.get("lambda_coupling"))


def cell_key(row: Mapping[str, Any]) -> tuple:
    """(model, domain, arm, lambda_fields, lambda_coupling): the grouping unit everywhere."""

    return (row["model"], row["domain"], row["arm"], *strength_pair(row))


def _declared_pair(value: Any, where: str) -> tuple[tuple, int | None]:
    """``((lambda_fields, lambda_coupling), pool_size or None)`` of one declared strength object."""

    if not isinstance(value, Mapping) or not set(OPTIONAL_FIELDS) <= set(value) \
            or set(value) - {*OPTIONAL_FIELDS, "pool_size"}:
        raise ContractError(f"{where}: a strength is an object with {list(OPTIONAL_FIELDS)} and optionally pool_size")
    pair = tuple(value[k] for k in OPTIONAL_FIELDS)
    size = value.get("pool_size")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
        raise ContractError(f"{where}: pool_size must be an integer >= 1")
    if pair == (None, None):
        return pair, size
    if not all(_finite(v) and v >= 0 for v in pair):
        raise ContractError(f"{where}: a strength pair is two finite numbers >= 0, or two nulls")
    return tuple(float(v) for v in pair), size


def parse_strength_declaration(entries: Any) -> dict[tuple, tuple]:
    """``(model, arm) -> (frozen pair, dose-response pairs, {pair: declared pool size})``.

    Each entry is ``{"model", "arm", "frozen_strength", "dose_response_strengths"?}`` and a strength object may
    carry the cell's declared ``pool_size`` (depth may differ between strengths and arms by design; the merge
    checks each cell against its own declaration). The single ``frozen_strength`` is the only pair eligible
    for the primary contrasts; ``dose_response_strengths`` are descriptive only. Refused: a repeated
    (model, arm), a missing or non-object ``frozen_strength``, a repeated dose, and a pair in both places.
    """

    if not isinstance(entries, list) or not entries:
        raise ContractError("protocol.strengths must be a non-empty list of per-(model, arm) declarations")
    out: dict[tuple, tuple] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not {"model", "arm", "frozen_strength"} <= set(entry) \
                or set(entry) - {"model", "arm", "frozen_strength", "dose_response_strengths"}:
            raise ContractError(f"protocol.strengths entry {entry!r}: keys must be model, arm, frozen_strength "
                                "and optionally dose_response_strengths")
        key = (entry["model"], entry["arm"])
        where = f"protocol.strengths[{key[0]}|{key[1]}]"
        if key[1] not in ARMS or key in out:
            raise ContractError(f"{where}: unknown arm or declared twice")
        frozen, frozen_size = _declared_pair(entry["frozen_strength"], f"{where}.frozen_strength")
        raw = entry.get("dose_response_strengths", [])
        if not isinstance(raw, list):
            raise ContractError(f"{where}.dose_response_strengths must be a list")
        parsed = [_declared_pair(d, f"{where}.dose_response_strengths") for d in raw]
        doses = tuple(p for p, _ in parsed)
        if len(set(doses)) != len(doses):
            raise ContractError(f"{where}: a dose-response strength is declared twice")
        if frozen in doses:
            raise ContractError(f"{where}: strength {frozen} is both the frozen strength and a dose-response strength")
        sizes = {p: n for p, n in [(frozen, frozen_size), *parsed] if n is not None}
        out[key] = (frozen, doses, sizes)
    return out


def require_frozen_strength(rows: Iterable[Mapping[str, Any]], declared: Any) -> None:
    """Evaluation guard: every strength in the records was declared, and each arm's frozen one is present.

    A (model, arm) without a declaration, a strength outside ``{frozen_strength} + dose_response_strengths``
    and a frozen strength with no records are all refused: a dose that the protocol did not pre-declare
    as frozen can never reach the primary contrasts, however the records were produced.
    """

    declaration = parse_strength_declaration(declared)
    seen: dict[tuple, set] = defaultdict(set)
    for row in rows:
        seen[(row["model"], row["arm"])].add(strength_pair(row))
    for key, pairs in sorted(seen.items(), key=str):
        if key not in declaration:
            raise ContractError(f"(model, arm) {key} has records but no declared frozen_strength")
        frozen, doses, _ = declaration[key]
        undeclared = sorted(pairs - {frozen, *doses}, key=str)
        if undeclared:
            raise ContractError(
                f"the strength was not frozen: (model, arm) {key} carries undeclared strength pair(s) "
                f"{undeclared}; evaluation domains must not be read at a strength the protocol did not declare"
            )
        if frozen not in pairs:
            raise ContractError(f"(model, arm) {key}: no records at its frozen_strength {frozen}")


def frozen_rows(rows: Iterable[Mapping[str, Any]], declared: Any) -> list[Mapping[str, Any]]:
    """The rows at each (model, arm)'s frozen strength: the only rows a primary contrast may read."""

    declaration = parse_strength_declaration(declared)
    return [r for r in rows if strength_pair(r) == declaration[(r["model"], r["arm"])][0]]


def domain_id(name: str) -> str:
    return name[:-4] if name.endswith(".pdb") else name


def read_fasta(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, list[str]]] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            entries.append((line[1:].strip(), []))
        elif line.strip():
            entries[-1][1].append(line.strip())
    return [(name, "".join(parts)) for name, parts in entries]


def require_asset(path: Path, option: str, *, directory: bool = False) -> Path:
    """Fail early, naming the option that supplies a missing asset."""

    path = Path(path)
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(
            f"missing {'directory' if directory else 'file'} {path}; stage it and pass it with {option}"
        )
    return path


def load_wild_types(fasta: Path, option: str = "--wild-type-fasta") -> dict[str, str]:
    wild = {domain_id(name): seq for name, seq in read_fasta(require_asset(fasta, option))}
    if len(wild) != 12:
        raise ValueError(f"expected twelve wild-type domains, found {len(wild)}")
    return wild


def load_strengths(protocol: Path, option: str = "--protocol") -> Any:
    """The protocol's ``strengths`` declaration, raw; ``parse_strength_declaration`` validates it."""

    declared = json.loads(require_asset(protocol, option).read_text()).get("strengths")
    if declared is None:
        raise ContractError(f"{protocol} declares no ``strengths``; the evaluation phase needs a frozen_strength per arm")
    return declared


def load_roles(protocol: Path, option: str = "--protocol") -> dict[str, str]:
    roles = json.loads(require_asset(protocol, option).read_text())["roles"]
    return {domain_id(d): role for role in ("development", "evaluation") for d in roles[role]}


# ----------------------------------------------------------------- contract


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_records(rows: Sequence[Mapping[str, Any]], wild_types: Mapping[str, str]) -> list[dict[str, Any]]:
    """Refuse the whole set on the first violation; return normalised records."""

    out: list[dict[str, Any]] = []
    ids: set[str] = set()
    cells: dict[tuple, tuple] = {}
    for n, raw in enumerate(rows, 1):
        where = f"record {n}"
        if not isinstance(raw, Mapping):
            raise ContractError(f"{where}: not an object")
        missing = [k for k in FIELDS if k not in raw]
        extra = [k for k in raw if k not in FIELDS and k not in OPTIONAL_FIELDS]
        if missing or extra:
            raise ContractError(f"{where}: missing fields {missing}, unknown fields {extra}")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ContractError(f"{where}: schema_version {raw['schema_version']!r} != {SCHEMA_VERSION}")
        row = dict(raw)
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ContractError(f"{where}: id must be a unique non-empty string")
        ids.add(row["id"])
        where = f"record {row['id']}"
        row["domain"] = domain_id(str(row["domain"]))
        if row["domain"] not in wild_types:
            raise ContractError(f"{where}: unknown domain {row['domain']!r}")
        if not isinstance(row["model"], str) or not row["model"]:
            raise ContractError(f"{where}: model must be a non-empty string")
        arm = row["arm"]
        if arm not in ARMS:
            raise ContractError(f"{where}: arm {arm!r} not in {ARMS}")
        if row["model"] != MODEL_FREE_ARMS.get(arm, row["model"]) or (
                arm not in MODEL_FREE_ARMS and row["model"] in MODEL_FREE_ARMS.values()):
            raise ContractError(f"{where}: model {row['model']!r} is reserved for, and required by, the "
                                f"model-independent arms {MODEL_FREE_ARMS}")
        if not _finite(row["lambda"]) or (row["lambda"] > 0) != (arm in GUIDED_ARMS) or row["lambda"] < 0:
            raise ContractError(f"{where}: lambda {row['lambda']!r} inconsistent with arm {arm}")
        if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
            raise ContractError(f"{where}: seed must be an integer")
        if not isinstance(row["attempt_budget"], int) or isinstance(row["attempt_budget"], bool) or row["attempt_budget"] < 1:
            raise ContractError(f"{where}: attempt_budget must be an integer >= 1")
        for key, nullable_for in NULL_OK.items():
            if row[key] is None:
                if arm not in nullable_for:
                    raise ContractError(f"{where}: {key} is null for arm {arm}")
            elif not _finite(row[key]):
                raise ContractError(f"{where}: {key} must be a finite number")
        if row["mean_step_entropy"] is not None and row["mean_step_entropy"] < 0:
            raise ContractError(f"{where}: mean_step_entropy is negative")
        sequence = row["sequence"]
        if not isinstance(sequence, str) or set(sequence) - set(AA):
            raise ContractError(f"{where}: sequence must use the 20 canonical residues only")
        expected = len(wild_types[row["domain"]])
        if len(sequence) != expected:
            raise ContractError(
                f"{where}: length {len(sequence)} != wild-type length {expected} for "
                f"{row['domain']}; fixed-length generation is refused, not skipped"
            )
        pair = [row.get(k) for k in OPTIONAL_FIELDS]
        if (pair[0] is None) != (pair[1] is None):
            raise ContractError(f"{where}: lambda_fields and lambda_coupling come together or not at all")
        if arm in MODEL_FREE_ARMS and pair[0] is not None:
            raise ContractError(f"{where}: arm {arm} has no language model, so its strength pair must be null")
        if pair[0] is not None:
            if not all(_finite(v) and v >= 0 for v in pair):
                raise ContractError(f"{where}: strength pair must be finite and >= 0")
            if (pair[0] > 0, pair[1] > 0) != STRENGTH_SHAPE[arm]:
                raise ContractError(f"{where}: strength pair {tuple(pair)} inconsistent with arm {arm}")
            nominal = pair[1] if pair[1] > 0 else pair[0]
            if row["lambda"] != nominal:
                raise ContractError(f"{where}: lambda {row['lambda']} is not the nominal dose {nominal} of the pair")
        cell = cell_key(row)
        signature = (row["lambda"], row["attempt_budget"])
        if cells.setdefault(cell, signature) != signature:
            raise ContractError(f"{where}: lambda/attempt_budget differ within cell {cell}")
        out.append(row)
    if not out:
        raise ContractError("no records")
    return out


def load_records(path: Path, wild_types: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return validate_records(rows, wild_types)


# ----------------------------------------------------------------- controls


def _codes(sequence: str) -> np.ndarray:
    return np.frombuffer(sequence.encode(), dtype=np.uint8)


def identity(a: str, b: str) -> float:
    """Percent identity at the identity correspondence of two equal-length sequences."""

    if len(a) != len(b) or not a:
        raise ValueError("identity needs two non-empty sequences of equal length")
    return 100.0 * float(np.mean(_codes(a) == _codes(b)))


def composition(sequences: Iterable[str]) -> np.ndarray:
    """Pooled residue frequencies over the 20 canonical letters (gaps and others ignored)."""

    counts = np.zeros(len(AA))
    for sequence in sequences:
        for i, letter in enumerate(AA):
            counts[i] += sequence.count(letter)
    if counts.sum() == 0:
        raise ValueError("no canonical residues to take a composition of")
    return counts / counts.sum()


def composition_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance, in [0, 1]."""

    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def load_msa(msa_dir: Path, domain: str, length: int, option: str = "--msa-dir") -> list[str]:
    """Gapped MSA rows of one domain, validated to the domain's column count."""

    path = require_asset(Path(msa_dir) / f"{domain}.pdb.fasta", option)
    rows = [seq for _, seq in read_fasta(path)]
    bad = [i for i, row in enumerate(rows) if len(row) != length]
    if bad:
        raise ValueError(f"{path}: {len(bad)} rows do not have {length} columns")
    return rows


def max_msa_identity(sequence: str, msa: Sequence[str]) -> float:
    """Highest percent identity to any MSA row.

    Identity is matches over the row's non-gap columns, for rows covering at
    least ``MIN_MSA_COVERAGE`` of the domain. The MSA includes the wild type, so
    a copy of the wild type is never novel.
    """

    query = _codes(sequence)
    best = 0.0
    for row in msa:
        codes = _codes(row.upper())
        aligned = codes != ord("-")
        n = int(aligned.sum())
        if n < MIN_MSA_COVERAGE * len(sequence):
            continue
        best = max(best, 100.0 * float(np.sum((codes == query) & aligned)) / n)
    return best


def pairwise_identity_matrix(sequences: Sequence[str]) -> np.ndarray:
    onehot = np.stack([
        np.stack([_codes(s) == ord(a) for a in AA]).astype(np.float32).ravel() for s in sequences
    ])
    return 100.0 * (onehot @ onehot.T) / len(sequences[0])


def diversity(sequences: Sequence[str]) -> dict[str, float]:
    """Within-cell mean pairwise identity and near-duplicate rate."""

    n = len(sequences)
    if n < 2:
        raise ValueError("diversity needs at least two sequences")
    ident = pairwise_identity_matrix(sequences)
    off = ~np.eye(n, dtype=bool)
    return {
        "mean_pairwise_identity": float(ident[off].mean()),
        "near_duplicate_rate": float(((ident >= NEAR_DUPLICATE_IDENTITY - 1e-6) & off).any(axis=1).mean()),
    }


# ---------------------------------------------------------- structure access


def load_structure(tree: Path, sequence: str) -> dict[str, Any]:
    """The folded object of one sequence from a run_structure_evidence tree."""

    tree = Path(tree)
    sha = se.sequence_digest(sequence)
    result = se.load_result(tree / "objects" / sha, se.tree_signature(tree), sha)
    if result is None or result["status"] != "ok":
        raise FileNotFoundError(f"no successful fold for sequence {sha[:12]} under {tree}: {result and result.get('reason')}")
    return {
        "object": f"objects/{sha}", "pdb": tree / "objects" / sha / "prediction.pdb",
        "plddt_mean": result["mean_ca_plddt"], "ptm": result["ptm"],
        "pdb_sha256": result["files_sha256"]["prediction.pdb"],
    }


def load_reference_manifest(path: Path) -> dict[str, Any]:
    """Wild-type reference manifest, with every stored PDB re-hashed."""

    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != REFERENCE_SCHEMA:
        raise ValueError(f"{path}: not a {REFERENCE_SCHEMA} manifest")
    for domain, entry in manifest["domains"].items():
        pdb = path.parent / entry["pdb"]
        if sha256_file(pdb) != entry["pdb_sha256"]:
            raise ValueError(f"reference structure for {domain} does not match its recorded hash")
    return manifest


def measure_records(
    records: Sequence[Mapping[str, Any]], *, wild_types: Mapping[str, str], msas: Mapping[str, Sequence[str]],
    tree: Path, reference_manifest: Path,
) -> list[dict[str, Any]]:
    """Per-sequence endpoints and controls. Raises on the first unfolded sequence."""

    manifest = load_reference_manifest(reference_manifest)
    if se.tree_signature(Path(tree)) != manifest["evaluation_signature"]:
        raise ValueError(
            "generated and wild-type folds carry different instrument settings "
            "(evaluation signature); the primary comparison would mix two configurations"
        )
    traces = {
        d: fold_geometry.parse_ca_trace((Path(reference_manifest).parent / e["pdb"]).read_text())
        for d, e in manifest["domains"].items()
    }
    family = {d: composition(r.replace("-", "").upper() for r in msa) for d, msa in msas.items()}
    out = []
    for rec in records:
        d = rec["domain"]
        wt = wild_types[d]
        if len(rec["sequence"]) != len(wt):
            raise ContractError(f"{rec['id']}: length differs from wild type")
        fold = load_structure(tree, rec["sequence"])
        mobile = fold_geometry.parse_ca_trace(Path(fold["pdb"]).read_text())
        row = {k: rec[k] for k in FIELDS if k not in ("schema_version", "sequence")}
        row.update(zip(OPTIONAL_FIELDS, strength_pair(rec)))
        row.update(
            sequence=rec["sequence"], length=len(rec["sequence"]),
            tm_score=fold_geometry.tm_score(mobile, traces[d])["tm_score"],
            plddt_mean=fold["plddt_mean"], ptm=fold["ptm"],
            structure_object=fold["object"], structure_pdb_sha256=fold["pdb_sha256"],
            composition_tv_to_family=composition_distance(composition([rec["sequence"]]), family[d]),
            composition_tv_to_wild_type=composition_distance(composition([rec["sequence"]]), composition([wt])),
            max_msa_identity=max_msa_identity(rec["sequence"], msas[d]),
            identity_to_wild_type=identity(rec["sequence"], wt),
        )
        out.append(row)
    return out


def cell_controls(measured: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per (model, domain, arm): diversity, near-duplicates, budget, novelty counts."""

    cells: dict[tuple, list] = defaultdict(list)
    for row in measured:
        cells[cell_key(row)].append(row)
    out = []
    for (model, domain, arm, lambda_fields, lambda_coupling), rows in sorted(cells.items(), key=str):
        out.append({
            "model": model, "domain": domain, "arm": arm, "n": len(rows),
            "lambda_fields": lambda_fields, "lambda_coupling": lambda_coupling,
            "attempt_budget": rows[0]["attempt_budget"], "lambda": rows[0]["lambda"],
            **(diversity([r["sequence"] for r in rows]) if len(rows) > 1 else {}),
            "n_above_novelty_ceiling": sum(r["max_msa_identity"] > NOVELTY_CEILING for r in rows),
            "novelty_ceiling_percent": NOVELTY_CEILING,
            "mean_step_entropy": None if rows[0]["mean_step_entropy"] is None
            else float(np.mean([r["mean_step_entropy"] for r in rows])),
        })
    return out
