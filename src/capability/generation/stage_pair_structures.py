"""Pairwise structural readouts of the frozen stage-pair set, derived from folded coordinates.

The project's structure instrument is ESMFold2
(:mod:`src.capability.generation.structure_evidence`, run through
``run_structure_evidence.py``), which persists, per folded sequence, the selected
diffusion sample's CA pLDDT, the full predicted-aligned-error matrix, pTM and a
PDB of all atom coordinates. Nothing here re-folds and nothing here changes that
instrument.

What it adds is the *pairwise geometry* two later analyses need and a folded PDB
already contains: the CB-CB distance matrix and a contact map at a declared
cutoff. ESM's own contact head is unusable on this checkpoint -- its regression
weights are randomly initialised -- so contacts must come from coordinates, and
re-deriving them later would mean re-folding. They are written once, keyed by the
same sequence identifier the frozen set uses, beside the instrument's own PAE so
that a contact and its predicted reliability travel together.

Confidence is not stability and not function. A high pLDDT says the predictor is
confident, which is an instrument reading on a sequence; no stability predictor
and no measurement is applied to a generated sequence anywhere in E18.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "d1_stage_pair_structures_v1"

#: Contact cutoff on the CB-CB distance, in angstrom. The 8 A CB-CB convention
#: this repository already uses for experimental contacts
#: (:mod:`src.capability.interactions.contact_enrichment`), so a generated
#: contact map and a measured one are the same definition.
CONTACT_CUTOFF_ANGSTROM: float = 8.0

#: Glycine has no CB. The standard substitution is its CA, which is what every
#: contact definition in this repository does, and it is recorded rather than
#: left implicit.
GLYCINE_USES_CA = True

CEILING: dict[str, str] = {
    "confidence_is_not_stability": (
        "pLDDT, pTM and PAE are predicted confidence. No stability predictor and no "
        "stability measurement is applied to a generated sequence"
    ),
    "contacts_are_predicted": (
        "the contact map is derived from predicted coordinates, so it is a prediction "
        "about a generated sequence and not an observed structure"
    ),
    "one_diffusion_sample": (
        "ESMFold2 is a diffusion model and the instrument keeps the highest-pTM sample; "
        "the geometry here describes that one sample"
    ),
    "within_instrument_only": (
        "the stage comparison is within one instrument, one checkpoint and one run. "
        "Confidences are never differenced against a retained ESMFold v1 number"
    ),
}


def parse_backbone(pdb_text: str) -> dict[str, Any]:
    """Residue-ordered CB coordinates (CA for glycine) and B-factors from one chain.

    Only ATOM records of the first chain are read, in residue order. A PDB whose
    residue numbering is not contiguous is accepted -- the index is positional,
    matching the folded sequence -- but a residue with neither CB nor CA is a
    defect and is refused, because a missing coordinate would silently drop a row
    and a column from every distance matrix built on it.
    """

    residues: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    chain: str | None = None
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        this_chain = line[21]
        if chain is None:
            chain = this_chain
        if this_chain != chain:
            continue
        number = int(line[22:26])
        if number not in residues:
            residues[number] = {"name": line[17:20].strip(), "atoms": {}, "b": {}}
            order.append(number)
        if atom in ("CA", "CB"):
            residues[number]["atoms"][atom] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
            residues[number]["b"][atom] = float(line[60:66])
    if not order:
        raise ValueError("the PDB carries no ATOM record")
    coordinates: list[tuple[float, float, float]] = []
    confidences: list[float] = []
    names: list[str] = []
    for number in order:
        record = residues[number]
        atom = "CB" if "CB" in record["atoms"] else "CA"
        if atom not in record["atoms"]:
            raise ValueError(f"residue {number} has neither CB nor CA")
        coordinates.append(record["atoms"][atom])
        confidences.append(record["b"].get(atom, float("nan")))
        names.append(record["name"])
    return {
        "chain": chain,
        "residue_numbers": order,
        "residue_names": names,
        "coordinates": np.asarray(coordinates, dtype=np.float64),
        "atom_b_factors": np.asarray(confidences, dtype=np.float64),
        "n_glycine_using_ca": sum(
            1 for number in order if "CB" not in residues[number]["atoms"]
        ),
    }


def pairwise_geometry(
    pdb_text: str, *, cutoff: float = CONTACT_CUTOFF_ANGSTROM
) -> dict[str, Any]:
    """CB-CB distances and the contact map of one folded structure."""

    if cutoff <= 0:
        raise ValueError("a contact cutoff must be positive")
    backbone = parse_backbone(pdb_text)
    coordinates = backbone["coordinates"]
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt((difference**2).sum(axis=-1))
    contact = distance <= cutoff
    np.fill_diagonal(contact, False)
    off_diagonal = ~np.eye(distance.shape[0], dtype=bool)
    return {
        "n_residues": int(coordinates.shape[0]),
        "cb_distance_angstrom": distance.astype(np.float32),
        "contact_map": contact,
        "contact_cutoff_angstrom": float(cutoff),
        "n_contacts": int(contact.sum() // 2),
        "contact_density": float(contact[off_diagonal].mean()),
        "atom_b_factors": backbone["atom_b_factors"].astype(np.float32),
        "n_glycine_using_ca": int(backbone["n_glycine_using_ca"]),
        "glycine_uses_ca": GLYCINE_USES_CA,
    }


def load_structure(root: Path, sequence_sha256: str) -> dict[str, Any] | None:
    """One folded object of a ``run_structure_evidence.py`` tree, or None if absent."""

    directory = Path(root) / "objects" / sequence_sha256
    result_path = directory / "result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        return {"status": result.get("status"), "directory": str(directory), "result": result}
    pdb = directory / "prediction.pdb"
    arrays = directory / "prediction.npz"
    if not (pdb.is_file() and arrays.is_file()):
        raise FileNotFoundError(
            f"{directory} reports status ok but is missing its prediction files; the "
            "tree is not the one that produced the result record"
        )
    with np.load(arrays) as handle:
        stored = {name: handle[name] for name in handle.files}
    return {
        "status": "ok",
        "directory": str(directory),
        "result": result,
        "pdb_text": pdb.read_text(encoding="utf-8"),
        "arrays": stored,
    }


def summarise_record(
    record: Mapping[str, Any], structure: Mapping[str, Any], *, cutoff: float
) -> dict[str, Any]:
    """One frozen-set record joined to its folded structure and pairwise geometry."""

    geometry = pairwise_geometry(structure["pdb_text"], cutoff=cutoff)
    if geometry["n_residues"] != int(record["length"]):
        raise ValueError(
            f"{record['id']}: the folded structure has {geometry['n_residues']} residues "
            f"and the frozen sequence has {record['length']}"
        )
    arrays = structure["arrays"]
    plddt = np.asarray(arrays["ca_plddt_0_100"], dtype=np.float64)
    pae = np.asarray(arrays["predicted_aligned_error_angstrom"], dtype=np.float64)
    off_diagonal = ~np.eye(pae.shape[0], dtype=bool)
    contact = geometry["contact_map"]
    return {
        "id": record["id"],
        "stage": record["stage"],
        "arm": record["arm"],
        "stream": record["stream"],
        "sequence_sha256": record["sequence_sha256"],
        "length": int(record["length"]),
        "mean_ca_plddt": float(plddt.mean()),
        "fraction_ca_plddt_ge70": float((plddt >= 70.0).mean()),
        "fraction_ca_plddt_ge90": float((plddt >= 90.0).mean()),
        "ptm": float(np.asarray(arrays["ptm"]).reshape(-1)[0]),
        "mean_pae_angstrom": float(pae[off_diagonal].mean()),
        "mean_pae_on_contacts_angstrom": (
            float(pae[contact].mean()) if contact.any() else None
        ),
        "n_contacts": geometry["n_contacts"],
        "contact_density": geometry["contact_density"],
        "contact_cutoff_angstrom": geometry["contact_cutoff_angstrom"],
        "diffusion_sample_index": int(
            np.asarray(arrays["diffusion_sample_index"]).reshape(-1)[0]
        ),
        "properties": dict(record.get("properties") or {}),
    }


def write_pairwise(
    directory: Path, record: Mapping[str, Any], structure: Mapping[str, Any], *, cutoff: float
) -> dict[str, Any]:
    """Persist one sequence's pairwise arrays, keyed by the frozen-set identifier."""

    geometry = pairwise_geometry(structure["pdb_text"], cutoff=cutoff)
    arrays = structure["arrays"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['id']}.npz"
    np.savez_compressed(
        path,
        cb_distance_angstrom=geometry["cb_distance_angstrom"],
        contact_map=geometry["contact_map"],
        predicted_aligned_error_angstrom=np.asarray(
            arrays["predicted_aligned_error_angstrom"], dtype=np.float32
        ),
        ca_plddt_0_100=np.asarray(arrays["ca_plddt_0_100"], dtype=np.float32),
        atom_b_factors=geometry["atom_b_factors"],
        contact_cutoff_angstrom=np.asarray([geometry["contact_cutoff_angstrom"]], dtype=np.float32),
        length=np.asarray([geometry["n_residues"]], dtype=np.int32),
    )
    return {"id": record["id"], "path": str(path), "n_residues": geometry["n_residues"]}


def stage_summary(rows: Sequence[Mapping[str, Any]], *, field: str) -> dict[str, Any]:
    """Per stage and per stream, the mean of one readout."""

    by_stage: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        by_stage.setdefault(str(row["stage"]), {}).setdefault(str(row["stream"]), []).append(
            float(value)
        )
    return {
        stage: {
            "n": sum(len(values) for values in streams.values()),
            "mean": float(np.mean([value for values in streams.values() for value in values])),
            "per_stream_mean": {
                stream: float(np.mean(values)) for stream, values in sorted(streams.items())
            },
        }
        for stage, streams in sorted(by_stage.items())
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} carries no record")
    return rows


def iter_structure_roots(roots: Iterable[Path]) -> list[Path]:
    present = [Path(root) for root in roots if (Path(root) / "objects").is_dir()]
    if not present:
        raise ValueError(
            "none of the supplied structure roots holds an objects/ directory; a "
            "structural summary of an unfolded set is refused rather than reported as "
            "an absence of confidence"
        )
    return present
