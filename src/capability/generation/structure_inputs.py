"""Shared structure inputs support for main measurements."""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..core.arms import env_path, require_input_path, REPO

ALPHAFOLD_ROOT = env_path("TRANSFER_ALPHAFOLD_DIR", REPO / "data/alphafold")


THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


@dataclass(frozen=True)
class Structure:
    """CA trace and per-residue confidence from one AlphaFold model."""

    accession: str
    sequence: str
    ca: np.ndarray
    plddt: np.ndarray
    n_non_canonical_residues: int

    def __post_init__(self) -> None:
        if self.ca.ndim != 2 or self.ca.shape[1] != 3:
            raise ValueError(f"{self.accession}: CA coordinates must have shape (n, 3)")
        if self.plddt.ndim != 1 or self.plddt.shape[0] != self.ca.shape[0]:
            raise ValueError(f"{self.accession}: pLDDT does not align with the CA trace")
        if len(self.sequence) != self.ca.shape[0]:
            raise ValueError(f"{self.accession}: sequence does not align with the CA trace")
        if self.ca.shape[0] == 0:
            raise ValueError(f"{self.accession}: no CA atoms")
        if not np.isfinite(self.ca).all() or not np.isfinite(self.plddt).all():
            raise ValueError(f"{self.accession}: non-finite coordinates or pLDDT")

    def __len__(self) -> int:
        return len(self.sequence)


def accession_from_alphafold_path(path: Path) -> str:
    parts = Path(path).name.split("-")
    if len(parts) < 3 or parts[0] != "AF":
        raise ValueError(f"{path}: not an AlphaFold model filename")
    return parts[1]


def read_alphafold_model(path: Path) -> Structure:
    """Parse CA coordinates, pLDDT (B-factor column) and the one-letter sequence.

    Residues outside the canonical twenty are counted rather than tolerated
    silently, so callers can exclude such models by an explicit predicate.
    """

    coordinates: list[tuple[float, float, float]] = []
    confidence: list[float] = []
    residues: list[str] = []
    skipped = 0
    previous = None
    with gzip.open(Path(path), "rt") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            number = int(line[22:26])
            if previous is not None and number <= previous:
                raise ValueError(f"{path}: CA residue numbers are not strictly increasing")
            previous = number
            name = line[17:20].strip()
            if name not in THREE_TO_ONE:
                skipped += 1
                continue
            coordinates.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
            confidence.append(float(line[60:66]))
            residues.append(THREE_TO_ONE[name])
    if not coordinates:
        raise ValueError(f"{path}: no canonical CA atoms")
    return Structure(
        accession=accession_from_alphafold_path(path),
        sequence="".join(residues),
        ca=np.asarray(coordinates, dtype=np.float64),
        plddt=np.asarray(confidence, dtype=np.float64),
        n_non_canonical_residues=skipped,
    )


def ca_secondary_structure(ca: np.ndarray) -> np.ndarray:
    """Three-state assignment from the CA trace alone (0 helix, 1 strand, 2 coil).

    P-SEA distance criterion. This is a coordinate-only approximation to DSSP
    that ignores hydrogen bonding; it is used because it applies uniformly to
    every AlphaFold model without an external dependency, and its absolute
    fractions must not be quoted as DSSP secondary structure content.
    """

    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError("CA coordinates must have shape (n, 3)")
    n = ca.shape[0]
    assignment = np.full(n, 2, dtype=np.int8)
    if n < 5:
        return assignment
    d2 = np.linalg.norm(ca[2:] - ca[:-2], axis=1)
    d3 = np.linalg.norm(ca[3:] - ca[:-3], axis=1)
    d4 = np.linalg.norm(ca[4:] - ca[:-4], axis=1)
    for start in range(n - 4):
        a, b, c = d2[start], d3[start], d4[start]
        if abs(a - 5.5) < 0.5 and abs(b - 5.3) < 0.5 and abs(c - 6.4) < 0.6:
            assignment[start : start + 5] = 0
        elif abs(a - 6.7) < 0.6 and abs(b - 9.9) < 0.9 and abs(c - 12.4) < 1.1:
            assignment[start : start + 5] = 1
    return assignment


def alphafold_models(root: Path = ALPHAFOLD_ROOT, *, limit: int | None = None) -> list[Path]:
    """AlphaFold PDB models in deterministic filename order.

    Filename order is UniProt-accession order, which front-loads whole-proteome
    dumps of closely related entries, so ``limit`` returns a taxonomically
    clustered prefix rather than a sample. It is kept because the full catalogue
    is what most callers want and because a frozen artefact was produced with it;
    :func:`alphafold_model_sample` is what a *limited* selection should use.
    """

    require_input_path(Path(root), "TRANSFER_ALPHAFOLD_DIR")
    paths = sorted(Path(root).glob("AF-*-model_v*.pdb.gz"))
    if not paths:
        raise RuntimeError(f"no AlphaFold PDB models under {root}")
    return paths if limit is None else paths[:limit]
