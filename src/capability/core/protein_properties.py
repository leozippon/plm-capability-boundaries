"""Shared protein properties utilities required by capability measurements."""
from __future__ import annotations

from .arms import AA20
from ..context.profiles import KYTE_DOOLITTLE

CHARGE_PH7: dict[str, float] = {residue: 0.0 for residue in AA20} | {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}


SIDE_CHAIN_VOLUME: dict[str, float] = {
    "A": 67.0, "R": 148.0, "N": 96.0, "D": 91.0, "C": 86.0,
    "Q": 114.0, "E": 109.0, "G": 48.0, "H": 118.0, "I": 124.0,
    "L": 124.0, "K": 135.0, "M": 124.0, "F": 135.0, "P": 90.0,
    "S": 73.0, "T": 93.0, "W": 163.0, "Y": 141.0, "V": 105.0,
}


PROPERTY_BASIS: dict[str, dict[str, float]] = {
    "hydropathy": dict(KYTE_DOOLITTLE),
    "charge": dict(CHARGE_PH7),
    "volume": dict(SIDE_CHAIN_VOLUME),
}




GRANTHAM_POLARITY: dict[str, float] = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2,
    "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0,
    "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}




AA_CLASSES: dict[str, str] = {
    "charged": "DEKRH",
    "hydrophobic": "AVLIMFW",
    "polar": "STNQY",
    "special": "CGP",
}


CLASS_NAMES: tuple[str, ...] = tuple(sorted(AA_CLASSES))


CLASS_OF_RESIDUE: dict[str, str] = {
    residue: name for name, residues in AA_CLASSES.items() for residue in residues
}
