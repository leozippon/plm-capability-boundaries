"""Shared sequence metadata utilities required by capability measurements."""
from __future__ import annotations

from .arms import env_path, REPO

PROTEINGYM_ROOT = env_path(
    "TRANSFER_PROTEINGYM_DIR", REPO / "data/proteingym/DMS_ProteinGym_substitutions"
)


def swissprot_accession(header: str) -> str:
    fields = header.split("|")
    if len(fields) < 3 or fields[0] != "sp":
        raise ValueError(f"unexpected Swiss-Prot header {header!r}")
    return fields[1]


from pathlib import Path
from .arms import env_path, require_input_path, REPO

PFAM_RESIDUE_TSV = env_path(
    "TRANSFER_PFAM_RESIDUE_TSV", REPO / "data/interpro/pfam_residue.tsv"
)


PFAM_TSV_HEADER = ("uniprot", "start", "end", "pfam_id")


def load_pfam_spans(
    path: Path = PFAM_RESIDUE_TSV, accessions: set[str] | None = None
) -> dict[str, list[tuple[int, int, str]]]:
    """Residue-level Pfam spans, optionally restricted to given accessions."""

    spans: dict[str, list[tuple[int, int, str]]] = {}
    require_input_path(Path(path), "TRANSFER_PFAM_RESIDUE_TSV")
    with Path(path).open(encoding="utf-8") as handle:
        header = tuple(next(handle).rstrip("\n").split("\t"))
        if header != PFAM_TSV_HEADER:
            raise ValueError(f"{path}: expected columns {PFAM_TSV_HEADER}, found {header}")
        for number, line in enumerate(handle, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"{path}:{number}: expected 4 columns, found {len(fields)}")
            accession, start, end, pfam = fields
            if accessions is not None and accession not in accessions:
                continue
            begin, finish = int(start), int(end)
            if begin < 1 or finish < begin:
                raise ValueError(f"{path}:{number}: invalid span {begin}-{finish}")
            spans.setdefault(accession, []).append((begin, finish, pfam))
    if not spans:
        raise RuntimeError(f"{path}: no Pfam spans matched the requested accessions")
    return spans
