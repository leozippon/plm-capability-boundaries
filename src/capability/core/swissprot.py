"""Shared swissprot support for main measurements."""
from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
from xml.etree.ElementTree import iterparse
from .arms import require_input_path, env_path, REPO

SWISSPROT_XML = env_path(
    "TRANSFER_SWISSPROT_XML", REPO / "data/swissprot/uniprot_sprot.xml.gz"
)


FULL_EC_PATTERN = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


class GoAnnotation(NamedTuple):
    """One ``<dbReference type="GO">`` of an entry, as the XML carries it."""

    go_id: str
    #: The curated term name with UniProt's aspect prefix stripped: the XML
    #: writes ``F:kinase activity`` and the aspect is kept separately so that a
    #: surface form is never masked with a stray ``F:`` attached to it.
    term: str
    #: ``F``, ``P`` or ``C``.
    aspect: str
    evidence: str


@dataclass(frozen=True)
class SwissProtEntry:
    """One Swiss-Prot entry, reduced to what a description cohort reads."""

    accession: str
    entry_name: str
    protein_name: str
    function_texts: tuple[str, ...]
    sequence: str
    ec: tuple[str, ...]
    go: tuple[GoAnnotation, ...]
    interpro: tuple[tuple[str, str], ...]
    pfam: tuple[tuple[str, str], ...]


def iter_swissprot_entries(path: Path = SWISSPROT_XML) -> Iterator[SwissProtEntry]:
    """Stream Swiss-Prot entries out of the release XML.

    The namespace is read from the document's own root rather than hard-coded.
    That check is inherited from ``scripts/ops/build_zymctrl_ec_labeled_swissprot.py``,
    which earned it: UniProt has shipped both ``http://uniprot.org/uniprot`` and
    ``https://uniprot.org/uniprot`` as the default namespace, and against the
    wrong literal every tag comparison fails, every entry is skipped, and the
    consumer writes an empty output and exits 0.

    EC numbers come from **both** places the release puts them -- the
    ``<ecNumber>`` elements inside ``<protein>`` and the ``<dbReference
    type="EC">`` cross-references -- deduplicated and sorted, because an entry
    can carry one and not the other.

    An entry without an accession or without a sequence is skipped, exactly as
    the parser this replaces skipped it: those two fields are the identity of the
    record, and there is nothing to yield without them. Everything else is
    yielded as found, including an empty ``protein_name`` or an empty ``ec``, so
    that eligibility is decided by the caller and is visible in the caller's own
    rejection counts.
    """

    path = Path(path)
    require_input_path(path, "TRANSFER_SWISSPROT_XML")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        namespace: str | None = None
        for event, payload in iterparse(handle, events=("start-ns", "end")):
            if event == "start-ns":
                prefix, uri = payload
                if prefix == "":
                    namespace = f"{{{uri}}}"
                continue
            element = payload
            if namespace is None:
                raise RuntimeError(
                    f"{path} declares no default XML namespace; this parser "
                    "resolves UniProt element names through it"
                )
            if element.tag != f"{namespace}entry":
                continue
            entry = _entry_from_element(element, namespace)
            if entry is not None:
                yield entry
            element.clear()


def _entry_from_element(element: Any, namespace: str) -> SwissProtEntry | None:
    accession_element = element.find(f"{namespace}accession")
    sequence_element = element.find(f"{namespace}sequence")
    if accession_element is None or sequence_element is None:
        return None
    if sequence_element.text is None:
        return None
    accession = (accession_element.text or "").strip()
    sequence = "".join(sequence_element.text.split())
    if not accession or not sequence:
        return None

    name_element = element.find(f"{namespace}name")
    entry_name = (name_element.text or "").strip() if name_element is not None else ""

    protein_name = ""
    protein = element.find(f"{namespace}protein")
    if protein is not None:
        recommended = protein.find(f"{namespace}recommendedName")
        if recommended is not None:
            full = recommended.find(f"{namespace}fullName")
            if full is not None and full.text:
                protein_name = " ".join(full.text.split())

    function_texts: list[str] = []
    for comment in element.findall(f"{namespace}comment[@type='function']"):
        for text in comment.findall(f"{namespace}text"):
            if text.text:
                collapsed = " ".join(text.text.split())
                if collapsed:
                    function_texts.append(collapsed)

    ec: set[str] = set()
    for number in element.findall(f".//{namespace}ecNumber"):
        if number.text and FULL_EC_PATTERN.fullmatch(number.text.strip()):
            ec.add(number.text.strip())
    for reference in element.findall(f".//{namespace}dbReference[@type='EC']"):
        identifier = reference.attrib.get("id", "").strip()
        if FULL_EC_PATTERN.fullmatch(identifier):
            ec.add(identifier)

    go: list[GoAnnotation] = []
    interpro: list[tuple[str, str]] = []
    pfam: list[tuple[str, str]] = []
    for reference in element.findall(f"{namespace}dbReference"):
        kind = reference.attrib.get("type")
        identifier = reference.attrib.get("id", "").strip()
        if not identifier:
            continue
        if kind == "GO":
            term, aspect, evidence = "", "", ""
            for prop in reference.findall(f"{namespace}property"):
                if prop.attrib.get("type") == "term":
                    value = prop.attrib.get("value", "")
                    aspect, _, term = value.partition(":")
                    term = term.strip()
                    aspect = aspect.strip()
                elif prop.attrib.get("type") == "evidence":
                    evidence = prop.attrib.get("value", "").strip()
            go.append(GoAnnotation(identifier, term, aspect, evidence))
        elif kind in ("InterPro", "Pfam"):
            entry_names = [
                prop.attrib.get("value", "").strip()
                for prop in reference.findall(f"{namespace}property")
                if prop.attrib.get("type") == "entry name"
            ]
            pair = (identifier, entry_names[0] if entry_names else "")
            (interpro if kind == "InterPro" else pfam).append(pair)

    return SwissProtEntry(
        accession=accession,
        entry_name=entry_name,
        protein_name=protein_name,
        function_texts=tuple(function_texts),
        sequence=sequence,
        ec=tuple(sorted(ec)),
        go=tuple(go),
        interpro=tuple(interpro),
        pfam=tuple(pfam),
    )
