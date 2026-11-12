#!/usr/bin/env python3
"""Refuse a candidate deletion set that contains a path documented as held.

A pattern-scoped sweep enumerates its candidates from the filesystem, so every
check it can make is a check about the files themselves and an individually
enumerated hold cannot stop it. That has now destroyed the last copy of
something twice: `results/audit_20261011_h200/PROVENANCE_LOSS_20261110.txt` and
`logs/maintenance/PROVENANCE_LOSS_20261111.txt`. This check is the step that
reads the hold documents on the sweep's behalf, and a sweep that does not run it
has not consulted them.

A path counts as held when a hold document names it between backticks, either
under a top-level section whose heading speaks of holding or on a line that
does. Naming is exact: the check asks whether a candidate is itself named, not
whether it sits under a named directory, because these documents name
directories in passing and a containment rule reads those as holds over whole
trees. A sweep inside a held tree is therefore not caught, and the hold on a
tree has to be read by a person. Within that scope the rule leans toward
refusal: a path named beside the word "held" for any reason is refused, because
a refusal costs one reading of the cited line while the other mistake costs a
file nothing can rebuild. The check fails closed -- a hold document that is
missing, unreadable, or yields no path at all refuses the whole set rather than
passing it.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

HOLD_DOCUMENTS = ("docs/REPO_DELETION_ALLOWLIST.md", "docs/HANDOFF_20261110.md")
HOLD_WORDS = re.compile(r"\bhold\b|\bholds\b|\bheld\b|\bholding\b|not removed|not touched|excluded", re.IGNORECASE)
QUOTED = re.compile(r"`([^`]+)`")
PATH_TOKEN = re.compile(r"[A-Za-z0-9._\-/]+")
REFUSED = "REFUSED"


class Hold(NamedTuple):
    path: str
    document: str
    line: int


def hold_token(token: str) -> str | None:
    """The repository-relative path a backquoted token names, or None."""
    text = token.strip().rstrip(".,;:)")
    while text.startswith("./"):
        text = text[2:]
    text = text.rstrip("/")
    if "/" not in text or not PATH_TOKEN.fullmatch(text):
        return None
    return text


def read_holds(root: Path, documents: Sequence[str] = HOLD_DOCUMENTS) -> dict[str, Hold]:
    """Every path the hold documents name as held, or refuse if they cannot be read."""
    holds: dict[str, Hold] = {}
    for name in documents:
        try:
            lines = (root / name).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise SystemExit(f"{REFUSED}: cannot read the hold document {name}: {error}")
        found = 0
        section = ""
        for number, line in enumerate(lines, 1):
            if line.startswith("## "):
                section = line
            if not (HOLD_WORDS.search(section) or HOLD_WORDS.search(line)):
                continue
            for held in filter(None, map(hold_token, QUOTED.findall(line))):
                found += 1
                holds.setdefault(held, Hold(held, name, number))
        if not found:
            raise SystemExit(
                f"{REFUSED}: no held path could be parsed from {name}, so the holds it "
                "states cannot be checked; the document's format has changed"
            )
    return holds


def candidate_path(candidate: str | os.PathLike[str], root: Path) -> str:
    """A candidate as a repository-relative POSIX path, or refuse."""
    path = Path(os.path.normpath(candidate))
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            raise SystemExit(
                f"{REFUSED}: {candidate} lies outside {root}, so the hold documents "
                "cannot speak for it"
            )
    text = path.as_posix()
    if text in {".", "", "/"} or text == ".." or text.startswith("../"):
        raise SystemExit(f"{REFUSED}: {candidate!r} is not a path inside {root}")
    return text


def refuse_held_paths(
    candidates: Sequence[str | os.PathLike[str]],
    root: str | os.PathLike[str],
    documents: Sequence[str] = HOLD_DOCUMENTS,
) -> int:
    """Pass the deletion set and return its size, or refuse it loudly and entirely."""
    base = Path(root).resolve()
    holds = read_holds(base, documents)
    checked = [candidate_path(candidate, base) for candidate in candidates]
    if not checked:
        raise SystemExit(f"{REFUSED}: no candidate paths were given, so nothing was checked")
    refused = [(candidate, holds[candidate]) for candidate in checked if candidate in holds]
    if refused:
        report = [
            f"{REFUSED}: {len(refused)} of {len(checked)} candidate paths are documented "
            "as held and must not be deleted:"
        ]
        for candidate, hold in refused:
            report.append(f"  {candidate} -- {hold.document}:{hold.line}")
        report.append(
            "Read each line cited above. Remove such a path only under an authorisation "
            "that names it, and record the decision beside the hold."
        )
        raise SystemExit("\n".join(report))
    return len(checked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refuse a deletion set that contains a documented hold.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="candidate paths, repository-relative or absolute; a single '-' reads them from stdin, one per line",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.paths
    if paths == ["-"]:
        paths = [line.strip() for line in sys.stdin if line.strip()]
    checked = refuse_held_paths(paths, args.root)
    print(f"{checked} candidate paths checked against {len(HOLD_DOCUMENTS)} hold documents; none is held")


if __name__ == "__main__":
    main()
