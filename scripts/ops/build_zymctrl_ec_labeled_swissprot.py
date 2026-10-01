#!/usr/bin/env python3
"""Build a reviewed-enzyme EC-labeled FASTA for ZymCTRL-style CLT training.

This is a fallback replacement for the missing original `ec_labeled.fasta`.
It parses Swiss-Prot XML, extracts full EC numbers, and writes records in
the sequence format ZymCTRL was trained on:

    <EC><sep><start><AA_SEQUENCE><end>

If a protein entry has multiple complete EC numbers, one FASTA record is
emitted per EC label.

**The XML reader is no longer this script's.** It imports
`iter_swissprot_entries` from `src.capability.core.swissprot`, which is now this
repository's one Swiss-Prot parser, and this script consumes it. Appendix B rule
12 -- a single declaration, imported, never reimplemented -- reaches *which
records exist* as much as it reaches what a model is fed, and a second parser is
a second answer to that question. What this script keeps is what is its own: the
EC-per-record fan-out, the length cut, and ZymCTRL's rendering.

Everything the old reader had earned is preserved in the shared one and is worth
naming, because both defences were paid for: the namespace is read from the
document's own root rather than hard-coded (UniProt has shipped both the `http`
and the `https` spelling, and against the wrong literal every entry is skipped
and an empty FASTA is written with exit 0), and the entry element is cleared
after each record so a 933 MB stream does not become a resident tree.

The refactor was checked rather than argued: run over an 8,000-entry prefix of
the release, the reader change reproduces this script's output byte for byte --
3,957 records, sha256 `77d9beff7dd7d688...`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.swissprot import (  # noqa: E402
    FULL_EC_PATTERN,
    iter_swissprot_entries,
)

#: ZymCTRL's training rendering. One record per (accession, EC) pair.
RECORD_TEMPLATE = "{ec}<sep><start>{sequence}<end>"
HEADER_TEMPLATE = ">{accession}|{ec}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xml",
        default=str(REPO_ROOT / "data/swissprot/uniprot_sprot.xml.gz"),
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "data/zymctrl/ec_labeled_swissprot.fasta"),
    )
    ap.add_argument("--max-seq-len", type=int, default=1022)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Written beside the destination and renamed only once the record count has
    # been checked. Streaming straight into `out_path` left an interrupted run
    # holding a partial FASTA that is indistinguishable from a complete one, and
    # a later staging command could then treat it as complete.
    temporary = out_path.with_name(f".{out_path.name}.partial")
    n_entries = 0
    n_records = 0
    try:
        with temporary.open("w") as out_f:
            for entry in iter_swissprot_entries(Path(args.xml)):
                # The shared iterator yields every entry; the EC requirement is
                # this script's own, and so is the length cut.
                if not entry.ec or len(entry.sequence) > args.max_seq_len:
                    continue
                n_entries += 1
                for ec in entry.ec:
                    if not FULL_EC_PATTERN.fullmatch(ec):
                        raise RuntimeError(
                            f"{entry.accession}: the shared iterator returned "
                            f"{ec!r}, which is not a four-field EC number; this "
                            "script renders one label per EC and a partial number "
                            "names a subclass rather than an activity"
                        )
                    record = RECORD_TEMPLATE.format(ec=ec, sequence=entry.sequence)
                    out_f.write(
                        HEADER_TEMPLATE.format(accession=entry.accession, ec=ec)
                        + f"\n{record}\n"
                    )
                    n_records += 1
                    if n_records % 50000 == 0:
                        print(f"Wrote {n_records:,} EC-labeled sequences")
        if n_records == 0:
            raise RuntimeError(
                f"{args.xml} yielded no EC-labelled records. An empty corpus is not a "
                "result: either the XML holds no reviewed enzymes with a complete "
                "four-field EC number, or its element names did not resolve. Nothing "
                "has been written to "
                f"{out_path}."
            )
        temporary.replace(out_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Done. Entries with EC: {n_entries:,}")
    print(f"Done. FASTA records: {n_records:,}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
