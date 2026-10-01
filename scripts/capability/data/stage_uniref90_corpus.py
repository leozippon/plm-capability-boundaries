#!/usr/bin/env python3
"""Stage the UniRef90 release FASTA and its DIAMOND index, bound to a published digest.

The pairwise sequence baseline's homolog support is bounded by UniRef50, which
collapses at 50% identity: raising the search's depth ceiling cannot recover
members the clustering removed, so a deeper corpus is the only way to raise the
depth. This stage retrieves one release file, refuses it unless the release's
own metadata declares a digest that the staged bytes reproduce, builds a DIAMOND
index from it, and writes a manifest carrying the release version, the retrieval
date, the exact endpoints and every file digest.

It draws no corpus, reads no measurement, and makes no random choice. The
existing search entry point then runs against the index it writes, unchanged.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.core.corpus_staging import (
    file_digests, parse_release_metalink, require_free_space, verify_staged_file)

UNIREF90_BASE = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90"
FASTA_NAME = "uniref90.fasta.gz"
#: Measured on the retained UniRef50 pair rather than derived: 25,345,024,406
#: index bytes against the 8,780,552,383-byte release payload the same index was
#: built from. Stated against the compressed payload because that is the only
#: size known before the transfer; per database letter the same index is 1.47
#: bytes, and gzipped UniRef50 is 0.508 bytes per letter, which is where the
#: ratio comes from. Used only to size the free-space guard, so an
#: under-estimate weakens the guard and an over-estimate merely reserves more:
#: the first run of this stage used 1.6 by conflating letters with compressed
#: bytes, under-sizing the guard by about 1.8 times, and only the 100 GB reserve
#: and the volume's 466 GB of headroom absorbed it. The UniRef90 pair this
#: stage then produced measures 1.94 rather than 2.89 -- 62,232,904,270 index
#: bytes against 32,107,879,019 payload bytes, because UniRef90's longer
#: headers compress differently from UniRef50's -- so the UniRef50 ratio kept
#: here over-reserves by about 1.5 times, which is the direction a guard on a
#: shared volume should err in.
INDEX_BYTES_PER_PAYLOAD_BYTE = 25345024406 / 8780552383
#: Extra free space a caller may ask this stage to leave behind. Zero by
#: default: the guard's own purpose is to refuse a transfer or an index build
#: that cannot complete, and reserving beyond that is a policy for a shared
#: volume rather than a property of this stage. The first run of this stage set
#: it to 100 GB while that policy was in force.
DEFAULT_RESERVE_GB = 0.0


def fetch_text(url: str, *, timeout: float = 120.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def download(url: str, destination: Path) -> list[str]:
    """Resume-capable transfer. Returns the command, for the manifest."""

    command = ["curl", "--fail", "--location", "--silent", "--show-error",
               "--continue-at", "-", "--output", str(destination), url]
    subprocess.run(command, check=True)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--diamond", type=Path, required=True)
    parser.add_argument("--base-url", default=UNIREF90_BASE)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--reserve-gb", type=float, default=DEFAULT_RESERVE_GB,
                        help="free space to leave beyond the payload and its index")
    parser.add_argument("--skip-index", action="store_true",
                        help="stage and verify the payload only")
    args = parser.parse_args()

    metalink_url = f"{args.base_url}/RELEASE.metalink"
    release_note_url = f"{args.base_url}/uniref90.release_note"
    fasta_url = f"{args.base_url}/{FASTA_NAME}"
    manifest_path = args.dest_dir / "uniref90_staging_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    retrieved = datetime.now(timezone.utc).isoformat()
    declared = parse_release_metalink(fetch_text(metalink_url), FASTA_NAME,
                                      metadata_url=metalink_url)
    release_note = fetch_text(release_note_url).strip()
    reserve = int(args.reserve_gb * 1e9)
    index_estimate = int(declared.size_bytes * INDEX_BYTES_PER_PAYLOAD_BYTE)
    before = require_free_space(args.dest_dir, declared.size_bytes + index_estimate,
                                reserve_bytes=reserve)
    print(f"free before {before / 1e9:.1f} GB; payload {declared.size_bytes / 1e9:.1f} GB; "
          f"index estimate {index_estimate / 1e9:.1f} GB", flush=True)

    payload = args.dest_dir / FASTA_NAME
    command = download(fasta_url, payload)
    staged = verify_staged_file(payload, declared)
    print(f"payload verified: {staged['digests']}", flush=True)

    manifest: dict[str, object] = {
        "schema": "uniref90_corpus_staging_v1",
        "retrieved_utc": retrieved,
        "release_note_url": release_note_url,
        "release_note": release_note,
        "payload": {"url": fasta_url, "command": command, **staged},
        "free_bytes_before": before,
    }

    if not args.skip_index:
        after_payload = require_free_space(args.dest_dir, index_estimate,
                                          reserve_bytes=reserve)
        index = args.dest_dir / "uniref90_full"
        build = [str(args.diamond), "makedb", "--in", str(payload), "-d", str(index),
                 "--threads", str(args.threads)]
        completed = subprocess.run(build, capture_output=True, text=True, check=True)
        built = index.with_suffix(".dmnd")
        info = subprocess.run([str(args.diamond), "dbinfo", "--db", str(built)],
                              capture_output=True, text=True, check=True).stdout.strip()
        sequences = letters = None
        for line in info.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "Sequences":
                sequences = int(parts[1])
            elif len(parts) == 2 and parts[0] == "Letters":
                letters = int(parts[1])
        if sequences is None or letters is None:
            raise RuntimeError("cannot read Sequences/Letters from `diamond dbinfo`")
        manifest["index"] = {
            "path": str(built), "bytes": built.stat().st_size,
            "sha256": file_digests(built, ("sha256",))["sha256"],
            "indexed_sequences": sequences, "indexed_letters": letters,
            "dbinfo": info, "makedb_command": build,
            "makedb_log_tail": completed.stderr.strip().splitlines()[-4:],
            "free_bytes_before_index": after_payload,
        }
        manifest["diamond"] = {
            "executable": str(args.diamond),
            "version": subprocess.run([str(args.diamond), "--version"], check=True,
                                      capture_output=True, text=True).stdout.strip(),
            "sha256": file_digests(args.diamond, ("sha256",))["sha256"]}

    manifest["free_bytes_after"] = require_free_space(args.dest_dir, 0, reserve_bytes=0)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "release_note"}, indent=1)[:2000])


if __name__ == "__main__":
    main()
