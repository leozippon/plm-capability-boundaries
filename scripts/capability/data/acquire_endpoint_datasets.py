#!/usr/bin/env python3
"""Fetch the three measured endpoints the capability map was missing.

Each resource is pinned here by URL and, where the host publishes one, by the
upstream checksum, and the fetch is refused when what arrives does not match.
Refusal is the point: an external resource that cannot be bound to a source
digest is not usable evidence, so a silent partial download must fail rather
than become a dataset directory.

  mgnify_stability_cho2026   MGnify-derived domain stabilities with 95% CIs, the
                             first support that can reach the remote-homology
                             band the retrieval gate cannot currently test.
  domainome_beltran2025      500 human domains by abundance PCA, the external
                             confirmation endpoint, from neither ProteinGym's
                             panel nor Tsuboyama 2023.
  skempi2                    Measured protein-protein binding affinities on
                             solved complexes, a candidate function endpoint.
  randomized_cores_escobedo2025
                             Randomized-core and -surface combinatorial
                             libraries, acquired to decide whether a third-order
                             endpoint exists that beats the floors the higher-
                             order gate has measured so far.

Downloads resume, so an interrupted run is re-runnable; a file already present
at its pinned digest is left alone.
"""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.core.io import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]

#: dataset -> (relative path, url, upstream checksum or None, bytes or None).
#: A ``None`` checksum records that the host publishes none, not that none was
#: checked: the measured sha256 is written to the registry either way.
RESOURCES: dict[str, tuple[tuple[str, str, str | None, int | None], ...]] = {
    "mgnify_stability_cho2026": (
        (
            "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv",
            "https://zenodo.org/api/records/19411306/files/"
            "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv/content",
            "md5:371a3ebbfdb5c3ca1f0f2af8b2b8d3a8",
            2233150308,
        ),
        (
            "README.md",
            "https://zenodo.org/api/records/19411306/files/README.md/content",
            None,
            None,
        ),
    ),
    "domainome_beltran2025": (
        (
            "Supplementary_Table_2_fitness_scores_normalized_domainranks.txt.zip",
            "https://zenodo.org/api/records/14356805/files/"
            "Supplementary_Table_2_fitness_scores_normalized_domainranks.txt.zip/content",
            "md5:d0e744d6e50be563e46254ecc8586f24",
            29295816,
        ),
        (
            "Supplementary_Table_1_library_design.txt",
            "https://zenodo.org/api/records/14356805/files/"
            "Supplementary_Table_1_library_design.txt/content",
            "md5:9efcf1ae4f90e975f0c0719c79d136a6",
            792,
        ),
    ),
    "randomized_cores_escobedo2025": (
        (
            "Escobedo_et_al_protein_randomization_data_files.zip",
            "https://zenodo.org/api/records/16266645/files/"
            "Escobedo_et_al_protein_randomization_data_files.zip/content",
            "md5:e7d6433bf7cca53071836a097ca63069",
            2226162297,
        ),
        (
            "combinatorialcores_github_repo.zip",
            "https://zenodo.org/api/records/16266645/files/"
            "combinatorialcores_github_repo.zip/content",
            "md5:6f0626ae43afba3b03087593197dc3a3",
            19726227,
        ),
    ),
    "skempi2": (
        (
            "skempi_v2.csv",
            "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv",
            None,
            None,
        ),
    ),
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fSL", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", str(destination), url],
        check=True,
    )


def acquire(dataset: str, data_root: Path) -> list[tuple[str, int, str]]:
    written: list[tuple[str, int, str]] = []
    for relpath, url, checksum, expected_bytes in RESOURCES[dataset]:
        destination = data_root / dataset / relpath
        started = time.time()
        fetch(url, destination)
        size = destination.stat().st_size
        if expected_bytes is not None and size != expected_bytes:
            raise SystemExit(
                f"{dataset}/{relpath} is {size} bytes, upstream declares {expected_bytes}"
            )
        if checksum is not None:
            algorithm, _, expected = checksum.partition(":")
            if algorithm != "md5":
                raise SystemExit(f"{dataset}/{relpath} pins an unsupported checksum {checksum}")
            observed = md5_file(destination)
            if observed != expected:
                raise SystemExit(
                    f"{dataset}/{relpath} has md5 {observed}, upstream declares {expected}"
                )
        digest = sha256_file(destination)
        elapsed = time.time() - started
        written.append((relpath, size, digest))
        upstream = checksum if checksum is not None else "no upstream checksum published"
        print(
            f"{dataset}/{relpath}: {size} bytes in {elapsed:.1f} s, "
            f"sha256 {digest}, {upstream}"
        )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        choices=sorted(RESOURCES),
        help="acquire only these; repeatable, default every one",
    )
    args = parser.parse_args(argv)
    for dataset in args.dataset or sorted(RESOURCES):
        acquire(dataset, args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
