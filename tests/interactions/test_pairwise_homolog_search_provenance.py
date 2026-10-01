"""The homolog search's manifest must state retention as a measured fact.

``source_fasta_retained`` was a hardcoded ``false``. A manifest that asserts a
provenance fact it never checked is worse than one that omits it, and it became
actively wrong the moment an index was built from a FASTA this host keeps. These
tests fix the field to reality in both states and require a declared-but-absent
source to fail rather than be reported as not retained.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capability" / "interactions" / "search_pairwise_homologs.py"
#: Two short sequences and one query that aligns to both, so a real search
#: returns real hits without staging a corpus.
CORPUS = (">s1\nMKVLAAGIVGLNLGGKQYEVMTRSDAEWRQLAEK\n"
          ">s2\nMKVLAAGIVGLNLGGKQYEVMTRSDAEWRQLAER\n")
QUERY_SEQUENCE = "MKVLAAGIVGLNLGGKQYEVMTRSDAEWRQLAEK"


def diamond_executable() -> str | None:
    return os.environ.get("DIAMOND_EXECUTABLE") or shutil.which("diamond")


def run_search(directory: Path, *extra: str) -> subprocess.CompletedProcess:
    catalogue = directory / "catalogue.json"
    catalogue.write_text(json.dumps([{"WT_name": "q1", "sequence": QUERY_SEQUENCE}]))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--diamond", str(diamond_executable()),
         "--database", str(directory / "corpus.dmnd"), "--catalogue", str(catalogue),
         "--out-dir", str(directory / "out"), "--threads", "2", *extra],
        capture_output=True, text=True, cwd=str(ROOT))


class DeclaredButAbsentSource(unittest.TestCase):
    def test_a_declared_source_fasta_that_is_not_on_disk_fails_the_run(self):
        """No index, no binary and no corpus are needed to reach this refusal.

        The check runs before anything is read, which is the point: a provenance
        declaration that cannot be verified stops the search instead of being
        written into the manifest as its opposite.
        """

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            catalogue = directory / "catalogue.json"
            catalogue.write_text(json.dumps([{"WT_name": "q1", "sequence": QUERY_SEQUENCE}]))
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--diamond", str(directory / "absent-diamond"),
                 "--database", str(directory / "absent.dmnd"), "--catalogue", str(catalogue),
                 "--out-dir", str(directory / "out"),
                 "--source-fasta", str(directory / "never-downloaded.fasta.gz")],
                capture_output=True, text=True, cwd=str(ROOT))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("retained source FASTA", completed.stderr)
        self.assertIn("FileNotFoundError", completed.stderr)


@unittest.skipIf(diamond_executable() is None,
                 "no diamond executable on PATH or in DIAMOND_EXECUTABLE")
class MeasuredRetention(unittest.TestCase):
    """Both states, end to end through the real search and the real manifest."""

    def index(self, directory: Path) -> Path:
        fasta = directory / "corpus.fasta"
        fasta.write_text(CORPUS)
        subprocess.run([str(diamond_executable()), "makedb", "--in", str(fasta),
                        "-d", str(directory / "corpus"), "--threads", "2"],
                       check=True, capture_output=True, text=True)
        return fasta

    def manifest(self, directory: Path) -> dict:
        return json.loads((directory / "out" / "search_manifest.json").read_text())

    def test_an_unretained_source_is_reported_as_unretained(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fasta = self.index(directory)
            fasta.unlink()
            completed = run_search(directory)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            database = self.manifest(directory)["database"]
        self.assertIs(database["source_fasta_retained"], False)
        self.assertIsNone(database["source_fasta"])
        self.assertIsNone(database["source_fasta_bytes"])

    def test_a_retained_source_is_reported_with_its_measured_length(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            fasta = self.index(directory)
            completed = run_search(directory, "--source-fasta", str(fasta))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            database = self.manifest(directory)["database"]
            expected = fasta.stat().st_size
        self.assertIs(database["source_fasta_retained"], True)
        self.assertEqual(database["source_fasta"], str(fasta))
        self.assertEqual(database["source_fasta_bytes"], expected)
        self.assertGreater(expected, 0)


if __name__ == "__main__":
    unittest.main()
