"""Conditions that must hold before an external corpus may enter a baseline.

The pairwise sequence baseline's depth is bounded by the corpus it was fitted
on, so deepening it means introducing bytes this repository did not produce. The
condition that must always hold is that those bytes are bound to a digest the
publisher declared: a resource whose release metadata carries no checkable
digest is refused, and staged bytes that do not reproduce the declared digest
are refused. The shared-volume guard is tested on the same footing, because a
transfer that fills the volume damages work outside this experiment.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.capability.core.corpus_staging import (
    PublishedDigest, file_digests, free_bytes, parse_release_metalink,
    require_free_space, verify_staged_file)

#: The shape UniProt publishes, reduced to the three files a test needs. The
#: md5 and size on ``uniref90.fasta.gz`` are the ones Release 2026_03 declares,
#: so a parser change that stops reading the real document fails here.
METALINK = """<?xml version="1.0" encoding="UTF-8"?>
<metalink xmlns="http://www.metalinker.org/" version="3.0">
 <files>
  <file name="uniref90.fasta.gz">
   <size>32107879019</size>
   <verification>
    <hash type="md5">34a83a402720e61b713e95db93d0f956</hash>
   </verification>
   <resources>
    <url type="ftp" location="uk" preference="100">ftp://example.invalid/uniref90.fasta.gz</url>
   </resources>
  </file>
  <file name="undeclared.fasta.gz">
   <size>1234</size>
   <resources>
    <url type="ftp">ftp://example.invalid/undeclared.fasta.gz</url>
   </resources>
  </file>
  <file name="exotic.fasta.gz">
   <size>1234</size>
   <verification>
    <hash type="crc32">deadbeef</hash>
   </verification>
  </file>
  <file name="unsized.fasta.gz">
   <verification>
    <hash type="md5">34a83a402720e61b713e95db93d0f956</hash>
   </verification>
  </file>
 </files>
</metalink>
"""
URL = "https://example.invalid/RELEASE.metalink"


class DeclaredDigest(unittest.TestCase):
    def test_a_declared_file_carries_its_size_digest_and_provenance(self):
        digest = parse_release_metalink(METALINK, "uniref90.fasta.gz", metadata_url=URL)
        self.assertEqual(digest.size_bytes, 32107879019)
        self.assertEqual(digest.algorithm, "md5")
        self.assertEqual(digest.value, "34a83a402720e61b713e95db93d0f956")
        self.assertEqual(digest.metadata_url, URL)
        # The manifest has to be able to say where the digest came from, not
        # only what it was; a value without its document cannot be re-checked.
        self.assertEqual(digest.record()["declared_by"], URL)

    def test_a_file_with_no_declared_digest_is_refused(self):
        """The negative path this module exists for.

        The release metadata names ``undeclared.fasta.gz`` and states its
        length, which is exactly the case that could be mistaken for
        verifiable. It is not, and staging it would put an unhashed corpus
        under a baseline.
        """

        with self.assertRaises(ValueError) as raised:
            parse_release_metalink(METALINK, "undeclared.fasta.gz", metadata_url=URL)
        self.assertIn("no digest", str(raised.exception))

    def test_a_digest_this_staging_cannot_check_is_refused_as_such(self):
        with self.assertRaises(ValueError) as raised:
            parse_release_metalink(METALINK, "exotic.fasta.gz", metadata_url=URL)
        self.assertIn("crc32", str(raised.exception))

    def test_a_file_with_no_declared_length_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            parse_release_metalink(METALINK, "unsized.fasta.gz", metadata_url=URL)
        self.assertIn("no size", str(raised.exception))

    def test_a_file_the_metadata_does_not_name_is_refused(self):
        with self.assertRaises(KeyError):
            parse_release_metalink(METALINK, "uniref100.fasta.gz", metadata_url=URL)


class StagedBytes(unittest.TestCase):
    payload = b"MKVLAAGIVGLNLGGK\n" * 64

    def staged(self, directory: Path, body: bytes) -> Path:
        path = directory / "payload.gz"
        path.write_bytes(body)
        return path

    def declaration(self, body: bytes) -> PublishedDigest:
        with tempfile.TemporaryDirectory() as raw:
            path = self.staged(Path(raw), body)
            md5 = file_digests(path, ("md5",))["md5"]
        return PublishedDigest(filename="payload.gz", size_bytes=len(body),
                               algorithm="md5", value=md5, metadata_url=URL)

    def test_matching_bytes_pass_and_record_the_repositorys_own_digest(self):
        declared = self.declaration(self.payload)
        with tempfile.TemporaryDirectory() as raw:
            path = self.staged(Path(raw), self.payload)
            record = verify_staged_file(path, declared)
        self.assertEqual(record["bytes"], len(self.payload))
        self.assertEqual(record["digests"]["md5"], declared.value)
        self.assertEqual(len(record["digests"]["sha256"]), 64)
        self.assertEqual(record["verified_against"]["declared_by"], URL)

    def test_a_truncated_transfer_is_refused_on_its_length(self):
        declared = self.declaration(self.payload)
        with tempfile.TemporaryDirectory() as raw:
            path = self.staged(Path(raw), self.payload[:-1])
            with self.assertRaises(ValueError) as raised:
                verify_staged_file(path, declared)
        self.assertIn("incomplete", str(raised.exception))

    def test_altered_bytes_of_the_declared_length_are_refused_on_the_digest(self):
        declared = self.declaration(self.payload)
        altered = b"R" + self.payload[1:]
        self.assertEqual(len(altered), len(self.payload))
        with tempfile.TemporaryDirectory() as raw:
            path = self.staged(Path(raw), altered)
            with self.assertRaises(ValueError) as raised:
                verify_staged_file(path, declared)
        self.assertIn("does not match", str(raised.exception))

    def test_a_missing_payload_is_refused(self):
        declared = self.declaration(self.payload)
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(FileNotFoundError):
                verify_staged_file(Path(raw) / "payload.gz", declared)


class SharedVolumeGuard(unittest.TestCase):
    def test_a_request_that_would_not_leave_the_reserve_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            available = free_bytes(directory)
            with self.assertRaises(RuntimeError) as raised:
                require_free_space(directory, available, reserve_bytes=1 << 30)
        self.assertIn("cannot complete", str(raised.exception))

    def test_a_request_that_fits_returns_the_free_space_it_measured(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            measured = require_free_space(directory, 1 << 10, reserve_bytes=1 << 10)
        self.assertGreater(measured, 1 << 11)

    def test_a_negative_request_is_a_defect_and_not_a_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError):
                require_free_space(Path(raw), -1, reserve_bytes=0)

    def test_the_guard_measures_the_volume_of_a_path_not_yet_created(self):
        with tempfile.TemporaryDirectory() as raw:
            nested = Path(raw) / "not" / "yet" / "there"
            self.assertEqual(free_bytes(nested), free_bytes(Path(raw)))


if __name__ == "__main__":
    unittest.main()
