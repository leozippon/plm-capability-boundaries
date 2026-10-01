"""The dataset registry has to describe the bytes that are actually staged.

Eleven of the twelve dataset directories that carried measurements into the
readout and pairwise cohorts had no recorded digest, release or retrieval route
at all, while the evidence discipline those cohorts are held to requires an
external resource to be hash-bound before it enters a baseline. A registry that
is not checked is the same gap wearing a manifest, so these tests read the
staged files: every listed file must exist at its recorded size, a bounded
sample must reproduce its recorded digest, and an altered or unlisted file must
be refused rather than tolerated.

The provenance half is authored rather than measured, so it is checked for a
different failure: a field that is neither a value nor an explicitly reasoned
unrecoverable marker. A fabricated release string is worse than an admitted
gap, and the schema is what keeps the gap visible.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.dataset_registry import (  # noqa: E402
    PROVENANCE_FIELDS,
    REGISTRY_RELPATH,
    SCHEMA_VERSION,
    inventory,
    is_volatile,
    structural_problems,
    verify_dataset,
)

REGISTRY_PATH = REPO_ROOT / REGISTRY_RELPATH
DATA_ROOT = REPO_ROOT / "data"

#: Digest verification of the staged copy is bounded so the suite stays quick:
#: every listed file is checked for existence and size, and files at or below
#: this size are rehashed until the byte budget is spent. The unbounded pass is
#: scripts/capability/data/verify_dataset_registry.py, which is the operational gate.
SAMPLE_MAX_FILE_BYTES = 4 << 20
SAMPLE_BUDGET_BYTES = 512 << 20


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


@unittest.skipUnless(
    REGISTRY_PATH.is_file(),
    f"{REGISTRY_RELPATH} is absent; data/ is git-ignored and need not be staged",
)
class StagedRegistry(unittest.TestCase):
    def test_the_registry_is_internally_consistent(self):
        problems = structural_problems(load_registry())
        self.assertEqual(problems, [], "\n".join(problems))

    def test_declared_datasets_and_staged_directories_are_the_same_set(self):
        registry = load_registry()
        declared = set(registry["datasets"])
        present = {path.name for path in DATA_ROOT.iterdir() if path.is_dir()}
        self.assertEqual(
            declared,
            present,
            "a staged dataset directory with no registry entry carries no provenance, "
            "and a declared dataset with no directory cannot be verified",
        )

    def test_every_listed_file_is_staged_at_its_recorded_size_and_digest(self):
        registry = load_registry()
        failures: list[str] = []
        rehashed = 0
        for name in sorted(registry["datasets"]):
            entry = registry["datasets"][name]
            if is_volatile(entry):
                continue
            dataset_failures, hashed, _ = verify_dataset(
                DATA_ROOT / name,
                entry["inventory"]["files"],
                max_bytes_per_file=SAMPLE_MAX_FILE_BYTES,
                budget_bytes=SAMPLE_BUDGET_BYTES,
            )
            failures.extend(dataset_failures)
            rehashed += hashed
        self.assertEqual(failures, [], "\n".join(failures))
        self.assertGreater(rehashed, 0, "no digest was recomputed, so this test proved nothing")

    def test_a_volatile_dataset_says_why_it_cannot_be_digest_bound(self):
        registry = load_registry()
        volatile = {
            name: entry["volatile"]["reason"]
            for name, entry in registry["datasets"].items()
            if is_volatile(entry)
        }
        for name, reason in sorted(volatile.items()):
            with self.subTest(dataset=name):
                self.assertTrue(reason.strip(), f"{name} is volatile with an empty reason")
        self.assertLess(
            len(volatile),
            len(registry["datasets"]),
            "every dataset is excluded from digest verification, so the registry binds nothing",
        )

    def test_the_provenance_of_every_dataset_records_a_licence_and_a_redistribution_rule(self):
        registry = load_registry()
        for name, entry in sorted(registry["datasets"].items()):
            with self.subTest(dataset=name):
                for field in ("licence", "redistribution"):
                    self.assertIn(field, entry)


class RefusalOnDisagreement(unittest.TestCase):
    """The negative paths, on a fixture rather than on the staged copy."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name) / "fixture"
        self.directory.mkdir()
        (self.directory / "kept.txt").write_text("measured bytes\n", encoding="utf-8")
        (self.directory / "nested").mkdir()
        (self.directory / "nested" / "also.txt").write_text("more bytes\n", encoding="utf-8")
        self.records = [record.as_json() for record in inventory(self.directory)]
        self.addCleanup(self.tmp.cleanup)

    def test_an_unaltered_directory_passes(self):
        failures, hashed, read = verify_dataset(self.directory, self.records)
        self.assertEqual(failures, [])
        self.assertEqual(hashed, 2)
        self.assertGreater(read, 0)

    def test_a_file_whose_digest_no_longer_matches_is_refused(self):
        altered = self.directory / "kept.txt"
        altered.write_text("measured bYtes\n", encoding="utf-8")
        self.assertEqual(
            altered.stat().st_size,
            next(r["bytes"] for r in self.records if r["path"] == "kept.txt"),
            "the edit must keep the size so the digest is what catches it",
        )
        failures, _, _ = verify_dataset(self.directory, self.records)
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("kept.txt", failures[0])
        self.assertIn(hashlib.sha256(altered.read_bytes()).hexdigest(), failures[0])

    def test_a_resized_file_is_refused_without_rehashing_it(self):
        (self.directory / "kept.txt").write_text("truncated\n", encoding="utf-8")
        failures, hashed, _ = verify_dataset(self.directory, self.records)
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("bytes, registry says", failures[0])
        self.assertEqual(hashed, 1)

    def test_a_listed_file_that_is_gone_is_refused(self):
        (self.directory / "nested" / "also.txt").unlink()
        failures, _, _ = verify_dataset(self.directory, self.records)
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("listed but missing", failures[0])

    def test_a_staged_file_the_registry_does_not_list_is_refused(self):
        (self.directory / "arrived_later.txt").write_text("unrecorded\n", encoding="utf-8")
        failures, _, _ = verify_dataset(self.directory, self.records)
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("not in the registry", failures[0])


class ProvenanceSchema(unittest.TestCase):
    """An admitted gap has to stay visible, and a missing field has to fail."""

    def registry_with(self, **overrides) -> dict:
        entry: dict = {field: "recorded" for field in PROVENANCE_FIELDS}
        entry["inventory"] = {
            "measured_utc": "2026-09-23T00:00:00Z",
            "file_count": 0,
            "total_bytes": 0,
            "files": [],
        }
        entry.update(overrides)
        return {"schema_version": SCHEMA_VERSION, "datasets": {"fixture": entry}}

    def test_a_fully_recorded_entry_passes(self):
        self.assertEqual(structural_problems(self.registry_with()), [])

    def test_an_unrecoverable_field_passes_only_with_a_reason(self):
        with_reason = {"recoverable": False, "reason": "no download record was retained"}
        self.assertEqual(structural_problems(self.registry_with(release=with_reason)), [])
        problems = structural_problems(self.registry_with(release={"recoverable": False}))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("unrecoverable without a reason", problems[0])

    def test_a_missing_or_empty_provenance_field_is_refused(self):
        registry = self.registry_with()
        del registry["datasets"]["fixture"]["release"]
        self.assertIn("fixture.release is missing", structural_problems(registry))
        self.assertIn(
            "fixture.release is an empty string",
            structural_problems(self.registry_with(release="   ")),
        )

    def test_a_volatile_marker_without_a_reason_is_refused(self):
        self.assertEqual(structural_problems(self.registry_with(volatile={"reason": "staging"})), [])
        problems = structural_problems(self.registry_with(volatile={}))
        self.assertEqual(problems, ["fixture.volatile carries no reason"])

    def test_inventory_arithmetic_that_disagrees_with_the_file_list_is_refused(self):
        registry = self.registry_with()
        registry["datasets"]["fixture"]["inventory"]["files"] = [
            {"path": "one", "bytes": 3, "sha256": "a" * 64}
        ]
        problems = structural_problems(registry)
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(any("file_count" in problem for problem in problems), problems)
        self.assertTrue(any("total_bytes" in problem for problem in problems), problems)


if __name__ == "__main__":
    unittest.main()
