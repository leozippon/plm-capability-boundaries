"""Native compilation, censored attempts, and the new fixed structural sample."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.capability.core.io import sha256_file
from src.capability.generation import progen3_generation as pg
from src.capability.models import progen3 as progen3_loader


class NativeAttemptAccounting(unittest.TestCase):
    def test_completed_and_censored_attempts_remain_distinct_and_present(self):
        complete = pg.generation_row("ACDEFGHIKLMNPQRSTVWY2<eos><pad>", "ACDEFGHIKLMNPQRSTVWY", 0)
        fragment = pg.generation_row("ACDEFGHIKLMNPQRSTVWY", None, 1)
        invalid = pg.generation_row("<unk>ACDE", None, 2)
        self.assertEqual(complete["source_stop_reason"], "native_terminal_eos")
        self.assertEqual(fragment["source_stop_reason"], "budget_censored_residue_continuation")
        self.assertEqual(fragment["sequence"], complete["sequence"])
        self.assertFalse(fragment["official_compilation_valid"])
        self.assertEqual(invalid["sequence"], "")
        self.assertEqual(invalid["raw_continuation"], "<unk>ACDE")
        self.assertEqual(invalid["support_status"], "empty_sequence")
        with self.assertRaisesRegex(ValueError, "compilation differs"):
            pg.generation_row("ACDE2<eos>", "AAAA", 3)

    def test_structure_selection_keeps_every_attempt_and_never_copies_native_metadata_to_shuffle(self):
        rows = [pg.generation_row("A" * (40 + i), None, i) for i in range(200)]
        for row in rows:
            row["near_duplicate_group"] = "one_group"
        subset, strata = pg.select_structure(rows)
        self.assertEqual(len(rows), 200)
        self.assertEqual(len(subset), 256)
        parents = [row for row in subset if row["role"] == "generation"]
        shuffles = [row for row in subset if row["role"] == "composition_shuffle"]
        self.assertEqual(sum(block["n"] for block in strata.values()), 128)
        self.assertAlmostEqual(sum(1 / row["inclusion_probability"] for row in parents), 200)
        self.assertTrue(all(row["raw_continuation"] is None for row in shuffles))
        self.assertTrue(all(row["official_compilation_valid"] is None for row in shuffles))
        self.assertEqual({row["paired_id"] for row in shuffles}, {row["id"] for row in parents})

    def test_cache_compatibility_returns_the_actual_cache_or_refuses_absence(self):
        model = SimpleNamespace()
        pg.install_generation_compatibility(model)
        cache = object()
        self.assertEqual(model._extract_past_from_model_output(SimpleNamespace(past_key_values=cache)), ("past_key_values", cache))
        with self.assertRaisesRegex(RuntimeError, "required KV cache"):
            model._extract_past_from_model_output(SimpleNamespace(past_key_values=None))


class TheFingerprintNamesFilesThatExist(unittest.TestCase):
    """The fingerprint runs before any inference, so a path it cannot resolve
    stops every generation run. It names the strict loader module, which lives
    in a different package from this runner."""

    def test_checkpoint_fingerprint_digests_every_declared_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            source = root / "source" / "progen3"
            source.mkdir(parents=True)
            for name in ("generator.py", "modeling.py", "batch_preparer.py", "tokenizer.json"):
                (source / name).write_text(name)
            fingerprint = pg.checkpoint_fingerprint(checkpoint, source.parent)
        self.assertEqual(set(fingerprint["checkpoint_files"]),
                         {"config.json", "model-00001-of-00001.safetensors"})
        self.assertEqual(set(fingerprint["upstream_generation_files"]),
                         {"generator.py", "modeling.py", "batch_preparer.py", "tokenizer.json"})
        loader = Path(progen3_loader.__file__)
        self.assertTrue(loader.is_file(), loader)
        self.assertEqual(fingerprint["strict_loader_sha256"], sha256_file(loader))


if __name__ == "__main__":
    unittest.main()
