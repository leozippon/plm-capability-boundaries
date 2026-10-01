"""EXP-R2-246 door: prompts, refusals, fake generation, score-blind sampling."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.generation import unconditional_generation as ug  # noqa: E402
from src.capability.core.arms import BOS_DIRECTION_N_TO_C, EOS_BOUNDED_BOUNDARY, N_TO_C_MARKER  # noqa: E402
from src.capability.models.joint_modes import rendering  # noqa: E402
from src.capability.models.proteinglm import NATIVE_PREFIX  # noqa: E402


def _stage():
    path = REPO / "scripts" / "capability" / "stages/unconditional_generation.py"
    spec = importlib.util.spec_from_file_location("stage48_unconditional_generation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptTable(unittest.TestCase):
    def test_native_prompts_and_close_tokens_match_the_declared_renderings(self):
        galactica = rendering("galactica")
        instruct = rendering("instructprotein")
        prollama = rendering("prollama")
        expected = {
            "protgpt2": ("<|endoftext|>\n", "<|endoftext|>"),
            "progen2-small": (N_TO_C_MARKER, "2"),
            "progen2-base": (N_TO_C_MARKER, "2"),
            "progen2-medium": (N_TO_C_MARKER, "2"),
            "progen2-large": (N_TO_C_MARKER, "2"),
            "progen2-xlarge": (N_TO_C_MARKER, "2"),
            "progen3-112m": (N_TO_C_MARKER, "<eos>"),
            "proteinglm-7b-clm": (NATIVE_PREFIX, "<eos>"),
            "protgpt3-1.3b": (f"<|bos|>{BOS_DIRECTION_N_TO_C}", "<|eos|>"),
            "rita-xl": (EOS_BOUNDED_BOUNDARY, EOS_BOUNDED_BOUNDARY),
            "galactica-125m": (galactica.protein_start, galactica.protein_end),
            "galactica-1.3b": (galactica.protein_start, galactica.protein_end),
            "galactica-6.7b": (galactica.protein_start, galactica.protein_end),
            "galactica-30b": (galactica.protein_start, galactica.protein_end),
            "instructprotein": (
                f"{instruct.prefix_marker}{instruct.protein_start}",
                instruct.protein_end,
            ),
            "prollama-stage-1": (prollama.protein_start, prollama.protein_end),
            "prollama": (prollama.protein_start, prollama.protein_end),
        }
        self.assertEqual(set(expected), set(ug.ADMITTED))
        self.assertEqual(len(ug.ADMITTED_ARMS), 17)
        self.assertNotIn("progen3-3b", ug.ADMITTED)
        for arm, (prompt, close) in expected.items():
            self.assertEqual(ug.prompt_for(arm), prompt, arm)
            self.assertEqual(ug.close_token_for(arm), close, arm)
        self.assertEqual(ug.arm_batch_size("rita-xl"), 1)
        self.assertEqual(ug.arm_batch_size("protgpt2"), 8)
        self.assertEqual(ug.GENERATION_ORDER[-1], "galactica-30b")
        self.assertFalse(ug.ADMITTED["instructprotein"].add_special_tokens)
        self.assertFalse(ug.ADMITTED["rita-xl"].add_special_tokens)
        self.assertTrue(ug.ADMITTED["protgpt2"].add_special_tokens)

    def test_refusals_name_the_excluded_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "EC class tag"):
            ug.prompt_for("zymctrl")
        with self.assertRaisesRegex(ValueError, "scoring wrap"):
            ug.require_admitted("llama-2-7b")
        with self.assertRaisesRegex(ValueError, "already complete"):
            ug.require_admitted("progen3-3b")
        with self.assertRaisesRegex(ValueError, "text-only"):
            ug.require_admitted("gpt2-large")

    def test_ensure_generate_mixes_in_generate_and_a_config(self):
        from transformers import PretrainedConfig

        class Dummy:
            def __init__(self):
                self.config = PretrainedConfig()
                self.generation_config = None

        dummy = Dummy()
        patched = ug.cg.ensure_generate(dummy)
        self.assertTrue(callable(patched.generate))
        self.assertIsNotNone(patched.generation_config)
        self.assertTrue(patched.generation_config._from_model_config)

    def test_forward_sampler_does_not_call_generate(self):
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.device = torch.device("cpu")
                self.embed = torch.nn.Embedding(8, 4)
                self.proj = torch.nn.Linear(4, 8)

            def generate(self, *args, **kwargs):
                raise AssertionError("forward sampler must not call generate")

            def forward(self, input_ids):
                hidden = self.embed(input_ids)
                return type("Out", (), {"logits": self.proj(hidden)})()

        class Tok:
            def __call__(self, text, return_tensors=None, add_special_tokens=False):
                return {"input_ids": torch.tensor([[1]])}

            def convert_tokens_to_ids(self, token):
                return 2

            def decode(self, ids, skip_special_tokens=False):
                return "A" * len(list(ids))

        spec = ug.ADMITTED["rita-xl"]
        raws = ug._sample_by_forward(spec, Tiny(), Tok(), n=2, seed=0)
        self.assertEqual(len(raws), 2)
        self.assertTrue(all(isinstance(item, str) for item in raws))


class FakeGeneration(unittest.TestCase):
    def test_eight_attempts_are_retained_including_an_empty_extract(self):
        raws = [
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTAYIAKQRQISFVKSHFS",
            "",
            "!!!not-a-residue-run",
            "AAAA2TRAILING",
            "G" * 40,
            "ACDE  FGHI",
            "VVVVVVVVVVVVVVVV",
        ]

        def fake_generate(**_kwargs):
            return list(raws)

        rows = ug.generate("progen2-small", n=8, generate_fn=fake_generate)
        self.assertEqual(len(rows), 8)
        self.assertEqual([row["raw_continuation"] for row in rows], raws)
        self.assertEqual(rows[2]["sequence"], "")
        self.assertEqual(rows[3]["sequence"], "")
        self.assertEqual(rows[2]["stop_status"], "empty_or_noncanonical")
        self.assertEqual(rows[4]["sequence"], "AAAA")
        self.assertEqual(rows[4]["stop_status"], "native_terminal")
        self.assertEqual(rows[6]["sequence"], "ACDEFGHI")
        for row in rows:
            self.assertEqual(row["arm"], "progen2-small")
            self.assertEqual(row["condition"], "unconditioned")
            self.assertIsNone(row["class_key"])
            self.assertIsNone(row["target_profile_hit"])
            self.assertEqual(row["length"], len(row["sequence"]))
            self.assertEqual(row["sequence_sha256"], ug.ge.sequence_hash(row["sequence"]))

    def test_structure_selection_ignores_permuted_profile_fields(self):
        rows = []
        for index, length in enumerate([40] * 40 + [180] * 40 + [300] * 40 + [600] * 40):
            row = ug.attempt_row("protgpt2", "A" * length, index)
            row.update(
                {
                    "target_profile_hit": index % 2 == 0,
                    "any_profile_hit": True,
                    "reference_identity": float(index),
                    "model_score": 1000 - index,
                    "pfam_families": [f"PF{index}"],
                }
            )
            rows.append(row)
        first, _ = ug.select_structure_parents(rows)
        parents = [row["id"] for row in first if row["role"] == "generation"]
        self.assertEqual(len(parents), 128)
        self.assertEqual(sum(row["role"] == "composition_shuffle" for row in first), 128)
        permuted = []
        for index, row in enumerate(rows):
            copy = dict(row)
            copy.update(
                {
                    "target_profile_hit": not row["target_profile_hit"],
                    "any_profile_hit": False,
                    "reference_identity": -row["reference_identity"],
                    "model_score": row["model_score"] + 17,
                    "pfam_families": ["PFNONE"],
                    "selected_for_structure": False,
                    "structure_exclusion_reason": None,
                    "phase": None,
                    "inclusion_probability": None,
                }
            )
            permuted.append(copy)
        second, _ = ug.select_structure_parents(permuted)
        self.assertEqual(
            [row["id"] for row in second if row["role"] == "generation"],
            parents,
        )


class StageInterface(unittest.TestCase):
    def test_interface_only_writes_no_scientific_ledger(self):
        stage = _stage()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "protgpt2"
            stage.main(
                [
                    "--stage",
                    "interface-only",
                    "--arm",
                    "protgpt2",
                    "--out",
                    str(out),
                    "--device",
                    "cpu",
                ]
            )
            self.assertTrue((out / ug.EXPECT_INTERFACE).is_file())
            self.assertFalse((out / ug.LEDGER_NAME).exists())
            record = json.loads((out / ug.EXPECT_INTERFACE).read_text(encoding="utf-8"))
            self.assertTrue(record["interface_only"])
            self.assertFalse(record["is_scientific_measurement"])
            self.assertEqual(record["prompt"], "<|endoftext|>\n")

    def test_generate_with_a_fake_model_writes_the_ledger_under_out(self):
        raws = ["ACDE", "", "GGGGGGGGGGGGGGGG"]

        def fake_generate(**_kwargs):
            return list(raws)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "tiny"
            rows = ug.generate("galactica-125m", n=3, generate_fn=fake_generate, output_dir=out)
            self.assertEqual(len(rows), 3)
            self.assertTrue((out / ug.LEDGER_NAME).is_file())
            self.assertTrue((out / ug.EXPECT_GENERATE).is_file())
            summary = json.loads((out / ug.EXPECT_GENERATE).read_text(encoding="utf-8"))
            self.assertEqual(summary["attempts"], 3)
            self.assertFalse(summary["is_scientific_measurement"])
            self.assertEqual(ug.extract_residues("galactica-125m", "MSPL1T-TH1S-Pl3A5EK"), "MK")

    def test_checkpoint_resolution_does_not_load_galactica_30b(self):
        path = ug.checkpoint_for_arm("galactica-30b")
        self.assertEqual(path.name, "galactica-30b")
        self.assertNotIn("load", path.name)


class CampaignManifest(unittest.TestCase):
    def test_tsv_has_seventeen_generate_cells_and_defers_30b(self):
        from scripts.capability.campaigns import ARCHIVE, records
        if not ARCHIVE.is_file():
            self.skipTest(
                "the historical recipe archive is host-local under h200/ and is not distributed"
            )
        text = records()["campaign_s48_unconditional_generation.tsv"]["content"]
        self.assertIn("not launch beside", text)
        rows = [line.split("\t") for line in text.splitlines() if line and not line.startswith("#")]
        self.assertEqual(len(rows), 17)
        arms = [fields[7].split()[-1] for fields in rows]
        self.assertEqual(arms, list(ug.GENERATION_ORDER))
        self.assertEqual(arms[-1], "galactica-30b")
        self.assertNotIn("progen3-3b", arms)
        self.assertEqual(rows[-1][0], "5")
        self.assertEqual({fields[2] for fields in rows if fields[0] != "5"}, {"0", "1", "2", "3"})
        self.assertTrue(all("48_unconditional_generation.py" == fields[3] for fields in rows))
        # Shape only: the archived env column names the project root of the machine that
        # wrote it, and that root is host-local. The runtime suffix and the absence of
        # whitespace are the parts that carry the recipe's identity.
        self.assertTrue(
            all(
                fields[5].startswith("TRANSFER_PYTHON=")
                and fields[5].endswith("/runtimes/ct-20260905/bin/python")
                and not any(character.isspace() for character in fields[5])
                for fields in rows
            )
        )


class WaiterParser(unittest.TestCase):
    SCRIPT = REPO / "h200" / "history" / "wait_then_queue_s48_unconditional_generation.sh"

    def test_argument_parser_and_required_pod(self):
        if not self.SCRIPT.is_file():
            self.skipTest("host-local H200 operations are not provisioned")
        syntax = subprocess.run(
            ["bash", "-n", str(self.SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        missing = subprocess.run(
            ["bash", str(self.SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            env={key: value for key, value in os.environ.items() if key != "H200_POD"},
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("H200_POD", missing.stderr)
        unknown = subprocess.run(
            ["bash", str(self.SCRIPT), "--surprise"],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "H200_POD": "not-persisted"},
        )
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown argument", unknown.stderr)
        body = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--now", body)
        self.assertIn("s29_galactica_instructprotein.status.tsv", body)
        self.assertIn("campaign_s46_homologue_expansion.status.tsv", body)
        self.assertIn("require_queue_finished", body)
        self.assertNotIn("source ", body)
        self.assertNotIn("refusing to launch s48 after", body)


if __name__ == "__main__":
    unittest.main()
