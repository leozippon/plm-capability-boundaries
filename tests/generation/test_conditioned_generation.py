"""EXP-R2-227: the native conditioning interface, and the paths that must refuse.

Written against the properties the registration froze rather than against the
implementation. The four negative paths the campaign turns on are the reason this
file exists: a class queue that has drifted from its digest, an oracle that is not
staged, an arm that cannot be prompted, and CLEAN, which must come back
*unavailable* and never as a stub. A stub oracle is an instrument that always
agrees, and the whole campaign is a rate read off an instrument.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core import protein_annotations as ci  # noqa: E402
from src.capability.generation import conditioned_generation as cg  # noqa: E402


def _queue(classes: int = 8) -> dict:
    entries = cg.build_queue(
        [(f"c{index}", f"label {index}", 200 + index) for index in range(classes)],
        seed=cg.DRAW_SEED,
    )
    payload = {
        "pre_registration": cg.PRE_REGISTRATION,
        "draw": {"seed": cg.DRAW_SEED, "classes_per_arm": classes},
        "arms": {"zymctrl": {"label_kind": "ec_number", "classes": [e.record() for e in entries]}},
    }
    payload["digest"] = cg.queue_digest(payload)
    return payload


class TheFrozenQueue(unittest.TestCase):
    def test_the_draw_is_a_seeded_permutation_and_not_the_head_of_the_list(self):
        candidates = [f"{index}" for index in range(200)]
        drawn = cg.seeded_draw(candidates, n=16, seed=cg.DRAW_SEED)
        self.assertEqual(len(set(drawn)), 16)
        self.assertNotEqual(list(drawn), candidates[:16], "the draw is a file-order prefix")
        self.assertEqual(drawn, cg.seeded_draw(candidates, n=16, seed=cg.DRAW_SEED))
        self.assertNotEqual(drawn, cg.seeded_draw(candidates, n=16, seed=cg.DRAW_SEED + 1))

    def test_a_draw_larger_than_the_admissible_set_is_refused_not_topped_up(self):
        with self.assertRaisesRegex(RuntimeError, "never topped up"):
            cg.seeded_draw(["a", "b"], n=16, seed=cg.DRAW_SEED)

    def test_the_mismatched_pairing_maps_no_class_to_itself(self):
        for size in range(2, 40):
            mapping = cg.derangement(size, seed=cg.DRAW_SEED)
            self.assertEqual(sorted(mapping), list(range(size)))
            self.assertFalse(
                any(index == value for index, value in enumerate(mapping)),
                "a fixed point makes a class its own negative",
            )

    def test_a_queue_that_drifts_from_its_digest_is_refused(self):
        payload = _queue()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "class_queue.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(cg.load_queue(path)["digest"], payload["digest"])

            drifted = json.loads(json.dumps(payload))
            drifted["arms"]["zymctrl"]["classes"][0]["key"] = "swapped"
            path.write_text(json.dumps(drifted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted from the digest"):
                cg.load_queue(path)

            relabelled = json.loads(json.dumps(payload))
            relabelled["arms"]["zymctrl"]["classes"][0]["label"] = "a different request"
            path.write_text(json.dumps(relabelled), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted from the digest"):
                cg.load_queue(path)

            repaired = json.loads(json.dumps(payload))
            repaired["arms"]["zymctrl"]["classes"][0]["mismatched_key"] = "elsewhere"
            path.write_text(json.dumps(repaired), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted from the digest"):
                cg.load_queue(path)

    def test_a_queue_without_a_digest_or_from_another_campaign_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "class_queue.json"
            payload = _queue()
            payload.pop("digest")
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a frozen queue"):
                cg.load_queue(path)

            other = _queue()
            other["pre_registration"] = "EXP-R2-000"
            other["digest"] = cg.queue_digest(other)
            path.write_text(json.dumps(other), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "was frozen for"):
                cg.load_queue(path)

    def test_an_arm_the_queue_does_not_carry_is_refused(self):
        with self.assertRaisesRegex(KeyError, "carries no arm"):
            cg.queue_entries(_queue(), "prollama")

    def test_the_referent_and_anchor_draws_are_disjoint(self):
        pool = [f"SEQ{index}" for index in range(500)]
        referent, anchor = cg.split_draw(pool, seed=cg.DRAW_SEED)
        self.assertEqual(len(referent), cg.REFERENT_DRAW)
        self.assertEqual(len(anchor), cg.ANCHOR_DRAW)
        self.assertFalse(set(referent) & set(anchor), "the anchor would price a referent fitted on itself")

    def test_a_class_below_the_two_hundred_record_cut_cannot_supply_both_draws(self):
        with self.assertRaisesRegex(RuntimeError, "DISJOINT|disjoint"):
            cg.split_draw([f"S{index}" for index in range(150)], seed=cg.DRAW_SEED)


class TheOracle(unittest.TestCase):
    def test_clean_is_reported_unavailable_and_is_never_stubbed(self):
        with tempfile.TemporaryDirectory() as directory:
            record = cg.clean_availability(Path(directory))
            self.assertFalse(record["runnable"])
            self.assertTrue(record["missing"])
            self.assertIn("always agrees", record["never_stubbed"])
            self.assertIn("is not evidence", record["historical_anchor_is_not_availability"])
            for key in ("ec_prediction", "predicted_ec", "rate"):
                self.assertNotIn(key, record, "an unavailable instrument must emit no prediction")

    def test_the_real_clean_checkout_is_source_only_on_this_host(self):
        record = cg.clean_availability(REPO_ROOT / "external/code/clean/CLEAN")
        self.assertFalse(record["runnable"])
        self.assertIn("CLEAN trained model weights (*.pt/*.pth)", record["missing"])

    def test_the_structural_covariates_withhold_together_when_esmfold_is_absent(self):
        foldseek = REPO_ROOT / "external/tools/foldseek-linux-avx2.tar.gz"
        if not foldseek.is_file():
            self.skipTest('the foldseek archive is host-local and is not distributed '
                          'with the code repository; withholding together needs one '
                          'covariate present and the other absent')
        with tempfile.TemporaryDirectory() as directory:
            record = cg.structural_covariate_availability(
                esmfold=Path(directory) / "absent",
                foldseek_tarball=foldseek,
            )
        self.assertFalse(record["runnable"])
        self.assertTrue(record["foldseek_archive_present"])
        self.assertTrue(any("ESMFold" in entry for entry in record["missing"]))

    def test_hmmscan_refuses_both_or_neither_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            query = Path(directory) / "q.fasta"
            query.write_text(">a\nMKV\n", encoding="utf-8")
            tool = ci.HmmerTool(
                hmmscan=Path("/nonexistent/hmmscan"),
                hmmpress=Path("/nonexistent/hmmpress"),
                version="3.4",
                tarball=query,
                tarball_sha256="0" * 64,
                hmmscan_sha256="0" * 64,
            )
            database = ci.PfamDatabase(
                path=Path("/nonexistent/Pfam-A.hmm"), source_gz=query, source_sha256="0" * 64, n_profiles=1
            )
            for kwargs in ({}, {"evalue": 1e-5, "gathering_threshold": True}):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    ci.run_hmmscan(
                        tool, database, query, Path(directory) / "out.tbl", threads=1, **kwargs
                    )

    def test_an_empty_referent_assigns_nothing_rather_than_everything(self):
        hits = {"a": [{"accession_unversioned": "PF00001"}]}
        with self.assertRaisesRegex(ValueError, "unmeasurable"):
            cg.assigned(hits, ["a"], ())

    def test_the_referent_needs_the_declared_share_of_the_referent_draw(self):
        names = [f"r{index}" for index in range(100)]
        hits = {
            name: [{"accession_unversioned": "PF00001"}]
            for name in names[: int(cg.REFERENT_FAMILY_SHARE * 100)]
        }
        hits[names[99]] = [{"accession_unversioned": "PF09999"}]
        self.assertEqual(cg.referent_from_draw(hits, names), ("PF00001",))

    def test_a_class_that_fails_its_anchor_is_unmeasurable_and_says_why(self):
        admitted = cg.anchor_record(real=[True] * 80 + [False] * 20, random=[False] * 100, referent=("PF00001",))
        self.assertTrue(admitted["admitted"])
        self.assertEqual(admitted["unmeasurable_reasons"], [])

        weak = cg.anchor_record(real=[True] * 60 + [False] * 40, random=[False] * 100, referent=("PF00001",))
        self.assertFalse(weak["admitted"])
        self.assertTrue(any("below the 0.70 floor" in reason for reason in weak["unmeasurable_reasons"]))

        leaky = cg.anchor_record(real=[True] * 90 + [False] * 10, random=[True] * 20 + [False] * 80, referent=("PF00001",))
        self.assertFalse(leaky["admitted"])
        self.assertTrue(any("ceiling" in reason for reason in leaky["unmeasurable_reasons"]))

        empty = cg.anchor_record(real=[], random=[], referent=())
        self.assertFalse(empty["admitted"])
        self.assertTrue(any("no Pfam family" in reason for reason in empty["unmeasurable_reasons"]))


class TheArmsAndTheirPrompts(unittest.TestCase):
    class _Tokenizer:
        eos_token = "<|endoftext|>"

    class _Handle:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    def setUp(self):
        self.handle = self._Handle(self._Tokenizer())

    def test_every_declared_arm_renders_the_prompt_its_own_rendering_declares(self):
        self.assertEqual(
            cg.prompt_for(self.handle, cg.arm("zymctrl"), "3.2.1.17"), "3.2.1.17<sep><start>"
        )
        self.assertEqual(
            cg.prompt_for(self.handle, cg.arm("prollama"), "Lysozyme-like domain superfamily"),
            "[Generate by superfamily] Superfamily=<Lysozyme-like domain superfamily> Seq=<",
        )
        self.assertEqual(cg.prompt_for(self.handle, cg.arm("progen2-medium"), None), "1")
        self.assertEqual(cg.prompt_for(self.handle, cg.arm("protgpt2"), None), "<|endoftext|>\n")
        self.assertEqual(
            cg.prompt_for(self.handle, cg.arm("qwen2.5-0.5b"), "Greek"),
            "The following passage is written in Greek.\n\n",
        )
        self.assertEqual(cg.prompt_for(self.handle, cg.arm("qwen2.5-0.5b"), None), "")

    def test_a_conditioned_arm_cannot_be_prompted_without_a_class(self):
        with self.assertRaisesRegex(ValueError, "separate floor arm"):
            cg.prompt_for(self.handle, cg.arm("zymctrl"), None)
        with self.assertRaisesRegex(ValueError, "separate floor arm"):
            cg.prompt_for(self.handle, cg.arm("prollama"), None)

    def test_an_unconditioned_floor_cannot_be_given_a_class(self):
        with self.assertRaisesRegex(ValueError, "takes no class label"):
            cg.prompt_for(self.handle, cg.arm("progen2-medium"), "3.2.1.17")

    def test_an_arm_this_campaign_does_not_declare_is_refused_by_name(self):
        with self.assertRaisesRegex(KeyError, "unknown generation arm"):
            cg.arm("progen2-large")

    def test_an_arm_with_no_end_delimiter_and_no_eos_cannot_be_extracted_from(self):
        class Bare:
            tokenizer = type("T", (), {"eos_token": None})()

        with self.assertRaisesRegex(ValueError, "no end delimiter"):
            cg.end_delimiter_for(Bare(), cg.arm("protgpt2"))

    def test_a_class_and_its_mismatched_partner_never_share_a_sample(self):
        first = cg.cell_seed(seed=cg.SAMPLING_SEED, arm_name="zymctrl", class_key="a", condition="requested")
        second = cg.cell_seed(seed=cg.SAMPLING_SEED, arm_name="zymctrl", class_key="a", condition="mismatched")
        third = cg.cell_seed(seed=cg.SAMPLING_SEED, arm_name="zymctrl", class_key="b", condition="requested")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(
            first,
            cg.cell_seed(seed=cg.SAMPLING_SEED, arm_name="zymctrl", class_key="a", condition="requested"),
        )

    def test_the_frozen_sampling_and_bootstrap_parameters_cannot_be_moved(self):
        frozen = dict(
            resamples=cg.BOOTSTRAP_RESAMPLES,
            bootstrap_seed=cg.BOOTSTRAP_SEED,
            sampling_seed=cg.SAMPLING_SEED,
            generations=cg.GENERATIONS_PER_CELL,
            top_p=cg.TOP_P,
            temperature=cg.TEMPERATURE,
            max_new_tokens=cg.MAX_NEW_TOKENS,
        )
        cg.require_frozen_parameters(**frozen)
        for name in frozen:
            moved = dict(frozen)
            moved[name] = frozen[name] + 1
            with self.assertRaisesRegex(ValueError, "not revisable"):
                cg.require_frozen_parameters(**moved)


class TheRatesAndTheCompound(unittest.TestCase):
    def test_a_burst_of_near_identical_samples_counts_once(self):
        burst = ["MKVLAAGIVGLNLGGKMKVLAAGIVGLNLGG"] * 90
        distinct = [
            "".join("ACDEFGHIKLMNPQRSTVWY"[(index * 7 + position * 3) % 20] for position in range(60))
            for index in range(10)
        ]
        sequences = burst + distinct
        groups, summary = cg.near_duplicate_group_ids(sequences, unit="residues")
        self.assertLess(summary["n_groups_including_empty"], len(sequences))
        hits = [True] * len(burst) + [False] * len(distinct)
        self.assertLess(
            cg.grouped_rate(hits, groups),
            float(np.mean([1.0 if value else 0.0 for value in hits])),
            "the burst still inflates the rate",
        )

    def test_empty_generations_stay_in_the_denominator_as_one_group(self):
        sequences = ["MKVLA" * 6, ""] + [""] * 20
        groups, summary = cg.near_duplicate_group_ids(sequences, unit="residues")
        self.assertEqual(summary["n_empty_records"], 21)
        self.assertEqual(len(set(int(value) for value in groups[1:])), 1)
        self.assertEqual(cg.grouped_rate([True] + [False] * 21, groups), 0.5)

    def test_a_cohort_below_the_eight_class_floor_reports_no_interval(self):
        block = cg.class_clustered_mean(
            {f"c{index}": 0.5 for index in range(4)}, resamples=64, seed=cg.BOOTSTRAP_SEED
        )
        self.assertTrue(block["degenerate"])
        self.assertIsNone(block["ci95"])
        self.assertIsNone(block["mean"])

    def test_a_clear_positive_and_a_clear_null_are_separated_by_the_interval(self):
        positive = cg.class_clustered_mean(
            {f"c{index}": 0.4 + 0.01 * index for index in range(16)},
            resamples=cg.BOOTSTRAP_RESAMPLES,
            seed=cg.BOOTSTRAP_SEED,
        )
        self.assertTrue(cg.lower_bound_positive(positive))
        null = cg.class_clustered_mean(
            {f"c{index}": (0.2 if index % 2 else -0.2) for index in range(16)},
            resamples=cg.BOOTSTRAP_RESAMPLES,
            seed=cg.BOOTSTRAP_SEED,
        )
        self.assertFalse(cg.lower_bound_positive(null))

    def test_every_declared_outcome_of_the_compound_is_reachable(self):
        strong = {f"c{index}": 0.4 + 0.01 * index for index in range(16)}
        # Clauses 1 and 2 hold on the cohort mean while fewer than half the classes
        # are individually positive, which is the reading that issues no arm-level
        # verdict and is reported per class instead.
        weak = {f"c{index}": (1.0 if index < 7 else -0.02) for index in range(16)}
        block = lambda values: cg.class_clustered_mean(  # noqa: E731
            values, resamples=cg.BOOTSTRAP_RESAMPLES, seed=cg.BOOTSTRAP_SEED
        )
        null = {f"c{index}": (0.2 if index % 2 else -0.2) for index in range(16)}

        passed = cg.compound_verdict(
            against_mismatch=block(strong), against_floor=block(strong), per_class_contrast=strong
        )
        self.assertEqual(passed["outcome"], "conditioning_moves_generation_toward_the_requested_class")

        clause_one = cg.compound_verdict(
            against_mismatch=block(null), against_floor=block(strong), per_class_contrast=null
        )
        self.assertEqual(
            clause_one["outcome"], "tag_moves_the_distribution_without_selecting_the_requested_class"
        )

        clause_two = cg.compound_verdict(
            against_mismatch=block(strong), against_floor=block(null), per_class_contrast=strong
        )
        self.assertEqual(
            clause_two["outcome"], "selective_against_mismatch_but_not_above_the_unconditioned_floor"
        )

        clause_three = cg.compound_verdict(
            against_mismatch=block(weak), against_floor=block(weak), per_class_contrast=weak
        )
        self.assertEqual(clause_three["outcome"], "class_selective_on_part_of_the_label_space")

        short = {f"c{index}": 0.4 for index in range(4)}
        self.assertEqual(
            cg.compound_verdict(
                against_mismatch=block(short), against_floor=block(short), per_class_contrast=short
            )["outcome"],
            "not_scored",
        )

    def test_the_compound_needs_all_three_clauses_and_names_the_licence(self):
        strong = {f"c{index}": 0.4 for index in range(16)}
        block = cg.class_clustered_mean(strong, resamples=256, seed=cg.BOOTSTRAP_SEED)
        verdict = cg.compound_verdict(
            against_mismatch=block, against_floor=block, per_class_contrast=strong
        )
        self.assertTrue(verdict["clause_1_requested_minus_mismatched"])
        self.assertTrue(verdict["clause_2_requested_minus_floor"])
        self.assertTrue(verdict["clause_3_half_the_classes_individually_positive"])
        self.assertIn("behavioural capability statement", verdict["licence"])


class TheTextPositiveControl(unittest.TestCase):
    def test_the_script_ranges_are_disjoint_and_exclude_latin(self):
        cg.script_ranges_are_disjoint()
        self.assertGreaterEqual(len(cg.SCRIPTS), cg.MINIMUM_CLASSES)
        self.assertNotIn("latin", {key for key, _, _ in cg.SCRIPTS})

    def test_the_oracle_is_exact_on_text_of_a_known_script(self):
        self.assertEqual(cg.assign_script("Δοκιμαστικό κείμενο στα ελληνικά"), "greek")
        self.assertEqual(cg.assign_script("Пример текста на русском языке"), "cyrillic")
        self.assertEqual(cg.assign_script("טקסט לדוגמה בעברית"), "hebrew")
        self.assertEqual(cg.assign_script("ข้อความตัวอย่างภาษาไทย"), "thai")

    def test_a_latin_continuation_is_assigned_to_no_script(self):
        self.assertIsNone(cg.assign_script("The following passage is written in Greek."))
        self.assertIsNone(cg.assign_script(""))
        self.assertIsNone(
            cg.assign_script("A long English sentence with one stray letter α in it"),
            "two stray letters must not carry a class",
        )

    def test_the_script_queue_carries_the_same_fixed_point_free_pairing(self):
        entries = cg.script_classes()
        self.assertEqual(len(entries), len(cg.SCRIPTS))
        self.assertFalse(any(entry.key == entry.mismatched_key for entry in entries))


class TheStageContract(unittest.TestCase):
    def setUp(self):
        import importlib.util

        path = REPO_ROOT / "scripts/capability/stages/conditioned_generation.py"
        spec = importlib.util.spec_from_file_location("stage45", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.stage = module

    def test_a_conditioned_arm_owes_two_cells_per_class_and_no_floor_cell(self):
        entries = cg.build_queue([(f"c{index}", f"l{index}", 200) for index in range(16)], seed=cg.DRAW_SEED)
        cells = self.stage._cells(cg.arm("zymctrl"), entries)
        self.assertEqual(len(cells), 32)
        self.assertEqual(
            {cell["condition"] for cell in cells}, {"requested", "mismatched"}
        )

    def test_a_floor_arm_owes_exactly_one_unconditioned_cell(self):
        cells = self.stage._cells(cg.arm("progen2-medium"), ())
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["condition"], "unconditioned_floor")
        self.assertIsNone(cells[0]["label"])

    def test_a_text_arm_is_its_own_floor(self):
        cells = self.stage._cells(cg.arm("qwen2.5-0.5b"), cg.script_classes())
        self.assertEqual(len(cells), 2 * len(cg.SCRIPTS) + 1)
        self.assertEqual(
            sum(1 for cell in cells if cell["condition"] == "unconditioned_floor"), 1
        )

    def test_the_declared_floor_is_the_one_clause_two_reads(self):
        self.assertEqual(self.stage.PRIMARY_FLOOR, "progen2-medium")
        self.assertIn(self.stage.PRIMARY_FLOOR, cg.FLOORS["zymctrl"])

    def test_every_artefact_carries_the_binding_ceiling(self):
        payload = self.stage._preamble("probe")
        for key in cg.CEILING:
            self.assertIn(key, payload["ceiling"])
        self.assertIn("does not reopen internal-feature steering", payload["not_the_retired_steering_line"])

    def test_generation_refuses_an_arm_whose_interface_check_is_absent_or_failed(self):
        import argparse

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(arm="zymctrl", self_check_dir=Path(directory))
            with self.assertRaisesRegex(RuntimeError, "has not passed its interface check"):
                self.stage.require_self_check(args)

            path = Path(directory) / "generation_self_check_zymctrl.json"
            path.write_text(json.dumps({"arm": "zymctrl", "passed": False}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "this arm stops"):
                self.stage.require_self_check(args)

            path.write_text(json.dumps({"arm": "protgpt2", "passed": True}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "this arm stops"):
                self.stage.require_self_check(args)

            path.write_text(json.dumps({"arm": "zymctrl", "passed": True}), encoding="utf-8")
            self.assertTrue(self.stage.require_self_check(args)["passed"])

    def test_generation_without_a_self_check_directory_is_refused_at_the_command_line(self):
        with self.assertRaises(SystemExit):
            argv = ["--stage", "generate", "--arm", "zymctrl"]
            parsed = self.stage.build_parser().parse_args(argv)
            self.assertIsNone(parsed.self_check_dir)
            sys.argv = ["conditioned_generation.py", *argv]
            self.stage.main()

    def test_the_frozen_queue_travels_with_the_frozen_code(self):
        if not (REPO_ROOT / "results").is_dir():
            self.skipTest("results/ is absent: the frozen queue is a retained result and is "
                          "not distributed with the code repository")
        self.assertTrue(
            self.stage.FROZEN_QUEUE.is_file(),
            "the frozen queue must exist in the checkout the stage is read from",
        )
        payload = cg.load_queue(self.stage.FROZEN_QUEUE)
        self.assertEqual(len(payload["arms"]["zymctrl"]["classes"]), cg.CLASSES_PER_ARM)
        self.assertEqual(payload["draw"]["seed"], cg.DRAW_SEED)

    def test_the_parser_refuses_a_stage_it_does_not_declare(self):
        with self.assertRaises(SystemExit):
            self.stage.build_parser().parse_args(["--stage", "invented"])


class TheScoringPathOnAPlantedWorld(unittest.TestCase):
    """The whole per-arm report, run against an oracle whose answer is known.

    The external oracle is stubbed here and ONLY here, and it is stubbed with a
    *planted* answer rather than an agreeable one: in the first world the arm
    generates exactly the class it was asked for, in the second it generates a
    class drawn without reference to the request. A pipeline that cannot tell
    those two apart is the failure this campaign exists to avoid, and the
    difference between them is precisely what a 1.73-nat leak cannot decide.
    """

    CLASSES = 16
    PER_CELL = 40

    def setUp(self):
        import argparse
        import importlib.util

        path = REPO_ROOT / "scripts/capability/stages/conditioned_generation.py"
        spec = importlib.util.spec_from_file_location("stage45_planted", path)
        self.stage = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.stage
        spec.loader.exec_module(self.stage)
        self.entries = cg.build_queue(
            [(f"c{index}", f"class {index}", 400) for index in range(self.CLASSES)],
            seed=cg.DRAW_SEED,
        )
        self.queue = {
            "digest": "planted",
            "arms": {"zymctrl": {"classes": [entry.record() for entry in self.entries]}},
        }
        self.anchors = {
            "arm_scorable": True,
            "admitted_classes": [entry.key for entry in self.entries],
            "classes": {
                entry.key: {
                    "admitted": True,
                    "referent": [f"PF{index:05d}"],
                    "real_rate": 1.0,
                    "random_rate": 0.0,
                }
                for index, entry in enumerate(self.entries)
            },
        }
        self.args = argparse.Namespace(
            work=Path(tempfile.mkdtemp()),
            hmmscan_threads=1,
            hmmscan_shards=1,
            bootstrap=cg.BOOTSTRAP_RESAMPLES,
            bootstrap_seed=cg.BOOTSTRAP_SEED,
            identity_corpus=None,
        )
        self.index = {entry.key: index for index, entry in enumerate(self.entries)}

    def _sequence(self, tag: str, index: int) -> str:
        rng = np.random.default_rng(abs(hash((tag, index))) % (2**32))
        return "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=120))

    def _generations(self) -> dict:
        cells = {}
        for entry in self.entries:
            for condition in ("requested", "mismatched"):
                key = f"{entry.key}|{condition}"
                cells[key] = {
                    "samples": [self._sequence(key, index) for index in range(self.PER_CELL)],
                    "statistics": {"n": self.PER_CELL},
                }
        return {"cells": cells, "not_run_reason": None}

    def _floors(self) -> dict:
        floors = {}
        for name in ("progen2-medium", "protgpt2"):
            key = "__unconditioned__|unconditioned_floor"
            floors[name] = {
                "cells": {
                    key: {
                        "samples": [self._sequence(name, index) for index in range(self.PER_CELL)],
                        "statistics": {"n": self.PER_CELL},
                    }
                }
            }
        return floors

    def _stub(self, world: str):
        entries = self.entries
        lookup = {entry.key: entry for entry in entries}
        index_of = self.index

        def annotate(sequences, **_):
            hits = {}
            for position, name in enumerate(sequences):
                source, cell, _ = name.split("#")
                if cell.endswith("unconditioned_floor"):
                    continue
                class_key, condition = cell.split("|")
                if world == "selective":
                    produced = (
                        class_key if condition == "requested" else lookup[class_key].mismatched_key
                    )
                elif world == "indiscriminate":
                    produced = entries[position % len(entries)].key
                else:  # pragma: no cover - a world nobody planted
                    raise AssertionError(world)
                hits[name] = [{"accession_unversioned": f"PF{index_of[produced]:05d}"}]
            return hits, {"label": world}

        return annotate

    def _report(self, world: str) -> dict:
        original = self.stage.cg.annotate
        self.stage.cg.annotate = self._stub(world)
        try:
            return self.stage._protein_arm_report(
                arm_name="zymctrl",
                queue=self.queue,
                anchors=self.anchors,
                generations=self._generations(),
                floors=self._floors(),
                args=self.args,
                tool=None,
                database=None,
            )
        finally:
            self.stage.cg.annotate = original

    def test_an_arm_that_generates_the_requested_class_clears_the_compound(self):
        report = self._report("selective")
        self.assertEqual(
            report["verdict"]["outcome"], "conditioning_moves_generation_toward_the_requested_class"
        )
        self.assertEqual(report["verdict"]["n_classes_individually_positive"], self.CLASSES)
        for block in report["per_class"].values():
            self.assertEqual(block["p_requested"], 1.0)
            self.assertEqual(block["p_mismatched"], 0.0)
            self.assertEqual(block["p_floor"]["progen2-medium"], 0.0)

    def test_an_arm_that_moves_the_distribution_without_selecting_fails_clause_one(self):
        report = self._report("indiscriminate")
        self.assertEqual(
            report["verdict"]["outcome"],
            "tag_moves_the_distribution_without_selecting_the_requested_class",
        )

    def test_a_shortfall_of_admitted_classes_stops_the_arm_rather_than_narrowing_it(self):
        self.anchors["arm_scorable"] = False
        self.anchors["shortfall_note"] = "fewer than eight classes survived the anchor"
        report = self._report("selective")
        self.assertEqual(report["verdict"]["outcome"], "not_scored")
        self.assertIn("fewer than eight", report["not_scored_reason"])

    def test_a_missing_declared_floor_is_refused_rather_than_substituted(self):
        original = self.stage.cg.annotate
        self.stage.cg.annotate = self._stub("selective")
        try:
            with self.assertRaisesRegex(RuntimeError, "not scored against a substitute"):
                self.stage._protein_arm_report(
                    arm_name="zymctrl",
                    queue=self.queue,
                    anchors=self.anchors,
                    generations=self._generations(),
                    floors={"protgpt2": self._floors()["protgpt2"]},
                    args=self.args,
                    tool=None,
                    database=None,
                )
        finally:
            self.stage.cg.annotate = original

    def test_the_max_identity_covariate_is_reported_not_run_rather_than_as_no_homology(self):
        report = self._report("selective")
        covariate = report["max_identity_to_corpus"]
        self.assertFalse(covariate["run"])
        self.assertIn("NOT RUN", covariate["reason"])
        self.assertIn("never claimed as novelty", covariate["ceiling"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
