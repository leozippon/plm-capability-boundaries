"""CPU contract for the InstructProtein family-generation feasibility probe.

Written against the properties the probe depends on rather than against its
implementation: the prompt is the released checkpoint's own family rendering, the
label is the naming channel its instruction data was written in, the residue span
is recovered from the token ids by the identity rule the family declares, the
class draw is the campaign's seeded permutation rather than a list a caller could
hand-pick, and the secondary read is a 5-mer comparison and nothing more. No
model, no GPU and no corpus is read here.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation import conditioned_generation as cg  # noqa: E402
from src.capability.models import joint_modes as jm  # noqa: E402


def _load_probe():
    path = REPO_ROOT / "scripts/capability/generation/instructprotein_family_generation_probe.py"
    spec = importlib.util.spec_from_file_location("instructprotein_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROBE = _load_probe()


class TheResidueSpan(unittest.TestCase):
    """One residue per declared token, taken by identity and not by position."""

    LETTERS = {50267: "A", 50268: "C", 50269: "D"}
    START = 50265
    END = 50266
    TEXT = 199

    def test_the_leading_residue_run_is_the_sequence(self):
        ids = [50267, 50268, 50269, self.END, self.TEXT]
        self.assertEqual(PROBE.residue_run(ids, self.LETTERS), "ACD")

    def test_a_generation_that_leaves_the_residue_space_stops_there(self):
        ids = [50267, self.TEXT, 50268]
        self.assertEqual(PROBE.residue_run(ids, self.LETTERS), "A")

    def test_a_generation_that_never_enters_the_residue_space_is_empty(self):
        # The prompt ends at the declared protein start, so a tail that opens with
        # a non-residue id is a failure of the interface and not a sequence.
        self.assertEqual(PROBE.residue_run([self.START, 50267], self.LETTERS), "")
        self.assertEqual(PROBE.residue_run([self.TEXT, 50267], self.LETTERS), "")
        self.assertEqual(PROBE.residue_run([], self.LETTERS), "")

    def test_an_unterminated_run_is_still_the_sequence(self):
        self.assertEqual(PROBE.residue_run([50267, 50268], self.LETTERS), "AC")


class ThePromptIsTheReleasedFamilyRendering(unittest.TestCase):
    """The intervention is the request the released checkpoint was tuned on."""

    LABEL = "crossover junction endodeoxyribonuclease ruvc domain"

    def test_the_prompt_is_the_released_scaffold_then_the_protein_start(self):
        declaration = jm.rendering("instructprotein")
        self.assertEqual(
            PROBE.prompt_for(declaration, self.LABEL),
            "Instruction: I would like a protein that is in "
            "crossover junction endodeoxyribonuclease ruvc domain.\n\n"
            "Output: One of the protein that meets the demand is <protein>",
        )
        self.assertTrue(PROBE.prompt_for(declaration, self.LABEL).endswith(declaration.protein_start))

    def test_the_assistant_prefix_is_present_and_is_the_released_one(self):
        # The first run supplied the instruction clause and stopped at "Output:".
        prompt = PROBE.prompt_for(jm.rendering("instructprotein"), self.LABEL)
        self.assertIn(PROBE.ASSISTANT_PREFIX, prompt)
        self.assertEqual(
            PROBE.FAMILY_INSTRUCTION_TEMPLATE.format(label=self.LABEL),
            "Instruction: I would like a protein that is in "
            "crossover junction endodeoxyribonuclease ruvc domain.\n\n"
            "Output: One of the protein that meets the demand is",
        )

    def test_the_released_template_is_not_the_declared_generic_template(self):
        declaration = jm.rendering("instructprotein")
        template = declaration.protein_context_template
        assert template is not None, "this checkpoint declares a generic context template"
        self.assertNotEqual(
            PROBE.FAMILY_INSTRUCTION_TEMPLATE,
            template.format(context=PROBE.FAMILY_INSTRUCTION_SENTENCE),
            "the released family rendering carries a blank line and an assistant "
            "prefix; the generic declared one carries neither",
        )

    def test_the_instruction_is_the_published_one_and_only_the_label_moves(self):
        declaration = jm.rendering("instructprotein")
        left = PROBE.prompt_for(declaration, "family one")
        right = PROBE.prompt_for(declaration, "family two")
        self.assertNotEqual(left, right)
        self.assertEqual(
            left.replace("family one", "{label}"), right.replace("family two", "{label}")
        )


class TheLabelIsTheTunedNamingChannel(unittest.TestCase):
    """A lower-case prose name ending in the entry's own type noun."""

    def test_the_type_noun_is_appended_only_where_the_name_lacks_it(self):
        self.assertEqual(PROBE.tuned_label("RadC-like JAB domain", "Domain"), "radc-like jab domain")
        self.assertEqual(
            PROBE.tuned_label("SAP domain superfamily", "Homologous_superfamily"),
            "sap domain superfamily",
        )
        self.assertEqual(
            PROBE.tuned_label("Nucleotide-binding protein YajQ/Smlt4090-like", "Family"),
            "nucleotide-binding protein yajq/smlt4090-like family",
        )
        self.assertEqual(PROBE.tuned_label("Cystatin domain", "Active_site"), "cystatin domain active site")

    def test_an_entry_type_the_released_form_does_not_cover_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not one the released label form covers"):
            PROBE.tuned_label("Something", "PTM")


class TheSoftRead(unittest.TestCase):
    """A 5-mer comparison on real member sets, labelled as a soft read."""

    REQUESTED = ["ACDEFGHIKLMNPQRSTVWY" * 3 + "ACDEFGHIKL"] * 3
    MISMATCHED = ["YWVTSRQPNMLKIHGFEDCA" * 3 + "YWVTSRQPNM"] * 3

    def _read(self, samples_by_condition):
        groups = {role: np.arange(len(value)) for role, value in samples_by_condition.items()}
        return PROBE.soft_read(
            samples_by_condition,
            groups,
            {"requested": self.REQUESTED, "mismatched": self.MISMATCHED},
            requested="requested",
            mismatched="mismatched",
            seed=PROBE.PROBE_SEED,
            resamples=64,
            bootstrap_seed=PROBE.PROBE_SEED,
        )

    def test_a_generation_of_the_requested_family_is_nearer_the_requested_family(self):
        near = "ACDEFGHIKLMNPQRSTVWY" * 3 + "ACDEFGHIKM"
        read = self._read({"requested": [near] * 20, "mismatched": [near] * 20})
        self.assertEqual(read["per_condition"]["requested"]["nearer_requested_rate_grouped"], 1.0)
        self.assertGreater(read["per_condition"]["requested"]["mean_similarity_requested"],
                           read["per_condition"]["requested"]["mean_similarity_mismatched"])

    def test_a_generation_of_the_mismatched_family_is_not(self):
        far = "YWVTSRQPNMLKIHGFEDCA" * 3 + "YWVTSRQPNL"
        read = self._read({"requested": [far] * 20, "mismatched": [far] * 20})
        self.assertEqual(read["per_condition"]["requested"]["nearer_requested_rate_grouped"], 0.0)
        self.assertIsNotNone(read["per_condition"]["requested"]["zero_hit_upper_bound_95"])

    def test_an_empty_generation_is_nearer_neither_family_and_stays_in_the_denominator(self):
        read = self._read({"requested": [""] * 20, "mismatched": [""] * 20})
        self.assertEqual(read["per_condition"]["requested"]["n"], 20)
        self.assertEqual(read["per_condition"]["requested"]["n_nearer_requested"], 0)
        self.assertEqual(read["per_condition"]["requested"]["mean_similarity_requested"], 0.0)

    def test_the_two_pools_are_scored_at_the_same_size(self):
        groups = {"requested": np.arange(4), "mismatched": np.arange(4)}
        read = PROBE.soft_read(
            {"requested": ["ACDEFGHIKLMNPQRSTVWY" * 3] * 4, "mismatched": ["ACDEFGHIKLMNPQRSTVWY" * 3] * 4},
            groups,
            {"requested": self.REQUESTED, "mismatched": self.MISMATCHED + ["ACDEFGHIKL" * 9]},
            requested="requested",
            mismatched="mismatched",
            seed=PROBE.PROBE_SEED,
            resamples=64,
            bootstrap_seed=PROBE.PROBE_SEED,
        )
        self.assertEqual(read["member_pool_size_per_side"], len(self.REQUESTED))
        self.assertEqual(read["staged_records_per_side"]["mismatched"], len(self.MISMATCHED) + 1)

    def test_the_contrast_is_the_estimator_the_hard_read_uses(self):
        read = self._read({"requested": ["ACDEFGHIKLMNPQRSTVWY" * 3] * 20,
                           "mismatched": ["ACDEFGHIKLMNPQRSTVWY" * 3] * 20})
        self.assertIn("difference", read["contrast_nearer_requested"])
        self.assertIn("ci95", read["contrast_nearer_requested"])
        self.assertIn("difference_in_differences", read)


class TheClassDraw(unittest.TestCase):
    """The campaign's seeded permutation, not a prefix and not a hand-pick."""

    def test_eligibility_needs_both_enough_records_and_a_published_name(self):
        census = {
            "PF1": {f"a{index}" for index in range(200)},
            "PF2": {f"b{index}" for index in range(199)},
            "PF3": {f"c{index}" for index in range(500)},
        }
        labels = {"PF1": ("one", "family one"), "PF2": ("two", "family two")}
        self.assertEqual(
            PROBE.eligible_families(census, labels, minimum=cg.MIN_CLASS_RECORDS),
            ("PF1",),
            "a family below the record count, or one the release does not name, is "
            "not a class this probe can draw or instruct",
        )

    def test_the_draw_is_reproducible_and_the_two_classes_differ(self):
        candidates = tuple(f"PF{index:05d}" for index in range(200))
        first = PROBE.class_draw(candidates, seed=PROBE.PROBE_SEED, n=2)
        self.assertEqual(first, PROBE.class_draw(candidates, seed=PROBE.PROBE_SEED, n=2))
        self.assertEqual(len(set(first)), 2)
        self.assertNotEqual(first, PROBE.class_draw(candidates, seed=PROBE.PROBE_SEED + 1, n=2))

    def test_the_draw_is_not_the_head_of_the_candidate_list(self):
        candidates = tuple(f"PF{index:05d}" for index in range(200))
        self.assertNotEqual(PROBE.class_draw(candidates, seed=PROBE.PROBE_SEED, n=2), candidates[:2])

    def test_a_draw_larger_than_the_admissible_set_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "never topped up"):
            PROBE.class_draw(("PF1",), seed=PROBE.PROBE_SEED, n=2)


if __name__ == "__main__":
    unittest.main()
