"""Corpus draws preserve declared seeds, disjoint windows and parent identity."""
import ast
import unittest
from pathlib import Path


def test_live_corpus_calls_declare_a_seed():
    root = Path(__file__).resolve().parents[2]
    missing = []
    for base in (root / "scripts/capability", root / "src/capability"):
        for path in base.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                if name in {"protein_cohort", "text_cohort"}:
                    if not any(k.arg == "seed" or k.arg is None for k in node.keywords):
                        missing.append((str(path.relative_to(root)), node.lineno))
    assert not missing, missing


class ASubsampleCarriesItsParentsDraw(unittest.TestCase):
    """A seeded subsample of a file-order prefix is still a file-order prefix."""

    def _cohort(self, seed):
        from src.capability.core.arms import Cohort, sampling_record

        return Cohort(
            name="pool",
            kind="protein",
            records=[f"AAAA{index}" for index in range(20)],
            min_symbols=1,
            max_symbols=10,
            metadata={
                "sampling": sampling_record(
                    seed=seed, skip=0, requested=20, eligible=100, corpus="plain_swissprot"
                )
            },
        )

    def test_a_file_order_parent_keeps_its_hazard_through_the_subsample(self):
        from src.capability.core.arms import FILE_ORDER_HAZARD
        from src.capability.core.measurement_support import subsample_cohort

        child = subsample_cohort(self._cohort(seed=None), 5, 7)
        self.assertEqual(child.sampling["mode"], "file_order")
        self.assertEqual(child.sampling["hazard"], FILE_ORDER_HAZARD)
        self.assertEqual(child.sampling["subsample_seed"], 7)

    def test_a_seeded_parent_is_recorded_as_seeded(self):
        from src.capability.core.measurement_support import subsample_cohort

        child = subsample_cohort(self._cohort(seed=11), 5, 7)
        self.assertEqual(child.sampling["mode"], "seeded_permutation")
        self.assertEqual(child.sampling["seed"], 11)
        self.assertNotIn("hazard", child.sampling)

    def test_the_parent_is_identified_so_the_draw_can_be_reproduced(self):
        from src.capability.core.measurement_support import subsample_cohort

        parent = self._cohort(seed=11)
        child = subsample_cohort(parent, 5, 7)
        self.assertEqual(child.sampling["subsample_parent_digest"], parent.digest)
        self.assertEqual(child.sampling["subsample_parent_size"], len(parent))
        self.assertEqual(child.sampling["subsample_size"], 5)

class ASeededSkipIsDisjoint(unittest.TestCase):
    """--cohort-skip has to index a disjoint window, or it is not a sensitivity."""

    def test_two_skips_at_one_seed_share_no_record(self):
        from src.capability.core.arms import selected_positions

        first = selected_positions(5000, n=400, skip=0, seed=20260728, label="a")
        second = selected_positions(5000, n=400, skip=400, seed=20260728, label="b")
        self.assertEqual(set(first) & set(second), set())

    def test_a_seeded_window_is_not_the_file_order_window(self):
        from src.capability.core.arms import selected_positions

        seeded = selected_positions(5000, n=400, skip=0, seed=20260728, label="a")
        self.assertNotEqual(seeded, list(range(400)))
