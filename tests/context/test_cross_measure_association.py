"""EXP-R2-247: refuse invented cells; withhold coefficients while pending."""

from __future__ import annotations

import unittest

from src.capability.context.cross_measure_association import (
    CELLS,
    Cell,
    associate,
    associate_pair,
    refuse_missing,
    spearman_rank,
)


class RefuseInvented(unittest.TestCase):
    def test_unavailable_cell_is_not_filled_in(self):
        self.assertEqual(CELLS["protgpt2"]["G1"].status, "unavailable")
        with self.assertRaisesRegex(ValueError, "refusing to invent"):
            refuse_missing("G1", "protgpt2")

    def test_pending_pair_has_no_coefficient(self):
        table = {
            "progen3-3b": {
                "G1": Cell("available", 1.5),
                "U1": Cell("available", 0.6),
            },
            "protgpt2": {
                "G1": Cell("unavailable"),
                "U1": Cell("pending"),
            },
        }
        record = associate_pair(
            "P-G1-U1",
            "G1",
            "U1",
            ("progen3-3b", "protgpt2"),
            table=table,
        )
        self.assertIn("protgpt2", record["pending"])
        self.assertFalse(record["computed"])
        self.assertIsNone(record["spearman"])


class RankRules(unittest.TestCase):
    def test_two_points_are_plus_or_minus_one(self):
        self.assertEqual(spearman_rank([1.0, 2.0], [10.0, 20.0]), 1.0)
        self.assertEqual(spearman_rank([1.0, 2.0], [20.0, 10.0]), -1.0)
        self.assertIsNone(spearman_rank([1.0, 1.0], [10.0, 20.0]))

    def test_injected_complete_table_computes_and_does_not_rank_models(self):
        table = {
            "a": {"G1": Cell("available", 1.0), "S1": Cell("available", 0.2)},
            "b": {"G1": Cell("available", 2.0), "S1": Cell("available", 0.4)},
            "c": {"G1": Cell("available", 3.0), "S1": Cell("available", 0.1)},
            "d": {"G1": Cell("available", 4.0), "S1": Cell("available", 0.8)},
            "e": {"G1": Cell("available", 5.0), "S1": Cell("available", 0.9)},
        }
        payload = associate(
            cells=table,
            pairs=(("P-G1-S1", "G1", "S1"),),
            arms=tuple(table),
        )
        self.assertTrue(payload["no_parameter_count_causal_claim"])
        self.assertTrue(payload["descriptive_not_causal"])
        row = payload["pooled"][0]
        self.assertTrue(row["computed"])
        self.assertEqual(row["n"], 5)
        self.assertIsNotNone(row["spearman"])


if __name__ == "__main__":
    unittest.main()
