"""CPU tests for the measurement-only MegaScale label-instrument qualification."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.stability.pairwise_stability import validate_cycle

SCRIPT = REPO / "scripts/capability/stability/qualify_megascale_label_instrument.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("qualify_megascale_label_instrument", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q = _load()
WT = "ACDEFGHIKL"


def _row(name: str, sequence: str, mut_type: str, background: str, cluster: str,
         *, y_t: float, y_c: float, width: float = 0.2, channel_width: float = 0.2,
         dg_ml: str | None = None, dna: str = "AAA", pair_name: float | str = float("nan"),
         ddg: str = "nan") -> dict:
    combined = (y_t + y_c) / 2.0
    row = {
        "name": name, "dna_seq": dna, "mut_type": mut_type, "WT_name": background,
        "WT_cluster": cluster, "aa_seq": sequence,
        "dG_ML": f"{combined:.6f}" if dg_ml is None else dg_ml,
        "ddG_ML": ddg, "pair_name": pair_name,
        "deltaG": combined, "deltaG_95CI": width,
        "deltaG_95CI_low": combined - width / 2, "deltaG_95CI_high": combined + width / 2,
    }
    for suffix, value in (("t", y_t), ("c", y_c)):
        row[f"deltaG_{suffix}"] = value
        row[f"deltaG_{suffix}_95CI"] = channel_width
        row[f"deltaG_{suffix}_95CI_low"] = value - channel_width / 2
        row[f"deltaG_{suffix}_95CI_high"] = value + channel_width / 2
    return row


def _mutate(sequence: str, *changes: tuple[int, str]) -> str:
    letters = list(sequence)
    for position, letter in changes:
        letters[position - 1] = letter
    return "".join(letters)


def _background(background: str, cluster: str, *, epsilon_t: float, epsilon_c: float,
                pairs: tuple[tuple[int, str, int, str], ...] = ((2, "W", 7, "Y"), (3, "M", 4, "V")),
                **kwargs: Any) -> list[dict]:
    rows = [_row(f"{background}_wt", WT, "wt", background, cluster, y_t=3.0, y_c=3.2, **kwargs)]
    singles = {}
    for i, a, j, b in pairs:
        for position, letter in ((i, a), (j, b)):
            if (position, letter) in singles:
                continue
            singles[(position, letter)] = (-0.4 * position / 10.0, -0.3 * position / 10.0)
            rows.append(_row(f"{background}_{letter}{position}",
                             _mutate(WT, (position, letter)),
                             f"{WT[position - 1]}{position}{letter}", background, cluster,
                             y_t=3.0 + singles[(position, letter)][0],
                             y_c=3.2 + singles[(position, letter)][1], **kwargs))
    for i, a, j, b in pairs:
        left, right = singles[(i, a)], singles[(j, b)]
        rows.append(_row(f"{background}_{a}{i}_{b}{j}", _mutate(WT, (i, a), (j, b)),
                         f"{WT[i - 1]}{i}{a}:{WT[j - 1]}{j}{b}", background, cluster,
                         y_t=3.0 + left[0] + right[0] + epsilon_t,
                         y_c=3.2 + left[1] + right[1] + epsilon_c, **kwargs))
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=Q.COLUMNS)


def _catalogue(*backgrounds: tuple[str, str, str]) -> dict:
    return {name: {"kind": kind, "sequence": WT, "cluster": cluster}
            for name, cluster, kind in backgrounds}


class AcceptedSupport(unittest.TestCase):
    def test_censored_strings_are_never_coerced_to_point_measurements(self):
        rows = _background("b1", "c1", epsilon_t=0.5, epsilon_c=0.5)
        for value in Q.CENSORED_DG_ML:
            rows.append(_row(f"x{value}", _mutate(WT, (9, "W")), "K9W", "b1", "c1",
                             y_t=-1.0, y_c=-1.0, dg_ml=value))
        mask, accounting = Q.accept(_frame(rows), per_channel_width=False)
        self.assertEqual(accounting["rows_censored_dG_ML"], {"<-1": 1, ">5": 1, "-": 1})
        self.assertEqual(int(mask.sum()), len(rows) - 3)

    def test_numeric_dG_ML_must_match_deltaG(self):
        rows = _background("b1", "c1", epsilon_t=0.0, epsilon_c=0.0)
        rows[0]["dG_ML"] = "2.0"
        with self.assertRaisesRegex(ValueError, "numeric dG_ML departs from deltaG"):
            Q.accept(_frame(rows), per_channel_width=False)

    def test_width_rules_apply_identically_to_both_channels(self):
        rows = _background("b1", "c1", epsilon_t=0.0, epsilon_c=0.0)
        wide_trypsin = _row("wt1", _mutate(WT, (9, "W")), "K9W", "b1", "c1",
                            y_t=1.0, y_c=1.0, channel_width=0.9)
        wide_chymotrypsin = dict(wide_trypsin)
        wide_chymotrypsin.update({"name": "wc1", "aa_seq": _mutate(WT, (10, "W")),
                                  "mut_type": "L10W", "deltaG_t_95CI": 0.2,
                                  "deltaG_c_95CI": 0.9})
        frame = _frame(rows + [wide_trypsin, wide_chymotrypsin])
        loose, _ = Q.accept(frame, per_channel_width=False)
        strict, accounting = Q.accept(frame, per_channel_width=True)
        self.assertEqual(int(loose.sum()) - int(strict.sum()), 2)
        self.assertEqual(accounting["rows_excluded_by_both_channel_width_rule"], 2)

    def test_inconsistent_confidence_width_is_fatal(self):
        rows = _background("b1", "c1", epsilon_t=0.0, epsilon_c=0.0)
        frame = _frame(rows)
        frame.loc[0, "deltaG_95CI_high"] = frame.loc[0, "deltaG_95CI_high"] + 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.parquet"
            frame.to_parquet(path)
            with self.assertRaisesRegex(ValueError, "deltaG_95CI differs"):
                Q.load_frame(Path(directory))

    def test_catalogue_sequence_must_match_the_wt_row(self):
        frame = _frame(_background("b1", "c1", epsilon_t=0.0, epsilon_c=0.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalogue.json"
            path.write_text(json.dumps([{"WT_name": "b1", "kind": "natural",
                                         "sequence": _mutate(WT, (1, "W"))}]))
            with self.assertRaisesRegex(ValueError, "differs from its wt row"):
                Q.load_catalogue(path, frame)


class CycleConstruction(unittest.TestCase):
    def setUp(self):
        self.rows = (_background("b1", "c1", epsilon_t=0.7, epsilon_c=0.5)
                     + _background("b2", "c1", epsilon_t=-0.3, epsilon_c=-0.2)
                     + _background("b3", "c2", epsilon_t=0.1, epsilon_c=0.4))
        self.frame = _frame(self.rows)
        self.catalogue = _catalogue(("b1", "c1", "natural"), ("b2", "c1", "natural"),
                                    ("b3", "c2", "natural"))

    def _cycles(self, statistic: str = "median"):
        mask, _ = Q.accept(self.frame, per_channel_width=True)
        values = Q.aggregate(self.frame, mask, statistic=statistic)
        return Q.build_cycles(values, self.catalogue)

    def test_planted_interaction_is_recovered_on_both_channels(self):
        cycles, _, accounting = self._cycles()
        self.assertEqual(accounting["backgrounds_with_cycles"], 3)
        self.assertEqual(len(cycles), 6)
        planted = {"b1": (0.7, 0.5), "b2": (-0.3, -0.2), "b3": (0.1, 0.4)}
        for name, part in cycles.groupby("WT_name"):
            np.testing.assert_allclose(part.trypsin, planted[name][0], atol=1e-9)
            np.testing.assert_allclose(part.chymotrypsin, planted[name][1], atol=1e-9)
        self.assertEqual(sorted(cycles.separation.unique().tolist()), [1, 5])

    def test_constructed_states_satisfy_the_shared_cycle_contract(self):
        _, states, _ = self._cycles()
        entry = states["b1"]
        for first, letter_a, second, letter_b, index in entry["doubles"]:
            single_a = _mutate(entry["wildtype"], (first + 1, letter_a))
            single_b = _mutate(entry["wildtype"], (second + 1, letter_b))
            positions = validate_cycle(entry["wildtype"], single_a, single_b,
                                       entry["sequences"][index])
            self.assertEqual(positions, (first, second))

    def test_a_background_without_an_accepted_wt_row_is_dropped_not_zero_filled(self):
        frame = self.frame[~((self.frame.WT_name == "b1") & (self.frame.mut_type == "wt"))]
        mask, _ = Q.accept(frame, per_channel_width=True)
        values = Q.aggregate(frame, mask, statistic="median")
        cycles, _, accounting = Q.build_cycles(values, self.catalogue)
        self.assertEqual(accounting["backgrounds_without_accepted_wt"], 1)
        self.assertNotIn("b1", set(cycles.WT_name))

    def test_repeated_rows_are_reduced_by_the_declared_statistic(self):
        outlier = dict(self.rows[0])
        outlier.update({"name": "b1_wt_second", "dna_seq": "CCC"})
        for suffix in ("t", "c"):
            outlier[f"deltaG_{suffix}"] = outlier[f"deltaG_{suffix}"] + 6.0
        frame = _frame(self.rows + [outlier])
        mask, _ = Q.accept(frame, per_channel_width=True)
        median = Q.aggregate(frame, mask, statistic="median")
        first = Q.aggregate(frame, mask, statistic="first")
        wt_median = median[(median.WT_name == "b1") & (median.aa_seq == WT)]
        wt_first = first[(first.WT_name == "b1") & (first.aa_seq == WT)]
        self.assertAlmostEqual(float(wt_median.deltaG_t.iloc[0]), 6.0)
        self.assertAlmostEqual(float(wt_first.deltaG_t.iloc[0]), 3.0)
        self.assertEqual(int(wt_median.n_rows.iloc[0]), 2)
        self.assertEqual(int(wt_median.n_dna.iloc[0]), 2)


class AgreementEstimates(unittest.TestCase):
    def test_resampling_units_are_groups_not_cycles(self):
        rows = (_background("b1", "c1", epsilon_t=0.7, epsilon_c=0.5)
                + _background("b2", "c1", epsilon_t=-0.3, epsilon_c=-0.2)
                + _background("b3", "c2", epsilon_t=0.1, epsilon_c=0.4))
        frame = _frame(rows)
        mask, _ = Q.accept(frame, per_channel_width=True)
        cycles, _, _ = Q.build_cycles(Q.aggregate(frame, mask, statistic="median"),
                                      _catalogue(("b1", "c1", "natural"), ("b2", "c1", "natural"),
                                                 ("b3", "c2", "natural")))
        self.assertEqual(len(cycles), 6)
        _, _, by_background = Q._units(cycles, "background")
        _, _, by_cluster = Q._units(cycles, "source_cluster")
        self.assertEqual(by_background["units"], 3)
        self.assertEqual(by_cluster["units"], 2)
        self.assertLess(by_cluster["effective_units_kish"], by_background["effective_units_kish"])

    def test_identical_channels_give_unit_agreement_and_zero_discordance(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=500)
        derived = Q._derive(Q._moments(values, values.copy()))
        self.assertAlmostEqual(derived["pearson_r"], 1.0, places=9)
        self.assertAlmostEqual(derived["per_channel_discordance_sd_kcal_mol"], 0.0, places=9)
        self.assertAlmostEqual(derived["shared_component_sd_kcal_mol"],
                               derived["sd_epsilon_trypsin_kcal_mol"], places=9)

    def test_shared_and_discordant_scales_separate_signal_from_channel_noise(self):
        rng = np.random.default_rng(1)
        shared = rng.normal(scale=0.5, size=200000)
        derived = Q._derive(Q._moments(shared + rng.normal(scale=0.2, size=shared.size),
                                       shared + rng.normal(scale=0.2, size=shared.size)))
        self.assertAlmostEqual(derived["shared_component_sd_kcal_mol"], 0.5, places=2)
        self.assertAlmostEqual(derived["per_channel_discordance_sd_kcal_mol"], 0.2, places=2)

    def test_weighted_quantile_follows_the_weights(self):
        values = np.array([0.0, 1.0, 2.0])
        self.assertEqual(Q._weighted_quantile(values, np.array([0.98, 0.01, 0.01]), 0.5), 0.0)
        self.assertEqual(Q._weighted_quantile(values, np.array([0.01, 0.01, 0.98]), 0.5), 2.0)
        self.assertEqual(Q._weighted_quantile(values, np.array([1.0, 1.0, 1.0]), 0.95), 2.0)


class PositiveControl(unittest.TestCase):
    def test_pair_table_numbering_must_match_the_wt_sequence(self):
        rows = _background("b1", "c1", epsilon_t=0.5, epsilon_c=0.5)
        rows[-1]["pair_name"] = "b1.pdb_hnet0"
        rows[-1]["mut_type"] = "Z3M:V4V"
        frame = _frame(rows)
        with self.assertRaisesRegex(ValueError, "pair-table numbering"):
            Q.documented_pairs(frame, _catalogue(("b1", "c1", "natural")))

    def test_documented_pairs_are_selected_only_from_the_source_column(self):
        rows = _background("b1", "c1", epsilon_t=0.5, epsilon_c=0.5)
        rows[-1]["pair_name"] = "b1.pdb_hnet0"
        pairs = Q.documented_pairs(_frame(rows), _catalogue(("b1", "c1", "natural")))
        self.assertEqual(pairs, {("b1", 3, 4): "hnet"})

    def test_additive_surface_residual_and_cycle_are_distinct_estimands(self):
        letters = "WYFMV"
        rows = [_row("b1_wt", WT, "wt", "b1", "c1", y_t=3.0, y_c=3.0)]
        effect = {letter: -0.1 * (index + 1) for index, letter in enumerate(letters)}
        for letter in letters:
            for position in (2, 7):
                rows.append(_row(f"b1_{letter}{position}", _mutate(WT, (position, letter)),
                                 f"{WT[position - 1]}{position}{letter}", "b1", "c1",
                                 y_t=3.0 + effect[letter], y_c=3.0 + effect[letter]))
        for a in letters:
            for b in letters:
                rows.append(_row(f"b1_{a}2_{b}7", _mutate(WT, (2, a), (7, b)),
                                 f"{WT[1]}2{a}:{WT[6]}7{b}", "b1", "c1",
                                 y_t=3.0 + effect[a] + effect[b] + (0.8 if (a, b) == ("W", "Y") else 0.0),
                                 y_c=3.0 + effect[a] + effect[b] + (0.8 if (a, b) == ("W", "Y") else 0.0)))
        frame = _frame(rows)
        mask, _ = Q.accept(frame, per_channel_width=True)
        _, states, _ = Q.build_cycles(Q.aggregate(frame, mask, statistic="median"),
                                      _catalogue(("b1", "c1", "natural")))
        fit = Q.additive_surface_residual(states["b1"], 2, 7, "trypsin")
        self.assertEqual(fit["cycle"].size, len(letters) ** 2)
        self.assertAlmostEqual(float(fit["cycle"].max()), 0.8, places=9)
        self.assertLess(float(fit["residual"].max()), 0.8)
        self.assertGreater(float(np.sqrt(np.mean((fit["residual"] - fit["cycle"]) ** 2))), 0.02)

    def test_a_double_without_both_singles_enters_neither_estimand(self):
        letters = "WYFMV"
        rows = [_row("b1_wt", WT, "wt", "b1", "c1", y_t=3.0, y_c=3.0)]
        for letter in letters:
            for position in (2, 7):
                if (letter, position) == ("V", 7):
                    continue
                rows.append(_row(f"b1_{letter}{position}", _mutate(WT, (position, letter)),
                                 f"{WT[position - 1]}{position}{letter}", "b1", "c1",
                                 y_t=2.8, y_c=2.8))
        for a in letters:
            for b in letters:
                rows.append(_row(f"b1_{a}2_{b}7", _mutate(WT, (2, a), (7, b)),
                                 f"{WT[1]}2{a}:{WT[6]}7{b}", "b1", "c1", y_t=2.6, y_c=2.6))
        frame = _frame(rows)
        mask, _ = Q.accept(frame, per_channel_width=True)
        cycles, states, _ = Q.build_cycles(Q.aggregate(frame, mask, statistic="median"),
                                           _catalogue(("b1", "c1", "natural")))
        fit = Q.additive_surface_residual(states["b1"], 2, 7, "trypsin")
        self.assertEqual(fit["cycle"].size, len(letters) * (len(letters) - 1))
        self.assertEqual(fit["cycle"].size, int((cycles.separation == 5).sum()))

    def test_a_pair_table_below_four_doubles_is_refused_not_extrapolated(self):
        rows = _background("b1", "c1", epsilon_t=0.5, epsilon_c=0.5)
        frame = _frame(rows)
        mask, _ = Q.accept(frame, per_channel_width=True)
        _, states, _ = Q.build_cycles(Q.aggregate(frame, mask, statistic="median"),
                                      _catalogue(("b1", "c1", "natural")))
        self.assertIsNone(Q.additive_surface_residual(states["b1"], 2, 7, "trypsin"))

    def test_both_channels_share_one_pair_table_support(self):
        letters = "WYFMV"
        rows = [_row("b1_wt", WT, "wt", "b1", "c1", y_t=3.0, y_c=3.0)]
        for letter in letters:
            for position in (2, 7):
                rows.append(_row(f"b1_{letter}{position}", _mutate(WT, (position, letter)),
                                 f"{WT[position - 1]}{position}{letter}", "b1", "c1",
                                 y_t=2.8, y_c=2.9))
        for a in letters:
            for b in letters:
                rows.append(_row(f"b1_{a}2_{b}7", _mutate(WT, (2, a), (7, b)),
                                 f"{WT[1]}2{a}:{WT[6]}7{b}", "b1", "c1",
                                 y_t=2.6, y_c=2.7, pair_name="b1.pdb_hnet0"))
        frame = _frame(rows)
        mask, _ = Q.accept(frame, per_channel_width=True)
        catalogue = _catalogue(("b1", "c1", "natural"))
        _, states, _ = Q.build_cycles(Q.aggregate(frame, mask, statistic="median"), catalogue)
        pairs = Q.documented_pairs(frame, catalogue)
        tables = Q.pair_table_estimands(states, pairs, catalogue)
        self.assertEqual(set(pairs), {("b1", 2, 7)})
        self.assertEqual(len(tables), len(letters) ** 2)
        self.assertEqual(list(tables.separation.unique()), [5])
        np.testing.assert_allclose(tables.trypsin, 0.0, atol=1e-9)
        np.testing.assert_allclose(tables.cycle_trypsin, 0.0, atol=1e-9)


class GlobalResponse(unittest.TestCase):
    def test_a_response_curve_in_the_additive_prediction_is_absorbed(self):
        rng = np.random.default_rng(3)
        size = 20000
        prediction = rng.uniform(-1.0, 4.0, size)
        curve = 0.6 * np.exp(-prediction)
        cycles = pd.DataFrame({
            "WT_name": ["b1"] * size, "cluster": ["c1"] * size, "kind": ["natural"] * size,
            "site_i": 1, "site_j": 12, "separation": 11,
            "additive_trypsin": prediction, "additive_chymotrypsin": prediction,
            "trypsin": curve + rng.normal(scale=0.05, size=size),
            "chymotrypsin": curve + rng.normal(scale=0.05, size=size)})
        adjusted, report = Q.global_response_adjusted(cycles)
        self.assertGreater(report["channels"]["trypsin"]["absorbed_variance_fraction"], 0.9)
        self.assertLess(adjusted.trypsin.std(), cycles.trypsin.std())
        before = Q._derive(Q._moments(cycles.trypsin.to_numpy(), cycles.chymotrypsin.to_numpy()))
        after = Q._derive(Q._moments(adjusted.trypsin.to_numpy(), adjusted.chymotrypsin.to_numpy()))
        self.assertGreater(before["pearson_r"], 0.9)
        # A twenty-bin step response underfits a steep curve, so the adjusted
        # agreement stays above zero; it is an upper bound, not a removal proof.
        self.assertLess(after["pearson_r"], before["pearson_r"] / 2.0)
        self.assertGreater(Q.global_response_adjusted(cycles, bins=200)[1]['channels']['trypsin']
                           ['absorbed_variance_fraction'],
                           report['channels']['trypsin']['absorbed_variance_fraction'])

    def test_epsilon_independent_of_the_prediction_survives_adjustment(self):
        rng = np.random.default_rng(4)
        size = 20000
        shared = rng.normal(scale=0.4, size=size)
        cycles = pd.DataFrame({
            "WT_name": ["b1"] * size, "cluster": ["c1"] * size, "kind": ["natural"] * size,
            "site_i": 1, "site_j": 12, "separation": 11,
            "additive_trypsin": rng.uniform(-1.0, 4.0, size),
            "additive_chymotrypsin": rng.uniform(-1.0, 4.0, size),
            "trypsin": shared + rng.normal(scale=0.1, size=size),
            "chymotrypsin": shared + rng.normal(scale=0.1, size=size)})
        adjusted, report = Q.global_response_adjusted(cycles)
        self.assertLess(report["channels"]["trypsin"]["absorbed_variance_fraction"], 0.01)
        after = Q._derive(Q._moments(adjusted.trypsin.to_numpy(), adjusted.chymotrypsin.to_numpy()))
        self.assertGreater(after["pearson_r"], 0.9)


class SignConvention(unittest.TestCase):
    def test_sign_convention_is_read_from_the_source_ddG_column(self):
        rows = [_row("b1_wt", WT, "wt", "b1", "c1", y_t=3.0, y_c=3.0)]
        rows.append(_row("b1_W2", _mutate(WT, (2, "W")), "C2W", "b1", "c1",
                         y_t=2.0, y_c=2.0, ddg="-1.0"))
        frame = _frame(rows)
        mask, _ = Q.accept(frame, per_channel_width=True)
        report = Q.sign_and_scale(frame, mask)
        self.assertEqual(report["single_mutant_rows"], 1)
        self.assertAlmostEqual(report["rms_residual_against_mutant_minus_wt_kcal_mol"], 0.0, places=9)
        self.assertAlmostEqual(report["rms_residual_against_wt_minus_mutant_kcal_mol"], 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
