"""The reward functions against hand-computed values, and their refusals.

A reward is the one thing in a reinforcement-learning experiment that cannot be
checked by watching the run: a wrong reward produces a perfectly healthy
trajectory toward the wrong place. So every term is checked against a value
computed by hand from the declared constants, and every refusal the declaration
promises is checked to happen.
"""

from __future__ import annotations

import math

import pytest

from src.capability.extensions import sequence_reward as sr


def _protonated(ph: float, pka: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def _deprotonated(ph: float, pka: float) -> float:
    return 1.0 - _protonated(ph, pka)


class TestNetCharge:
    def test_single_glycine_is_the_two_termini_alone(self):
        """Glycine has no ionizable side chain, so one G is the termini."""

        expected = (_protonated(7.0, sr.PKA_N_TERMINUS)
                    - _deprotonated(7.0, sr.PKA_C_TERMINUS))
        assert sr.net_charge("G") == pytest.approx(expected, abs=1e-12)
        # and the termini are counted once per chain, not once per residue
        assert sr.net_charge("GGGGG") == pytest.approx(expected, abs=1e-12)

    def test_lysine_run_adds_one_protonated_side_chain_each(self):
        termini = (_protonated(7.0, sr.PKA_N_TERMINUS)
                   - _deprotonated(7.0, sr.PKA_C_TERMINUS))
        side = _protonated(7.0, sr.PKA_POSITIVE["K"])
        assert sr.net_charge("KKKK") == pytest.approx(termini + 4 * side, abs=1e-12)

    def test_histidine_is_partly_protonated_at_ph_seven(self):
        """H's pKa of 6.5 puts it near a quarter charged, which is why it is not a count."""

        share = _protonated(7.0, sr.PKA_POSITIVE["H"])
        assert 0.2 < share < 0.3
        termini = (_protonated(7.0, sr.PKA_N_TERMINUS)
                   - _deprotonated(7.0, sr.PKA_C_TERMINUS))
        assert sr.net_charge("H") == pytest.approx(termini + share, abs=1e-12)

    def test_a_hand_computed_mixed_sequence(self):
        sequence = "MKKKRRDEEAAGGLLVVTTSSYY"
        expected = (
            _protonated(7.0, sr.PKA_N_TERMINUS)
            - _deprotonated(7.0, sr.PKA_C_TERMINUS)
            + 3 * _protonated(7.0, sr.PKA_POSITIVE["K"])
            + 2 * _protonated(7.0, sr.PKA_POSITIVE["R"])
            - 1 * _deprotonated(7.0, sr.PKA_NEGATIVE["D"])
            - 2 * _deprotonated(7.0, sr.PKA_NEGATIVE["E"])
            - 2 * _deprotonated(7.0, sr.PKA_NEGATIVE["Y"])
        )
        assert sr.net_charge(sequence) == pytest.approx(expected, abs=1e-12)

    def test_charge_is_monotone_in_added_lysine_and_glutamate(self):
        base = sr.net_charge("GGGGGGGGGG")
        assert sr.net_charge("GGGGGGGGGGK") > base
        assert sr.net_charge("GGGGGGGGGGE") < base

    def test_an_empty_sequence_has_no_termini(self):
        """Not the +0.976 an amino terminus would contribute to a chain that exists."""

        assert sr.net_charge("") == 0.0

    def test_a_non_canonical_character_carries_no_side_chain(self):
        assert sr.net_charge("GXG") == pytest.approx(sr.net_charge("GGG"), abs=1e-12)

    def test_the_declared_ph_is_seven(self):
        assert sr.REFERENCE_PH == 7.0
        # and the function is continuous in pH rather than a count of residues
        assert sr.net_charge("H", ph=4.0) > sr.net_charge("H", ph=9.0)


class TestLowComplexity:
    def test_a_homopolymer_is_wholly_low_complexity(self):
        sequence = "A" * 40
        assert sr.composition_entropy_bits(sequence) == 0.0
        assert sr.low_complexity_fraction(sequence) == 1.0

    def test_a_short_homopolymer_is_read_as_one_window(self):
        """Shorter than a window, so it must not escape the mask for being short."""

        assert len("AAAA") < sr.LOW_COMPLEXITY_WINDOW
        assert sr.low_complexity_fraction("AAAA") == 1.0

    def test_twelve_distinct_residues_are_not_low_complexity(self):
        window = "ACDEFGHIKLMN"
        assert len(window) == sr.LOW_COMPLEXITY_WINDOW
        assert sr.composition_entropy_bits(window) == pytest.approx(math.log2(12))
        assert sr.low_complexity_fraction(window) == 0.0

    def test_the_threshold_is_where_the_declaration_puts_it(self):
        """Four residues in equal proportion carry 2.0 bits, inside the 2.2-bit limit."""

        window = "ACDE" * 3
        assert sr.composition_entropy_bits(window) == pytest.approx(2.0)
        assert 2.0 <= sr.LOW_COMPLEXITY_ENTROPY_BITS
        assert sr.low_complexity_fraction(window) == 1.0

    def test_a_labelled_window_covers_every_residue_it_contains(self):
        """The union rule: one low-entropy window labels its whole span."""

        sequence = "ACDEFGHIKLMNPQRSTVWY" + "A" * sr.LOW_COMPLEXITY_WINDOW
        fraction = sr.low_complexity_fraction(sequence)
        assert 0.0 < fraction < 1.0
        # the run itself plus the window positions that reach back over it
        assert fraction * len(sequence) >= sr.LOW_COMPLEXITY_WINDOW

    def test_empty_sequence_has_no_residues_to_label(self):
        assert sr.low_complexity_fraction("") == 0.0

    def test_a_zero_window_is_refused(self):
        with pytest.raises(ValueError, match="at least one residue"):
            sr.low_complexity_fraction("ACDEFG", window=0)

    def test_the_limit_is_the_published_quarter(self):
        assert sr.LOW_COMPLEXITY_LIMIT == 0.25


class TestDiagnostics:
    def test_longest_single_residue_run(self):
        assert sr.longest_single_residue_run("") == 0
        assert sr.longest_single_residue_run("ACDE") == 1
        assert sr.longest_single_residue_run("ACCCDE") == 3
        assert sr.longest_single_residue_run("AAACCCCAA") == 4

    def test_composition_entropy_of_a_two_residue_alphabet(self):
        assert sr.composition_entropy_bits("ACACAC") == pytest.approx(1.0)
        assert sr.composition_entropy_bits("") == 0.0

    def test_statistics_counts_non_canonical_residues(self):
        stats = sr.statistics("ACDXZ")
        assert stats.length == 5
        assert stats.non_canonical_residues == 2


class TestPlateau:
    def test_one_inside_the_window(self):
        assert sr.plateau(5.0, low=4.0, high=6.0, half_width=2.0) == 1.0
        assert sr.plateau(4.0, low=4.0, high=6.0, half_width=2.0) == 1.0
        assert sr.plateau(6.0, low=4.0, high=6.0, half_width=2.0) == 1.0

    def test_linear_decay_to_zero_over_the_half_width(self):
        assert sr.plateau(7.0, low=4.0, high=6.0, half_width=2.0) == pytest.approx(0.5)
        assert sr.plateau(8.0, low=4.0, high=6.0, half_width=2.0) == 0.0
        assert sr.plateau(99.0, low=4.0, high=6.0, half_width=2.0) == 0.0
        assert sr.plateau(3.0, low=4.0, high=6.0, half_width=2.0) == pytest.approx(0.5)

    def test_a_hard_window_is_refused_because_it_gives_no_gradient(self):
        with pytest.raises(ValueError, match="degenerate group"):
            sr.plateau(5.0, low=4.0, high=6.0, half_width=0.0)

    def test_an_unordered_window_is_refused(self):
        with pytest.raises(ValueError, match="ordered"):
            sr.plateau(5.0, low=6.0, high=4.0, half_width=1.0)

    def test_a_non_finite_value_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            sr.plateau(float("nan"), low=0.0, high=1.0, half_width=1.0)


class TestRewards:
    def _in_window(self) -> sr.SequenceStatistics:
        """A sequence inside every declared window, built from the declaration.

        Not a literal: the windows are the thing under test, so a fixture that
        hard-coded one would stop testing the declaration the moment it moved.
        """

        low_length, high_length = sr.LENGTH_WINDOW
        low_charge, high_charge = sr.CHARGE_WINDOW
        length = (low_length + high_length) // 2
        # A varied backbone carrying no lysine, into which lysines are placed at
        # evenly spaced positions. Spaced rather than clustered on purpose: a
        # leading run of lysines would trip the shared low-complexity term, which
        # is what that term is for, so a fixture that has to satisfy every window
        # at once has to spread its charge.
        backbone = ("ACDEFGHILMNPQSTVWY" * 20)[:length]
        for count in range(1, length):
            candidate = list(backbone)
            for index in range(count):
                candidate[(index * length) // count] = "K"
            stats = sr.statistics("".join(candidate))
            if (low_charge <= stats.net_charge <= high_charge
                    and stats.low_complexity_fraction < sr.LOW_COMPLEXITY_LIMIT):
                return stats
        raise AssertionError("no candidate satisfied every declared window at once")

    def test_a_sequence_in_every_window_scores_one_on_the_environment_reward(self):
        stats = self._in_window()
        low, high = sr.CHARGE_WINDOW
        assert low <= stats.net_charge <= high
        assert sr.LENGTH_WINDOW[0] <= stats.length <= sr.LENGTH_WINDOW[1]
        assert stats.low_complexity_fraction < sr.LOW_COMPLEXITY_LIMIT
        assert sr.reward(sr.REWARD_ENVIRONMENT, stats) == pytest.approx(1.0)

    def test_every_reward_is_bounded_in_the_unit_interval(self):
        stats = self._in_window()
        for name, value in sr.rewards(stats, mean_ca_plddt=100.0).items():
            assert 0.0 <= value <= 1.0, name
        assert sr.reward(sr.REWARD_MEMORY, stats, mean_ca_plddt=100.0) == pytest.approx(1.0)
        # the positive control's reward is bounded alike at any window it could take
        for window in (sr.CHARGE_WINDOW, (32.0, 36.0)):
            assert 0.0 <= sr.environment_reward_at(stats, charge_window=window) <= 1.0

    def test_the_memory_reward_is_the_complexity_term_times_the_plddt_share(self):
        stats = sr.statistics("ACDEFGHIKLMNPQRSTVWY" * 4)
        expected = sr.complexity_term(stats) * (62.5 / sr.PLDDT_SCALE)
        assert sr.reward(sr.REWARD_MEMORY, stats, mean_ca_plddt=62.5) == pytest.approx(expected)

    def test_the_joint_reward_is_the_equal_weight_mean_of_the_two(self):
        stats = sr.statistics("ACDEFGHIKLMNPQRSTVWY" * 4)
        memory = sr.reward(sr.REWARD_MEMORY, stats, mean_ca_plddt=55.0)
        environment = sr.reward(sr.REWARD_ENVIRONMENT, stats)
        both = sr.reward(sr.REWARD_BOTH, stats, mean_ca_plddt=55.0)
        assert both == pytest.approx(0.5 * (memory + environment))

    def test_the_complexity_term_is_shared_by_every_reward(self):
        """A poly-lysine solution to the charge window is forbidden, not rewarded.

        Including at the positive control's harder window, which poly-lysine would
        otherwise reach even more easily than the declared one.
        """

        poly = sr.statistics("K" * 90)
        low, high = sr.CHARGE_WINDOW
        assert poly.net_charge > high  # it overshoots the window as well
        assert poly.low_complexity_fraction == 1.0
        assert sr.complexity_term(poly) == 0.0
        for name, value in sr.rewards(poly, mean_ca_plddt=90.0).items():
            assert value == 0.0, name
        for window in sr.CHARGE_LADDER if hasattr(sr, "CHARGE_LADDER") else ((32.0, 36.0),):
            assert sr.environment_reward_at(poly, charge_window=window) == 0.0

    def test_a_structure_reward_with_no_structure_is_refused_not_paid_zero(self):
        stats = sr.statistics("ACDEFGHIKLMNPQRSTVWY")
        with pytest.raises(ValueError, match="refusing to score it as zero"):
            sr.reward(sr.REWARD_MEMORY, stats)
        with pytest.raises(ValueError, match="refusing to score it as zero"):
            sr.reward(sr.REWARD_BOTH, stats)
        # the environment reward needs none and is paid
        assert 0.0 <= sr.reward(sr.REWARD_ENVIRONMENT, stats) <= 1.0

    def test_an_empty_completion_is_paid_the_declared_value_without_a_fold(self):
        stats = sr.statistics("")
        for name in sr.REWARD_NAMES:
            assert sr.reward(name, stats) == sr.EMPTY_COMPLETION_REWARD

    def test_an_unknown_reward_is_refused(self):
        with pytest.raises(KeyError, match="unknown reward"):
            sr.reward("stability", sr.statistics("ACDE"))

    def test_a_plddt_off_its_declared_scale_is_refused(self):
        with pytest.raises(ValueError, match="outside the 0-100 scale"):
            sr.plddt_term(-1.0)
        with pytest.raises(ValueError, match="outside the 0-100 scale"):
            sr.plddt_term(101.0)
        with pytest.raises(ValueError, match="finite"):
            sr.plddt_term(float("nan"))

    def test_the_zero_to_one_convention_cannot_be_detected_and_is_not_claimed(self):
        """A caller that passed the 0-1 scale would be silently wrong by a factor of 100.

        The guard cannot catch it: a mean CA-pLDDT of 0.62 on the 0-100 scale is
        a legitimate reading for a sequence with no structure at all. The project's
        own ``summarize_esmfold2`` is what rescales, and it validates the raw
        array against [0, 1] before multiplying, so the only caller is already
        on the right scale. This records the limitation rather than pretending
        the term defends against it.
        """

        assert sr.plddt_term(0.62) == pytest.approx(0.0062)

    def test_rewards_omits_the_structure_rewards_it_cannot_compute(self):
        stats = sr.statistics("ACDEFGHIKLMNPQRSTVWY")
        computable = {name for name in sr.REWARD_NAMES
                      if not (sr.REWARD_NEEDS_DECLARED_WINDOW[name]
                              and not sr.hard_window_declared())}
        without = sr.rewards(stats)
        assert set(without) == computable - {name for name in sr.REWARD_NAMES
                                             if sr.REWARD_NEEDS_STRUCTURE[name]}
        with_fold = sr.rewards(stats, mean_ca_plddt=50.0)
        assert set(with_fold) == computable

    def test_the_declared_objective_hits_are_the_windows_themselves(self):
        stats = self._in_window()
        hits = sr.meets_declared_objective(stats, mean_ca_plddt=80.0)
        assert hits["charge_window"] and hits["length_window"]
        assert hits["low_complexity_ok"] and hits["environment_conjunction"]
        assert hits["plddt_criterion"] and hits["memory_conjunction"]
        assert sr.meets_declared_objective(stats, mean_ca_plddt=60.0)["plddt_criterion"] is False


class TestStructuralSupport:
    """Where a structural reward is defined, and why it stops short of a peptide.

    Measured on the staged instrument: mean CA-pLDDT 85.3 at two residues and
    87.8 at four, against about 47 on the base pool's own samples, and an
    IndexError at one. A structural reward paid there would let a policy outscore
    every natural-length sequence by emitting peptides, and the gain would be the
    instrument describing its input rather than a bias anyone installed.
    """

    def test_it_is_the_project_s_own_declared_structural_support(self):
        from src.capability.generation import unconditional_generation as ug

        assert list(sr.STRUCTURE_SUPPORT) == list(ug.POLICY["structure_support"])

    def test_a_structural_reward_outside_the_support_is_the_empty_value(self):
        for length in (1, 2, 4, 8, sr.STRUCTURE_SUPPORT[0] - 1):
            stats = sr.statistics(("ACDEFGHIKLMNPQRSTVWY" * 2)[:length])
            assert stats.length == length
            assert sr.within_structure_support(stats) is False
            for name in (sr.REWARD_MEMORY, sr.REWARD_BOTH):
                assert sr.reward(name, stats, mean_ca_plddt=95.0) == \
                    sr.EMPTY_COMPLETION_REWARD, (name, length)

    def test_the_closed_form_rewards_are_unaffected_by_it(self):
        """The environment arm has no structural term, so its support is the sequence."""

        stats = sr.statistics("ACDEFGHI")
        assert sr.within_structure_support(stats) is False
        assert sr.reward(sr.REWARD_ENVIRONMENT, stats) > 0.0

    def test_inside_the_support_a_structural_reward_is_paid(self):
        low = sr.STRUCTURE_SUPPORT[0]
        stats = sr.statistics(("ACDEFGHIKLMNPQRSTVWY" * 2)[:low])
        assert sr.within_structure_support(stats) is True
        assert sr.reward(sr.REWARD_MEMORY, stats, mean_ca_plddt=80.0) > 0.0

    def test_inside_the_support_a_missing_fold_is_still_refused(self):
        """The refusal is kept for the case it was written for."""

        stats = sr.statistics("ACDEFGHIKLMNPQRSTVWY" * 3)
        with pytest.raises(ValueError, match="inside the declared structural support"):
            sr.reward(sr.REWARD_MEMORY, stats)

    def test_rewards_omits_a_structural_reward_only_when_it_could_be_measured(self):
        inside = sr.statistics("ACDEFGHIKLMNPQRSTVWY" * 3)
        assert sr.REWARD_MEMORY not in sr.rewards(inside)
        outside = sr.statistics("ACDEFGHI")
        # outside the support it is a measured zero, not an absent measurement
        assert sr.rewards(outside)[sr.REWARD_MEMORY] == sr.EMPTY_COMPLETION_REWARD

    def test_the_upper_end_of_the_support_is_honoured_too(self):
        stats = sr.statistics("ACDEFGHIKLMNPQRSTVWY" * 60)
        assert stats.length > sr.STRUCTURE_SUPPORT[1]
        assert sr.within_structure_support(stats) is False
        assert sr.reward(sr.REWARD_MEMORY, stats, mean_ca_plddt=90.0) == \
            sr.EMPTY_COMPLETION_REWARD
