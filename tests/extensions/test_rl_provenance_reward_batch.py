"""The trainer's reward path over a group, including the tail that ended a run.

A nine-hour run of arm `memory_and_environment` died at step 117 because one
sampled completion was a single residue and the folding instrument raises an
IndexError there. The group reward path is where that has to be survivable, and
where the declared structural support has to be applied, so it is tested here
against a stub instrument that answers exactly as the real one does: a reading
inside the support, nothing outside it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.capability.extensions import rl_environment_provenance as rp
from src.capability.extensions import sequence_reward as sr

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "scripts/capability/extensions/train_rl_provenance.py"


def _trainer():
    spec = importlib.util.spec_from_file_location("train_rl_provenance", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubFold:
    """A reading inside the declared support and nothing outside it.

    The real instrument raises an IndexError at one residue and returns a
    confidence that describes the input rather than a structure below sixteen, so
    :meth:`read_many` is what declines to fold there; this stub answers the same
    way and records what it was asked to fold.
    """

    def __init__(self, value=72.5):
        self.value = value
        self.asked = []

    def read_many(self, sequences):
        low, high = sr.STRUCTURE_SUPPORT
        out = []
        for sequence in sequences:
            if low <= len(sequence) <= high:
                self.asked.append(sequence)
                out.append(type("R", (), {
                    "mean_ca_plddt": self.value, "ptm": 0.8,
                    "diffusion_sample_index": 0, "seconds": 1.0,
                    "as_record": lambda self_: {"length": 0, "mean_ca_plddt": 72.5,
                                                "ptm": 0.8,
                                                "diffusion_sample_index": 0,
                                                "seconds": 1.0},
                })())
            else:
                out.append(None)
        return out


#: The group that killed the run: a one-residue completion, an empty one, a
#: peptide inside neither, and ordinary in-support sequences.
GROUP = ["A", "", "ACDEFGH", "ACDEFGHIKLMNPQRSTVWY" * 5, "ACDEFGHIKLMNPQRSTVWY" * 3]


class TestRewardBatchSurvivesTheTail:
    def test_a_one_residue_completion_no_longer_ends_the_run(self):
        module = _trainer()
        scorer = _StubFold()
        values, records = module.reward_batch(rp.ARM_MEMORY, GROUP, scorer)
        assert len(values) == len(GROUP)
        assert all(isinstance(value, float) for value in values)
        # the instrument was never asked to fold anything outside the support
        assert all(sr.STRUCTURE_SUPPORT[0] <= len(s) <= sr.STRUCTURE_SUPPORT[1]
                   for s in scorer.asked)
        assert "A" not in scorer.asked and "" not in scorer.asked

    def test_out_of_support_completions_are_paid_the_empty_value(self):
        module = _trainer()
        values, _ = module.reward_batch(rp.ARM_MEMORY, GROUP, _StubFold())
        assert values[0] == sr.EMPTY_COMPLETION_REWARD   # one residue
        assert values[1] == sr.EMPTY_COMPLETION_REWARD   # empty
        assert values[2] == sr.EMPTY_COMPLETION_REWARD   # seven residues
        assert values[3] > 0.0 and values[4] > 0.0       # in support

    def test_the_group_record_shows_the_tail_the_means_hide(self):
        module = _trainer()
        _, records = module.reward_batch(rp.ARM_MEMORY, GROUP, _StubFold())
        summary = module.group_record(records)
        assert summary["length_min"] == 0
        assert summary["empty_completions"] == 1
        assert summary["below_structure_support"] == 2  # the 1-mer and the 7-mer
        assert summary["length_max"] == 100

    def test_the_joint_arm_takes_the_same_rule(self):
        module = _trainer()
        values, _ = module.reward_batch(rp.ARM_BOTH, GROUP, _StubFold())
        assert values[0] == sr.EMPTY_COMPLETION_REWARD
        assert values[3] > 0.0

    def test_the_environment_arm_needs_no_instrument_and_scores_the_short_ones(self):
        """Its reward has no structural term, so the support rule does not reach it."""

        module = _trainer()
        values, _ = module.reward_batch(rp.ARM_ENVIRONMENT, GROUP, None)
        assert values[1] == sr.EMPTY_COMPLETION_REWARD   # empty is still empty
        assert values[2] > 0.0                           # a 7-mer is scored

    def test_a_group_of_in_support_sequences_with_no_reading_is_refused(self):
        """The refusal is kept for the case it was written for."""

        module = _trainer()

        class _Silent:
            def read_many(self, sequences):
                return [None] * len(sequences)

        with pytest.raises(ValueError, match="inside the declared structural support"):
            module.reward_batch(rp.ARM_MEMORY, ["ACDEFGHIKLMNPQRSTVWY" * 3], _Silent())
