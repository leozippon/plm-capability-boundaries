"""The pool sampler's own logic: cell identity, seed discipline and the fold cohort.

The model path is not exercised here -- it was proved on the real checkpoint, which
loaded through the project's generation lane under its declared shape check and
wrote a ledger naming its weights. What is exercised is everything the sampler adds
around that, because those are the parts that can be silently wrong: a cell whose
seed is not the declared one, a pool read from another pool's ledgers, and a cohort
the project's own structure runner would refuse.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.capability.extensions import rl_environment_provenance as rp
from src.capability.generation import structure_evidence as se
from src.capability.generation import unconditional_generation as ug

ROOT = Path(__file__).resolve().parents[2]
SAMPLER = ROOT / "scripts/capability/extensions/sample_rl_provenance_pools.py"


def _sampler():
    spec = importlib.util.spec_from_file_location("sample_rl_provenance_pools", SAMPLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cell(out: Path, pool: str, stream: int, sequences, *, seed=None,
                checkpoint="/declared/checkpoint"):
    """A real lane cell, written by the lane's own immutable writer."""

    module = _sampler()
    directory = out / module.CELL.format(pool=pool, stream=stream)
    rows = [ug.attempt_row(rp.POLICY_ARM, sequence + "<|eos|>", index)
            for index, sequence in enumerate(sequences)]
    realised = rp.pool_seeds(pool)[stream] if seed is None else seed
    ug.write_attempt_ledger(directory, rows, arm=rp.POLICY_ARM, seed=realised,
                            checkpoint=checkpoint)
    return directory


SEQUENCES = ["ACDEFGHIKLMNPQRSTVWY" * 3, "MKKRRDEEAAGGLLVVTTSSYYACDEFGHIKLMNPQ", ""]


class TestCellIdentity:
    def test_a_cell_is_named_for_its_pool_and_stream(self, tmp_path):
        module = _sampler()
        names = [path.name for path in module.cell_dirs(tmp_path, rp.POOL_BASE)]
        assert names == [f"rlp_{rp.POOL_BASE}_stream{index}" for index in range(rp.STREAMS)]

    def test_a_pool_reads_its_own_streams_with_the_stream_attached(self, tmp_path):
        module = _sampler()
        for stream in range(rp.STREAMS):
            _write_cell(tmp_path, rp.POOL_BASE, stream, SEQUENCES)
        rows = module.read_pool(tmp_path, rp.POOL_BASE)
        assert len(rows) == rp.STREAMS * len(SEQUENCES)
        assert sorted({row["stream"] for row in rows}) == list(range(rp.STREAMS))
        assert {row["pool"] for row in rows} == {rp.POOL_BASE}
        assert {row["checkpoint"] for row in rows} == {"/declared/checkpoint"}


class TestSeedDiscipline:
    def test_a_cell_whose_seed_is_not_the_declared_one_is_refused(self, tmp_path):
        """The seed is what distinguishes the cells, so a wrong one is a wrong pool."""

        module = _sampler()
        for stream in range(rp.STREAMS):
            _write_cell(tmp_path, rp.POOL_BASE, stream, SEQUENCES)
        _write_cell(tmp_path, rp.POOL_BASE, 1, SEQUENCES, seed=999)
        with pytest.raises(SystemExit, match="not the declared"):
            module.read_pool(tmp_path, rp.POOL_BASE)

    def test_a_missing_stream_is_refused_rather_than_averaged_over(self, tmp_path):
        module = _sampler()
        for stream in range(rp.STREAMS - 1):
            _write_cell(tmp_path, rp.POOL_BASE, stream, SEQUENCES)
        with pytest.raises(SystemExit, match="no generation record"):
            module.read_pool(tmp_path, rp.POOL_BASE)

    def test_no_two_pools_share_a_stream_seed(self):
        seeds = [seed for pool in rp.EVALUATED_POOLS for seed in rp.pool_seeds(pool)]
        assert len(set(seeds)) == len(seeds)


class TestFoldCohort:
    def test_the_cohort_is_distinct_sequences_and_the_runner_accepts_it(self, tmp_path):
        module = _sampler()
        for stream in range(rp.STREAMS):
            _write_cell(tmp_path, rp.POOL_BASE, stream, SEQUENCES)
        args = type("A", (), {"pool": rp.POOL_BASE, "out": tmp_path})()
        assert module.cohort(args) == 0
        path = tmp_path / f"{rp.POOL_BASE}_{module.COHORT}"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # three streams of the same two non-empty sequences: one fold each
        assert len(rows) == 2
        assert all(row["id"] == row["sequence_sha256"] for row in rows)
        assert all(row["id"] == se.sequence_digest(row["sequence"]) for row in rows)
        # the runner's own reader is the acceptance check, and it ran inside cohort()
        assert len(se.load_cohort(path)) == 2

    def test_an_empty_continuation_carries_no_fold(self, tmp_path):
        module = _sampler()
        for stream in range(rp.STREAMS):
            _write_cell(tmp_path, rp.POOL_BASE, stream, SEQUENCES)
        args = type("A", (), {"pool": rp.POOL_BASE, "out": tmp_path})()
        module.cohort(args)
        rows = [json.loads(line) for line in
                (tmp_path / f"{rp.POOL_BASE}_{module.COHORT}").read_text().splitlines()
                if line.strip()]
        assert all(row["sequence"] for row in rows)


class TestCheckpointResolution:
    def test_the_base_and_the_published_sibling_resolve_from_the_declaration(self):
        module = _sampler()
        assert module.checkpoint_for(rp.POOL_BASE, policies=None) == rp.policy_checkpoint()
        assert (module.checkpoint_for(rp.POOL_PUBLISHED_DPO, policies=None)
                == rp.published_dpo_checkpoint())

    def test_a_trained_arm_needs_its_run_directory_and_refuses_without_one(self, tmp_path):
        module = _sampler()
        with pytest.raises(SystemExit, match="--policies must name"):
            module.checkpoint_for(rp.ARM_ENVIRONMENT, policies=None)
        with pytest.raises(SystemExit, match="no trained checkpoint"):
            module.checkpoint_for(rp.ARM_ENVIRONMENT, policies=tmp_path)

    def test_a_trained_arm_resolves_to_its_latest_checkpoint(self, tmp_path):
        module = _sampler()
        directory = tmp_path / rp.ARM_ENVIRONMENT / "latest"
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")
        assert module.checkpoint_for(rp.ARM_ENVIRONMENT, policies=tmp_path) == directory
