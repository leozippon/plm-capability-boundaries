"""Reading an arm's generation products: the digest guard and the shard merge.

Two defects here would not announce themselves. A unit id names an anchor and a
partner, so analysing an arm against a different cohort build than it was
generated from reads the right draws at the wrong positions, and every number
downstream would look ordinary. And two shards of one arm entering the panel as
two columns would resample the same backbone under two names, narrowing a
simultaneous band that should not narrow. Both are tested on synthetic product
directories, which is the cheapest way to put a shard exactly on the boundary.
"""

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.io import sha256_file  # noqa: E402
from src.capability.forcing import forcing_design as D  # noqa: E402


def _stage(name: str):
    path = ROOT / f"scripts/capability/forcing/{name}.py"
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = _stage("analyse_forcing_gate")
COVERAGE = _stage("screen_forcing_coverage")


def write_products(
    directory: Path, *, arm: str, cohort: Path, unit_ids, shard=(0, 1), **overrides,
) -> Path:
    """A minimal but complete generation product directory."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "objects").mkdir(exist_ok=True)
    lines = []
    for unit_id in unit_ids:
        for condition in D.CONDITIONS:
            cell = f"{unit_id.replace(':', '_')}__{condition}__{D.MODE_SAMPLED}"
            np.save(
                directory / "objects" / f"{cell}.npy",
                np.full((2, 4, len(AA20)), 1.0 / len(AA20), dtype=np.float32),
            )
            lines.append(json.dumps({
                "cell": cell, "unit_id": unit_id, "accession": unit_id.split(":")[0],
                "length_band": "short", "condition": condition, "mode": D.MODE_SAMPLED,
                "partner": 70, "reference_positions": [71, 72, 73],
                "draws": [], "censored_draws": 0,
            }))
    (directory / "cells.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "status": "complete", "arm": arm, "in_declared_panel": True,
        "token_grid": {"marker_tokens": 1}, "dtype": D.DTYPE,
        "draws_per_cell": 2, "sampling_seed": D.SAMPLING_SEED,
        "decoding": {"temperature": D.TEMPERATURE, "top_p": D.TOP_P, "top_k": D.TOP_K},
        "shard": {"index": shard[0], "count": shard[1]},
        "cells": len(lines), "journal": "cells.jsonl", "objects": "objects",
        "censoring": {"censored_draws": 0, "total_draws": 2 * len(lines),
                      "cells_with_any_censoring": 0},
        "cohort": {"path": str(cohort), "sha256": sha256_file(cohort)},
    }
    summary.update(overrides)
    (directory / "forcing_generation.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory


@pytest.fixture
def cohort(tmp_path) -> Path:
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({"schema": "x", "unit_rows": []}), encoding="utf-8")
    return path


class TestCohortDigestGuard:
    def test_products_from_the_declared_cohort_are_admitted(self, tmp_path, cohort):
        directory = write_products(
            tmp_path / "a", arm="progen2-medium", cohort=cohort, unit_ids=["P1:40:70"],
        )
        arms = STAGE.read_arms([directory], cohort_sha256=sha256_file(cohort))
        assert len(arms) == 1 and len(arms[0]["records"]) == 2

    def test_products_from_another_cohort_build_are_refused(self, tmp_path, cohort):
        directory = write_products(
            tmp_path / "a", arm="progen2-medium", cohort=cohort, unit_ids=["P1:40:70"],
        )
        with pytest.raises(SystemExit, match="right draws at the wrong positions"):
            STAGE.read_arms([directory], cohort_sha256="0" * 64)

    def test_an_incomplete_arm_is_refused(self, tmp_path, cohort):
        directory = write_products(
            tmp_path / "a", arm="progen2-medium", cohort=cohort, unit_ids=["P1:40:70"],
            status="running",
        )
        with pytest.raises(SystemExit, match="not complete"):
            STAGE.read_arms([directory], cohort_sha256=sha256_file(cohort))

    def test_a_cell_without_its_array_is_refused(self, tmp_path, cohort):
        directory = write_products(
            tmp_path / "a", arm="progen2-medium", cohort=cohort, unit_ids=["P1:40:70"],
        )
        next(iter((directory / "objects").iterdir())).unlink()
        with pytest.raises(SystemExit, match="no distribution array"):
            STAGE.read_arms([directory], cohort_sha256=sha256_file(cohort))


class TestShardMerge:
    def test_two_shards_of_one_arm_become_one_column(self, tmp_path, cohort):
        first = write_products(
            tmp_path / "s0", arm="progen2-xlarge", cohort=cohort,
            unit_ids=["P1:40:70"], shard=(0, 2),
        )
        second = write_products(
            tmp_path / "s1", arm="progen2-xlarge", cohort=cohort,
            unit_ids=["P2:50:80"], shard=(1, 2),
        )
        arms = STAGE.read_arms([first, second], cohort_sha256=sha256_file(cohort))
        assert len(arms) == 1
        assert len(arms[0]["records"]) == 4
        assert arms[0]["summary"]["cells"] == 4
        assert len(arms[0]["directories"]) == 2

    def test_censoring_counts_are_summed_across_shards(self, tmp_path, cohort):
        first = write_products(
            tmp_path / "s0", arm="progen2-xlarge", cohort=cohort, unit_ids=["P1:40:70"],
            censoring={"censored_draws": 3, "total_draws": 48, "cells_with_any_censoring": 1},
        )
        second = write_products(
            tmp_path / "s1", arm="progen2-xlarge", cohort=cohort, unit_ids=["P2:50:80"],
            censoring={"censored_draws": 5, "total_draws": 48, "cells_with_any_censoring": 2},
        )
        arms = STAGE.read_arms([first, second], cohort_sha256=sha256_file(cohort))
        assert arms[0]["summary"]["censoring"] == {
            "censored_draws": 8, "total_draws": 96, "cells_with_any_censoring": 3,
        }

    def test_overlapping_shards_are_refused(self, tmp_path, cohort):
        """Overlap would double-count the same draws of the same backbone."""

        first = write_products(
            tmp_path / "s0", arm="progen2-xlarge", cohort=cohort, unit_ids=["P1:40:70"],
        )
        second = write_products(
            tmp_path / "s1", arm="progen2-xlarge", cohort=cohort, unit_ids=["P1:40:70"],
        )
        with pytest.raises(SystemExit, match="double-count"):
            STAGE.read_arms([first, second], cohort_sha256=sha256_file(cohort))

    @pytest.mark.parametrize(
        "override",
        [
            {"dtype": "bfloat16"},
            {"draws_per_cell": 48},
            {"sampling_seed": 1},
            {"token_grid": {"marker_tokens": 2}},
            {"decoding": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}},
        ],
    )
    def test_shards_differing_on_the_measurement_identity_are_refused(
        self, tmp_path, cohort, override,
    ):
        first = write_products(
            tmp_path / "s0", arm="progen2-xlarge", cohort=cohort, unit_ids=["P1:40:70"],
        )
        second = write_products(
            tmp_path / "s1", arm="progen2-xlarge", cohort=cohort, unit_ids=["P2:50:80"],
            **override,
        )
        with pytest.raises(SystemExit, match="not shards of one"):
            STAGE.read_arms([first, second], cohort_sha256=sha256_file(cohort))

    def test_different_arms_stay_separate_columns(self, tmp_path, cohort):
        first = write_products(
            tmp_path / "a", arm="progen2-medium", cohort=cohort, unit_ids=["P1:40:70"],
        )
        second = write_products(
            tmp_path / "b", arm="rita-xl", cohort=cohort, unit_ids=["P1:40:70"],
        )
        arms = STAGE.read_arms([first, second], cohort_sha256=sha256_file(cohort))
        assert [arm["summary"]["arm"] for arm in arms] == ["progen2-medium", "rita-xl"]

    def test_every_identity_field_the_merge_checks_is_declared(self):
        assert set(STAGE.SHARD_IDENTITY) == {
            "arm", "token_grid", "dtype", "draws_per_cell", "sampling_seed", "decoding",
        }


class TestCoverageBands:
    """A reference-coverage band and a remoteness band are different questions.

    Measured on the staged UniRef90 release: every one of the 111 backbones is a
    reviewed Swiss-Prot entry and therefore a verbatim member of a UniRef90
    cluster, so the self-inclusive band put all 111 in one level. That is the
    right answer to "is this covered" and no answer at all to "how remote is its
    nearest relative", which is why both are computed and why a one-level stratum
    has to announce itself rather than look like a control that passed.
    """

    class Hit:
        def __init__(self, nident, qlen, slen):
            self.nident, self.qlen, self.slen = nident, qlen, slen

    def test_an_exact_full_length_match_is_the_query_itself(self):
        assert COVERAGE.is_self_hit(self.Hit(nident=200, qlen=200, slen=200))

    def test_a_shorter_or_longer_subject_is_not_a_self_hit(self):
        assert not COVERAGE.is_self_hit(self.Hit(nident=200, qlen=200, slen=260))
        assert not COVERAGE.is_self_hit(self.Hit(nident=120, qlen=200, slen=200))

    def test_a_multi_level_band_is_usable_as_a_stratum(self):
        report = COVERAGE.band_degeneracy(
            {"A": "ge95_near_duplicate", "B": "lt30_no_detectable_homology"},
            name="nonself_identity_band",
        )
        assert report["levels"] == 2
        assert report["usable_as_a_stratum"] and report["consequence"] is None

    def test_a_single_level_band_declares_itself_unusable(self):
        report = COVERAGE.band_degeneracy(
            {"A": "ge95_near_duplicate", "B": "ge95_near_duplicate"},
            name="identity_band",
        )
        assert report["levels"] == 1
        assert not report["usable_as_a_stratum"]
        assert "not a control that was satisfied" in report["consequence"]
        assert report["counts"] == {"ge95_near_duplicate": 2}


class TestMaskingIsOff:
    """Low-complexity masking off is a correctness requirement in both searches.

    DIAMOND's default masking truncates high-identity alignments, which
    under-states identity over the query for exactly the near-verbatim pairs both
    of this lane's searches turn on: the within-panel ceiling that must exclude
    near-identical backbones, and the coverage band that must not read a verbatim
    corpus member as mere close homology. The repository's shared search helper
    documents the same hazard with its own measured example. These assertions
    exist so that removing the flag as tidying fails loudly.
    """

    def test_the_within_panel_identity_screen_disables_masking(self):
        builder = _stage("build_forcing_cohort")
        source = Path(
            ROOT / "scripts/capability/forcing/build_forcing_cohort.py"
        ).read_text()
        assert '"--masking", "0"' in source
        assert builder.self_identity_exclusions.__doc__ is not None

    def test_the_shared_search_helper_disables_masking(self):
        import inspect

        from src.capability.context import homology

        source = inspect.getsource(homology.run_diamond_blastp)
        assert '"--masking"' in source and '"0"' in source

    def test_the_coverage_screen_searches_through_that_helper(self):
        """So it inherits the flag rather than re-spelling it."""

        source = Path(
            ROOT / "scripts/capability/forcing/screen_forcing_coverage.py"
        ).read_text()
        assert "run_diamond_blastp(" in source
        assert "--masking" not in source
