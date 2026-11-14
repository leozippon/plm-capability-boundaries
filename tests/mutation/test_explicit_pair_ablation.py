"""The pair-level ablation against Profile, Potts and GEMME: what must always hold.

Three groups. The construction invariants every tier must satisfy on any input:
a declared tolerance is honoured for every named quantity, a pair that violates
one of them is excluded, an exact balance is exact, and the reverse tier equates
the model and leaves the statistics free. The reproducibility the bands claim:
one seed, one answer, and a tier below the eight-unit floor is reported rather
than banded. And the refusals, which have to fire: an incomplete upstream
baseline, a table that does not cover the support, a non-finite score, a
duplicate key, a pair with no phenotype gap and a tolerance that is not a
tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.capability.extensions import statistical_ladder as ladder  # noqa: E402
from src.capability.mutation import explicit_pair_ablation as epa  # noqa: E402
from src.capability.mutation import matched_discrimination as md  # noqa: E402

SAMPLES = ROOT / "manuscript/data/prediction-details/mutation-samples.csv.gz"
LADDER = ROOT / "results/extensions/statistical_ladder_20261106/accepted-scores"


# --------------------------------------------------------------------------- #
# fixtures: a pair set shaped exactly like the real cohort's
# --------------------------------------------------------------------------- #


def pair_frame(n: int = 240, *, clusters: int = 12, seed: int = 11) -> pd.DataFrame:
    """Oriented candidate pairs with the three statistics attached.

    The ranks are drawn once and attached through the production path, so the
    distance, direction and vote columns under test are the ones the analysis
    builds and not a re-derivation of them.
    """

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "assay": [f"ASSAY_{index % 6}" for index in range(n)],
        "wildtype_residue": ["A"] * n,
        "mutant_residue": ["V"] * n,
        "position_a": np.arange(1, n + 1),
        "position_b": np.arange(n + 1, 2 * n + 1),
        "sample_a": [f"a{index}" for index in range(n)],
        "sample_b": [f"b{index}" for index in range(n)],
        "group_id": [index % clusters for index in range(n)],
        "z_a": rng.normal(size=n) + 2.0,
        "z_b": rng.normal(size=n),
    })
    frame = epa.orient(frame)
    for name in epa.STATISTICS:
        frame = epa.attach_covariate(frame, name, ranks(frame, rng, ties=name == "gemme"))
    return frame


def ranks(frame: pd.DataFrame, rng: np.random.Generator, *, ties: bool = False) -> dict[str, float]:
    """A rank value per pair member; ``ties`` makes a fifth of the pairs exact ties."""

    values: dict[str, float] = {}
    for index, (left, right) in enumerate(zip(frame["sample_a"], frame["sample_b"], strict=True)):
        value = float(rng.normal())
        values[left] = value
        values[right] = value if ties and index % 5 == 0 else float(rng.normal())
    return values


def baseline_document(
    name: str, identifiers, *, column: str, score=0.0, status=None, counts=None, extra=None
) -> dict:
    """An upstream baseline document in the shape the statistical ladder reads.

    The frozen admission contract pins the R1 row count, so a document that is
    meant to reach :func:`read_baseline` carries that many rows; a document that
    only has to be joined does not.
    """

    rows = []
    for index, identifier in enumerate(identifiers):
        row = {
            "sample_id": identifier,
            "assay": "ASSAY_0",
            "wildtype_id": "q00000",
            "mutation": f"A{index + 1}V",
            "n_substitutions": 1,
            column: float(score) if np.isscalar(score) else float(score[index]),
            "status": "fitted" if name == "potts" else "scored",
        }
        if status is not None:
            row["status"] = status[index]
        if name == "gemme":
            row["n_runs"] = 3
        rows.append(row)
    document = {
        "counts": dict(ladder.BASELINE_CONTRACT[name]["counts"]) | (counts or {}),
        "samples": rows,
    }
    if name == "gemme":
        document["repeats"] = 3
        document["incomplete"] = []
    if extra:
        document.update(extra)
    return document


def write_document(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def read_table(path: Path, document: dict) -> dict:
    """One score table read through the production reader, without admission."""

    return ladder.read_score_table(write_document(path, document), metadata_required=True)


@pytest.fixture(scope="module")
def contract_identifiers() -> list[str]:
    """As many sample ids as the frozen admission contract requires."""

    return [f"c{index:06d}" for index in range(ladder.BASELINE_CONTRACT["potts"]["counts"]["samples"])]


def support_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{index}" for index in range(n)],
        "unit_id": ["ASSAY_0"] * n,
        "group_id": [0] * n,
        "protein_id": ["q00000"] * n,
        "profile_score": np.linspace(-1.0, 1.0, n),
    })


# --------------------------------------------------------------------------- #
# the tolerance: honoured for every quantity, and violating one excludes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tolerance", [0.05, 0.2, 0.5, 1.0])
def test_tolerance_is_honoured_for_every_named_quantity(tolerance):
    frame = pair_frame()
    matched = epa.tolerance_matched(frame, epa.STATISTICS, tolerance)
    for name in epa.STATISTICS:
        assert (matched[f"{name}_distance"].to_numpy() <= tolerance).all()
    expected = np.ones(len(frame), dtype=bool)
    for name in epa.STATISTICS:
        expected &= frame[f"{name}_distance"].to_numpy() <= tolerance
    assert len(matched) == int(expected.sum())


def test_a_pair_violating_one_tolerance_is_excluded():
    frame = pair_frame()
    tolerance = 0.3
    matched = epa.tolerance_matched(frame, epa.STATISTICS, tolerance)
    kept = set(zip(matched["sample_a"], matched["sample_b"]))
    offenders = 0
    for row in frame.itertuples(index=False):
        distances = [getattr(row, f"{name}_distance") for name in epa.STATISTICS]
        if max(distances) <= tolerance:
            continue
        offenders += 1
        assert (row.sample_a, row.sample_b) not in kept
        # and it would have been kept had only the satisfied quantities been named
        satisfied = [name for name, value in zip(epa.STATISTICS, distances) if value <= tolerance]
        if satisfied:
            partial = epa.tolerance_matched(frame, satisfied, tolerance)
            assert (row.sample_a, row.sample_b) in set(
                zip(partial["sample_a"], partial["sample_b"])
            )
    assert offenders > 0


def test_adding_a_control_block_never_widens_a_matched_tier():
    frame = pair_frame()
    for tolerance in epa.TOLERANCE_LADDER:
        profile_only = epa.tolerance_matched(frame, ("profile",), tolerance)
        all_three = epa.tolerance_matched(frame, epa.STATISTICS, tolerance)
        assert len(all_three) <= len(profile_only)
        assert set(zip(all_three["sample_a"], all_three["sample_b"])) <= set(
            zip(profile_only["sample_a"], profile_only["sample_b"])
        )


def test_a_tolerance_must_be_positive_and_finite():
    frame = pair_frame()
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            epa.tolerance_matched(frame, epa.STATISTICS, bad)
    with pytest.raises(ValueError, match="at least one quantity"):
        epa.tolerance_matched(frame, (), 0.2)


def test_an_unattached_quantity_cannot_be_matched():
    frame = pair_frame()
    with pytest.raises(ValueError, match="unenforceable"):
        epa.tolerance_matched(frame, ("model",), 0.2)


# --------------------------------------------------------------------------- #
# the exact balance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("names", [("profile",), ("profile", "potts"), epa.STATISTICS])
def test_balance_is_exact_on_every_named_vote(names):
    frame = pair_frame()
    balanced, record = epa.balance_votes(frame, names, seed=3)
    assert record["certified_direction_accuracy"] == {name: 0.5 for name in names}
    for name in names:
        assert float(balanced[f"{name}_direction"].to_numpy().mean()) == 0.5
    assert epa.certify_balance(balanced, names) == {name: 0.5 for name in names}


def test_balance_retains_all_neutral_pairs_and_drops_partially_neutral_ones():
    frame = pair_frame()
    votes = np.stack([frame[f"{name}_vote"].to_numpy() for name in epa.STATISTICS], axis=1)
    all_neutral = int((votes == 0).all(axis=1).sum())
    partial = int(((votes == 0).any(axis=1) & ~(votes == 0).all(axis=1)).sum())
    balanced, record = epa.balance_votes(frame, epa.STATISTICS, seed=3)
    assert record["all_neutral_retained"] == all_neutral
    assert record["partially_neutral_excluded"] == partial
    assert partial > 0, "the fixture must exercise the partially neutral path"
    kept = np.stack([balanced[f"{name}_vote"].to_numpy() for name in epa.STATISTICS], axis=1)
    assert not ((kept == 0).any(axis=1) & ~(kept == 0).all(axis=1)).any()


def test_balance_is_deterministic_under_its_seed_and_moves_with_it():
    frame = pair_frame()
    first, _ = epa.balance_votes(frame, epa.STATISTICS, seed=5)
    again, _ = epa.balance_votes(frame, epa.STATISTICS, seed=5)
    other, _ = epa.balance_votes(frame, epa.STATISTICS, seed=6)
    pd.testing.assert_frame_equal(first, again)
    assert len(first) == len(other)
    assert not first.equals(other)


def test_certify_balance_refuses_an_unbalanced_tier():
    frame = pair_frame()
    balanced, _ = epa.balance_votes(frame, epa.STATISTICS, seed=3)
    tilted = pd.concat(
        [balanced, balanced.loc[balanced["profile_direction"] == 1.0].head(4)], ignore_index=True
    )
    with pytest.raises(ValueError, match="not exactly balanced on profile"):
        epa.certify_balance(tilted, epa.STATISTICS)
    with pytest.raises(ValueError, match="cannot be certified balanced"):
        epa.certify_balance(balanced.head(0), ("profile",))


def test_balance_refuses_an_empty_set_and_an_empty_vote_list():
    frame = pair_frame()
    with pytest.raises(ValueError, match="at least one vote"):
        epa.balance_votes(frame, (), seed=3)
    with pytest.raises(ValueError, match="empty pair set"):
        epa.balance_votes(frame.head(0), epa.STATISTICS, seed=3)


# --------------------------------------------------------------------------- #
# the reverse tier equates the model and not the statistics
# --------------------------------------------------------------------------- #


def test_reverse_tiers_equate_the_model_and_leave_the_statistics_free():
    frame = pair_frame()
    rng = np.random.default_rng(99)
    frame = epa.attach_covariate(frame, epa.MODEL, ranks(frame, rng))
    open_tier = md.greedy_disjoint(frame, seed=md.SPLIT_SEED)
    built = epa.reverse_tiers(frame, open_tier, tolerances=(0.2,))
    assert set(built) == {"T_bal_M", "T_tol_M(0.2)", "T_tol_bal_M(0.2)"}

    balanced = built["T_bal_M"]["pairs"]
    assert float(balanced[f"{epa.MODEL}_direction"].to_numpy().mean()) == 0.5
    # the statistics are not equated by a reverse tier: at least one of the three
    # still orders the pairs away from chance, which is the quantity being read.
    assert any(
        float(balanced[f"{name}_direction"].to_numpy().mean()) != 0.5 for name in epa.STATISTICS
    )
    matched = built["T_tol_M(0.2)"]["pairs"]
    assert (matched[f"{epa.MODEL}_distance"].to_numpy() <= 0.2).all()
    assert built["T_bal_M"]["construction"]["balanced_votes"] == [epa.MODEL]
    conjunction = built["T_tol_bal_M(0.2)"]["pairs"]
    assert (conjunction[f"{epa.MODEL}_distance"].to_numpy() <= 0.2).all()
    if len(conjunction):
        assert float(conjunction[f"{epa.MODEL}_direction"].to_numpy().mean()) == 0.5


def test_forward_tiers_equate_the_statistics_and_name_every_declared_tier():
    frame = pair_frame()
    built = epa.forward_tiers(frame, tolerances=(0.2, 0.8))
    assert set(built) == {
        "T_open", "T_bal_P", "T_bal_PEG",
        "T_tol_P(0.2)", "T_tol_PEG(0.2)", "T_tol_bal_PEG(0.2)",
        "T_tol_P(0.8)", "T_tol_PEG(0.8)", "T_tol_bal_PEG(0.8)",
    }
    for name in epa.STATISTICS:
        assert float(built["T_bal_PEG"]["pairs"][f"{name}_direction"].to_numpy().mean()) == 0.5
        assert (built["T_tol_PEG(0.8)"]["pairs"][f"{name}_distance"].to_numpy() <= 0.8).all()
    assert float(
        built["T_bal_P"]["pairs"]["profile_direction"].to_numpy().mean()
    ) == 0.5
    # no variant is used twice inside a tier
    for tier in built.values():
        pairs = tier["pairs"]
        used = list(pairs["sample_a"]) + list(pairs["sample_b"])
        assert len(set(used)) == len(used)


# --------------------------------------------------------------------------- #
# bands
# --------------------------------------------------------------------------- #


def test_bands_are_reproducible_from_a_fixed_seed_and_move_with_it():
    frame = pair_frame()
    columns = {name: frame[f"{name}_direction"].to_numpy(float) for name in epa.STATISTICS}
    clusters = frame["group_id"].to_numpy()
    first = epa.family_bands(columns, clusters, draws=400, seed=20261114)
    again = epa.family_bands(columns, clusters, draws=400, seed=20261114)
    other = epa.family_bands(columns, clusters, draws=400, seed=20261115)
    assert first == again
    assert first["band"]["critical_value"] != other["band"]["critical_value"]
    for name in epa.STATISTICS:
        assert first["columns"][name]["pooled"] == other["columns"][name]["pooled"]


def test_the_endpoint_is_the_pooled_pair_fraction_the_balance_is_defined_on():
    """A balanced tier's reference rule sits at exactly 0.5 on the reported endpoint."""

    frame = pair_frame()
    balanced, _ = epa.balance_votes(frame, epa.STATISTICS, seed=3)
    record = epa.family_bands(
        {name: balanced[f"{name}_direction"].to_numpy(float) for name in epa.STATISTICS},
        balanced["group_id"].to_numpy(),
        draws=400,
    )
    for name in epa.STATISTICS:
        column = record["columns"][name]
        assert column["pooled"] == 0.5
        assert column["simultaneous"][0] <= column["pooled"] <= column["simultaneous"][1]
    assert "pooled pair fraction" in record["endpoint"]


def test_the_simultaneous_band_is_never_narrower_than_the_pointwise_one():
    frame = pair_frame()
    record = epa.family_bands(
        {name: frame[f"{name}_direction"].to_numpy(float) for name in epa.STATISTICS},
        frame["group_id"].to_numpy(),
        draws=400,
    )
    for name in epa.STATISTICS:
        column = record["columns"][name]
        wide = column["simultaneous"][1] - column["simultaneous"][0]
        narrow = column["pointwise"][1] - column["pointwise"][0]
        assert wide >= narrow - 1e-12


def test_a_family_below_the_unit_floor_is_not_estimable():
    frame = pair_frame(clusters=4)
    record = epa.family_bands(
        {"profile": frame["profile_direction"].to_numpy(float)},
        frame["group_id"].to_numpy(),
        draws=200,
    )
    assert record["status"] == "not_estimable"
    assert record["degenerate"] is True
    assert record["n_clusters"] == 4
    assert record["band"] is None
    assert record["columns"]["profile"]["simultaneous"] is None
    assert record["columns"]["profile"]["verdict"]["verdict"] == "withheld"
    assert record["columns"]["profile"]["pooled"] is not None


def test_the_verdict_rule_is_the_declared_one():
    assert md.level_verdict([0.52, 0.60])["verdict"] == "supported"
    assert md.level_verdict([0.46, 0.54])["verdict"] == "bounded_negative"
    assert md.level_verdict([0.44, 0.58])["verdict"] == "unresolved"
    record = epa.declared_constants()
    assert record["null_accuracy"] == 0.5
    assert record["practical_margin"] == 0.05
    assert record["primary_tolerance"] == 0.20
    assert record["tolerance_ladder"] == [0.05, 0.10, 0.20, 0.40, 0.80]
    assert record["bootstrap_seed"] == 20261114


def test_a_family_refuses_a_misaligned_or_non_finite_column():
    frame = pair_frame()
    clusters = frame["group_id"].to_numpy()
    with pytest.raises(ValueError, match="one direction value per pair"):
        epa.family_bands({"x": np.zeros(len(frame) - 1)}, clusters, draws=50)
    broken = frame["profile_direction"].to_numpy(float).copy()
    broken[0] = np.nan
    with pytest.raises(ValueError, match="no finite direction value"):
        epa.family_bands({"x": broken}, clusters, draws=50)
    with pytest.raises(ValueError, match="at least one column"):
        epa.family_bands({}, clusters, draws=50)
    with pytest.raises(ValueError, match="unknown verdict source"):
        epa.family_bands(
            {"x": frame["profile_direction"].to_numpy(float)}, clusters, draws=50,
            verdict_from="whatever",
        )
    foreign = md.ClusterResampler([100, 101, 102, 103, 104, 105, 106, 107], draws=50, seed=1)
    with pytest.raises(ValueError, match="this tier's own clusters"):
        epa.family_bands(
            {"x": frame["profile_direction"].to_numpy(float)}, clusters, resampler=foreign,
        )


# --------------------------------------------------------------------------- #
# refusals on the statistics
# --------------------------------------------------------------------------- #


def test_an_incomplete_upstream_baseline_is_refused(tmp_path, contract_identifiers):
    identifiers = contract_identifiers
    path = write_document(
        tmp_path / "potts-unconverged.json",
        baseline_document("potts", identifiers, column="potts_delta",
                          counts={"backgrounds_not_converged": 1}),
    )
    with pytest.raises(ValueError, match="disagree with the frozen baseline contract"):
        epa.read_baseline("potts", path)
    path = write_document(
        tmp_path / "potts-short.json",
        baseline_document("potts", identifiers[:-1], column="potts_delta"),
    )
    with pytest.raises(ValueError, match="disagree with the frozen baseline contract"):
        epa.read_baseline("potts", path)
    path = write_document(
        tmp_path / "potts-unfitted.json",
        baseline_document("potts", identifiers, column="potts_delta",
                          status=["fitted"] * (len(identifiers) - 1) + ["not_converged"]),
    )
    with pytest.raises(ValueError, match="rows are not fitted"):
        epa.read_baseline("potts", path)
    path = write_document(
        tmp_path / "gemme-two-runs.json",
        baseline_document("gemme", identifiers, column="gemme_score", extra={"repeats": 2}),
    )
    with pytest.raises(ValueError, match="upstream repeats is 2"):
        epa.read_baseline("gemme", path)
    path = write_document(
        tmp_path / "gemme-incomplete.json",
        baseline_document("gemme", identifiers, column="gemme_score",
                          extra={"incomplete": ["q00000"]}),
    )
    with pytest.raises(ValueError, match="a complete baseline declares it empty"):
        epa.read_baseline("gemme", path)
    with pytest.raises(ValueError, match="not one of the admitted baselines"):
        epa.read_baseline("profile", path)


def test_a_non_finite_score_is_never_imputed(tmp_path):
    support = support_frame()
    identifiers = list(support["sample_id"])
    scores = [0.0] * len(identifiers)
    scores[2] = float("nan")
    table = read_table(
        tmp_path / "potts-nan.json",
        baseline_document("potts", identifiers, column="potts_delta", score=scores),
    )
    with pytest.raises(ValueError, match="non-finite"):
        epa.attach_statistics(support, {"potts": table, "gemme": table})
    constant = {
        name: read_table(
            tmp_path / f"{name}-constant.json",
            baseline_document(name, identifiers, column=epa.SOURCE_COLUMN[name], score=0.0),
        )
        for name in epa.BASELINE_STATISTICS
    }
    with pytest.raises(ValueError, match="there is no support left"):
        epa.attach_statistics(support, constant)


def test_the_statistics_are_joined_onto_one_shared_rank_scale(tmp_path):
    support = support_frame()
    identifiers = list(support["sample_id"])
    scores = np.linspace(-2.0, 2.0, len(identifiers))
    tables = {
        "potts": read_table(
            tmp_path / "potts.json",
            baseline_document("potts", identifiers, column="potts_delta", score=scores),
        ),
        "gemme": read_table(
            tmp_path / "gemme.json",
            baseline_document("gemme", identifiers, column="gemme_score", score=scores[::-1]),
        ),
    }
    frame, refusals = epa.attach_statistics(support, tables)
    assert refusals["rows_lost"] == 0
    assert refusals["by_statistic"] == {}
    assert np.allclose(frame["potts"].to_numpy(), scores)
    assert np.allclose(frame["gemme"].to_numpy(), scores[::-1])
    for name in epa.STATISTICS:
        column = frame[f"rank_{name}"].to_numpy()
        assert abs(float(column.std(ddof=0)) - 1.0) < 1e-12
        assert abs(float(column.mean())) < 1e-12
    assert set(epa.rank_lookup(frame, "potts")) == set(identifiers)
    with pytest.raises(ValueError, match="no attached rank column"):
        epa.rank_lookup(frame, "model")


def test_a_statistic_with_no_within_assay_spread_is_refused_by_reference(tmp_path):
    """The ladder's refused_degenerate condition, applied before the rank scale."""

    support = pd.concat(
        [support_frame(), support_frame().assign(
            sample_id=[f"t{index}" for index in range(6)], unit_id="ASSAY_1", protein_id="q00001",
            group_id=1,
        )],
        ignore_index=True,
    )
    identifiers = list(support["sample_id"])
    potts = read_table(
        tmp_path / "potts.json",
        baseline_document("potts", identifiers, column="potts_delta",
                          score=np.arange(len(identifiers), dtype=float)),
    )
    # GEMME is identically zero over ASSAY_1, exactly as the published R1 table is
    # for the seven wild types its own counts record as degenerate.
    dead = np.arange(len(identifiers), dtype=float)
    dead[6:] = 0.0
    gemme = read_table(
        tmp_path / "gemme.json",
        baseline_document("gemme", identifiers, column="gemme_score", score=dead),
    )
    frame, refusals = epa.attach_statistics(support, {"potts": potts, "gemme": gemme})
    assert refusals["assays_refused"] == ["ASSAY_1"]
    assert refusals["references_refused"] == ["q00001"]
    assert refusals["rows_offered"] == 12
    assert refusals["rows_retained"] == 6
    assert refusals["rows_lost"] == 6
    detail = refusals["by_statistic"]["gemme"]["references"]["q00001"]
    assert detail["status"] == epa.DEGENERATE_STATUS
    assert detail["rows"] == 6
    assert detail["constant_value"] == 0.0
    assert set(frame["unit_id"]) == {"ASSAY_0"}
    assert epa.degenerate_blocks(frame) == {}


def test_a_table_that_does_not_cover_the_support_is_refused_by_reference(tmp_path):
    support = support_frame()
    identifiers = list(support["sample_id"])
    table = read_table(
        tmp_path / "potts.json",
        baseline_document("potts", identifiers[:-1], column="potts_delta"),
    )
    with pytest.raises(ValueError, match="does not cover the support.*references"):
        epa.coverage("potts", support, table)
    with pytest.raises(ValueError, match="does not cover the support"):
        epa.attach_statistics(support, {"potts": table, "gemme": table})
    extra = read_table(
        tmp_path / "potts-extra.json",
        baseline_document("potts", identifiers + ["unexpected"], column="potts_delta"),
    )
    with pytest.raises(ValueError, match="extra sample_id"):
        epa.coverage("potts", support, extra)


def test_a_duplicate_key_is_refused_on_either_side(tmp_path):
    support = support_frame()
    identifiers = list(support["sample_id"])
    table = read_table(
        tmp_path / "potts-duplicate.json",
        baseline_document("potts", identifiers[:-1] + [identifiers[0]], column="potts_delta"),
    )
    with pytest.raises(ValueError, match="carries duplicate sample_id"):
        epa.coverage("potts", support, table)
    clean = read_table(
        tmp_path / "potts.json",
        baseline_document("potts", identifiers, column="potts_delta"),
    )
    with pytest.raises(ValueError, match="support carries duplicate sample_id"):
        epa.coverage("potts", pd.concat([support, support.head(1)]), clean)


def test_the_support_needs_every_baseline(tmp_path):
    support = support_frame()
    table = read_table(
        tmp_path / "potts.json",
        baseline_document("potts", list(support["sample_id"]), column="potts_delta"),
    )
    with pytest.raises(ValueError, match=r"needs the \['gemme'\] baseline tables"):
        epa.attach_statistics(support, {"potts": table})


@pytest.mark.skipif(
    not (SAMPLES.exists() and (LADDER / "potts/scores.json").exists()
         and (LADDER / "gemme/scores.json").exists()),
    reason="the retained panel or the accepted ladder scores are not staged here",
)
def test_the_real_ladder_scores_are_admitted_and_cover_the_real_support():
    """The frozen inputs this reading actually runs on, end to end through admission."""

    support = md.common_support(pd.read_csv(SAMPLES))
    tables = {}
    for name in epa.BASELINE_STATISTICS:
        record = epa.read_baseline(name, LADDER / f"{name}/scores.json")
        assert record["admission"]["rows"] == 25728
        assert epa.coverage(name, support, record["table"])["rows_lost"] == 0
        tables[name] = record["table"]
    frame, refusals = epa.attach_statistics(support, tables)
    # The seven wild types the ladder's own GEMME counts record as degenerate are
    # refused here, by reference, before any pair is built.
    assert refusals["rows_offered"] == 25728
    assert refusals["references_refused"] == [
        "q00025", "q00037", "q00102", "q00159", "q00176", "q00180", "q00186",
    ]
    assert list(refusals["by_statistic"]) == ["gemme"]
    assert refusals["rows_lost"] == 896
    assert len(frame) == 24832
    assert frame["unit_id"].nunique() == 194
    for name in epa.STATISTICS:
        spread = epa.rank_distribution(frame, name)
        assert abs(spread["sd"] - 1.0) < 1e-9
        assert epa.degenerate_blocks(frame) == {}


def test_a_pair_with_no_phenotype_gap_cannot_be_oriented():
    frame = pd.DataFrame({
        "assay": ["ASSAY_0"], "wildtype_residue": ["A"], "mutant_residue": ["V"],
        "position_a": [1], "position_b": [2], "sample_a": ["a"], "sample_b": ["b"],
        "group_id": [0], "z_a": [1.0], "z_b": [1.0],
    })
    with pytest.raises(ValueError, match="no phenotype gap"):
        epa.orient(frame)


def test_a_pair_member_without_a_score_is_refused():
    frame = pair_frame(n=8)
    incomplete = {key: value for key, value in zip(
        list(frame["sample_a"]), range(len(frame))
    )}
    with pytest.raises(ValueError, match="no finite score"):
        epa.attach_covariate(frame, epa.MODEL, incomplete)


def test_tier_labels_name_only_declared_tiers():
    assert epa.tier_label("T_open") == "T_open"
    assert epa.tier_label("T_tol_PEG", 0.2) == "T_tol_PEG(0.2)"
    with pytest.raises(ValueError, match="not a declared tier"):
        epa.tier_label("T_whatever")
