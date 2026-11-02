"""The matched-candidate discrimination rows, and the refusals that keep them honest.

The module reads six retained artifacts and writes every row from one of them, so
the conditions that must always hold are the ones the ledger itself is for. Every
row names the file, the pointer into it and the digest of that file, and the digest
is the digest of the bytes the value was read from, not of a neighbouring copy. An
interval carries the unit it was resampled over and the draw count behind it.

Two things about the experiment's two supports make the rest. The masked arm is
reported beside the autoregressive arms and is never one of them, so no aggregate
that the manuscript writes over "the eight autoregressive arms" may take a bound
from it, and the masked arm is the extreme of two of those spans, so a leak would
move the bound the Results quote. And a verdict count is a count of one outcome on
one side: an arm that resolves downward is a finding of its own and is not a gain.

The module's guards are tested the way it applies them, against a mutated copy of an
artifact in a temporary root: the row that would otherwise claim something the
artifact no longer says must be refused rather than written. The artifacts are the
retained ones, because the structure the rows are positioned by is the artifacts'
own and a hand-built stand-in would only test itself; the whole module is skipped
where the checkout does not carry them.
"""
import copy
import hashlib
import json
import shutil

import pytest

from scripts.capability.reporting import matched_discrimination_quantities as matched
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError, REPO_ROOT, resolve

ARTIFACTS = (matched.EXPANDED, matched.COHORT, matched.CONTRACT, matched.PANEL, matched.PANEL_A3,
             matched.MANIFEST)
MASKED = "esm2_t33_650M_UR50D"
PREFIX = f"{matched.FAMILY}/"
COUNTS = f"{matched.FAMILY}/retained_panel"


@pytest.fixture(autouse=True, scope="module")
def _the_retained_artifacts_are_staged():
    missing = [path for path in ARTIFACTS if not (REPO_ROOT / path).is_file()]
    if missing:
        pytest.skip(f"{missing[0]} is not staged in this checkout")


def build(root=None) -> Ledger:
    ledger = Ledger(Artifacts() if root is None else Artifacts(root))
    matched.add_matched_discrimination_quantities(ledger)
    return ledger


@pytest.fixture(scope="module")
def retained() -> Ledger:
    return build()


def load(relative: str) -> dict:
    return json.loads((REPO_ROOT / relative).read_text())


def stage(tmp_path, artifact: str | None = None, mutate=None) -> Ledger:
    """The module run over a temporary root holding a mutated copy of one artifact."""
    for relative in ARTIFACTS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == artifact:
            payload = copy.deepcopy(load(relative))
            mutate(payload)
            target.write_text(json.dumps(payload))
        else:
            shutil.copyfile(REPO_ROOT / relative, target)
    return build(tmp_path)


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


# --------------------------------------------------------------------------- #
# What every row carries.
# --------------------------------------------------------------------------- #

def test_every_row_names_its_source_unit_and_resampling_record(retained):
    assert retained.quantities
    for row in retained.quantities:
        assert row.id.startswith(PREFIX) and row.family == matched.FAMILY, row.id
        assert row.source_path in ARTIFACTS, row.id
        assert row.source_pointer and all(row.source_pointer), row.id
        assert len(row.source_sha256) == 64, row.id
        assert row.unit, row.id
        assert row.support_id in retained.supports, row.id
        # An interval is an estimate's, and an estimate always carries one here:
        # every cell of this experiment is resampled and none is a bare point.
        assert (row.interval is not None) == (row.kind == "estimate"), row.id
        if row.interval is not None:
            assert row.resampling_unit, row.id
            assert isinstance(row.resampling_draws, int) and row.resampling_draws > 0, row.id
            assert row.interval_kind and row.verdict, row.id


def test_the_recorded_digest_is_the_digest_of_the_file_at_the_source_path(retained, tmp_path):
    def assert_current(ledger: Ledger, root) -> None:
        digests = {}
        for row in ledger.quantities:
            if row.source_path not in digests:
                digests[row.source_path] = hashlib.sha256(
                    (root / row.source_path).read_bytes()).hexdigest()
            assert row.source_sha256 == digests[row.source_path], row.id
        assert set(digests) <= set(ARTIFACTS)

    assert_current(retained, REPO_ROOT)
    # A row follows the bytes it was read from: changing one leaf of one artifact
    # moves that row's value and that file's digest, and no other file's.
    leaf = "scoring/variant_sequences"
    before = rows(retained)[f"{matched.FAMILY}/expanded/{leaf}"]
    moved = stage(tmp_path, matched.COHORT, lambda payload: payload["scoring_cost"].__setitem__(
        "unique_variant_sequences", payload["scoring_cost"]["unique_variant_sequences"] + 1))
    after = rows(moved)[f"{matched.FAMILY}/expanded/{leaf}"]
    assert after.value == before.value + 1
    assert after.source_sha256 != before.source_sha256
    assert after.source_sha256 == hashlib.sha256(
        (tmp_path / matched.COHORT).read_bytes()).hexdigest()
    assert_current(moved, tmp_path)
    assert {row.id: row.source_sha256 for row in moved.quantities
            if row.source_path == matched.EXPANDED} == {
        row.id: row.source_sha256 for row in retained.quantities
        if row.source_path == matched.EXPANDED}


def test_every_quoted_span_is_a_min_max_pair_over_one_unit(retained):
    emitted = rows(retained)
    bounds = [identifier for identifier, row in emitted.items() if row.kind == "range_bound"]
    assert bounds
    for identifier in bounds:
        stem, bound = identifier.rsplit("/", 1)
        assert bound in ("min", "max"), identifier
        low, high = emitted[f"{stem}/min"], emitted[f"{stem}/max"]
        assert low.value <= high.value
        # The ends of a span over the five tiers sit on five supports, by design:
        # a tier is a support. What they share is the unit the span is quoted in.
        assert low.unit == high.unit, stem


# --------------------------------------------------------------------------- #
# The masked arm.
# --------------------------------------------------------------------------- #

def test_the_masked_arm_never_enters_an_autoregressive_aggregate(retained):
    expanded = load(matched.EXPANDED)
    autoregressive = sorted(arm for arm, record in expanded["arms"].items()
                            if record["pooled_with_autoregressive_arms"])
    assert len(autoregressive) == 8 and MASKED not in autoregressive

    def margin(arm):
        return arm["contrasts"]["D_model_minus_D_position_T_double_balanced"]["difference"]

    def quintile(index):
        return lambda arm: arm["tiers"]["T_match"]["magnitude_stratified"]["bins"][index]["accuracy"]

    spans = {
        "T_double_balanced/model_minus_position_span": margin,
        "T_match/autoregressive_magnitude_q1_span": quintile(0),
        "T_match/autoregressive_magnitude_q5_span": quintile(4),
    }
    emitted = rows(retained)
    for stem, read in spans.items():
        values = {arm: read(expanded["arms"][arm]) for arm in autoregressive}
        for bound, pick in (("min", min), ("max", max)):
            row = emitted[f"{matched.FAMILY}/expanded/{stem}/{bound}"]
            arm = row.source_pointer[1]
            assert arm in autoregressive, f"{stem}/{bound} takes its bound from {arm}"
            assert row.value == values[arm] == pick(values.values())
    # The masked arm is the highest of two of those spans' members, so a leak would
    # have moved the bound the Results quote rather than left it where it was.
    for stem in ("T_double_balanced/model_minus_position_span",
                 "T_match/autoregressive_magnitude_q5_span"):
        top = emitted[f"{matched.FAMILY}/expanded/{stem}/max"].value
        assert spans[stem](expanded["arms"][MASKED]) > top, stem
    # Its own span is over its own five quintiles and nothing else, and every
    # accuracy row of it says it is not pooled.
    own = f"{matched.FAMILY}/expanded/T_match/esm2-650m/magnitude_span"
    accuracies = [quintile(index)(expanded["arms"][MASKED]) for index in range(5)]
    for bound, pick in (("min", min), ("max", max)):
        row = emitted[f"{own}/{bound}"]
        assert row.source_pointer[:2] == ("arms", MASKED)
        assert row.value == pick(accuracies)
    masked_rows = [row for identifier, row in emitted.items()
                   if identifier.endswith("/esm2-650m/direction_accuracy")]
    assert len(masked_rows) == len(matched.TIER_ORDER)
    for row in masked_rows:
        assert row.source_pointer[1] == MASKED and "never pooled with them" in row.claim


@pytest.mark.parametrize("arm, pooled", [(MASKED, True), ("progen2-large", False)])
def test_an_arm_the_artifact_pools_differently_from_the_roster_refuses(tmp_path, arm, pooled):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, matched.EXPANDED, lambda payload: payload["arms"][arm].__setitem__(
            "pooled_with_autoregressive_arms", pooled))
    message = str(refusal.value)
    assert "pooled_with_autoregressive_arms" in message and "disagrees with the roster" in message


# --------------------------------------------------------------------------- #
# What an artifact has to say before a row is written.
# --------------------------------------------------------------------------- #

def moved_decisive_tier(payload):
    payload["decisive_tier"] = "T_balanced"


def verdict_outside_the_declaration(payload):
    payload["arms"]["progen2-small"]["tiers"]["T_open"]["evaluation"]["verdict"] = "equivalent"


def a_vote_that_is_not_cancelled(payload):
    certified = payload["balanced_tier_construction"]["T_double_balanced"]["balance_per_stratum"][
        "evaluation"]["certified_direction_accuracy"]
    certified["position"] = 0.51


def a_margin_that_covers_zero(payload):
    payload["arms"]["progen3-112m"]["contrasts"]["D_model_minus_D_position_T_double_balanced"][
        "excludes_zero"] = False


def a_capacity_axis_declared_after_the_fact(payload):
    payload["capacity_axis"]["T_double_balanced"]["status"] = "retrospective"


def another_resampling_unit(payload):
    payload["inference"]["resampling_unit"] = "assay"


def a_protein_arm_that_is_a_bounded_negative(payload):
    payload["E21"]["arms"]["protgpt2"]["D_open"]["across_seeds"]["verdict"] = "bounded_negative"


@pytest.mark.parametrize("artifact, mutate, message", [
    pytest.param(matched.EXPANDED, moved_decisive_tier, "the decisive tier is",
                 id="decisive-tier"),
    pytest.param(matched.EXPANDED, verdict_outside_the_declaration,
                 "outside the declaration's three outcomes", id="verdict"),
    pytest.param(matched.EXPANDED, a_vote_that_is_not_cancelled, "not cancelled exactly",
                 id="certified-vote"),
    pytest.param(matched.EXPANDED, a_margin_that_covers_zero, "does not exclude zero",
                 id="margin"),
    pytest.param(matched.EXPANDED, a_capacity_axis_declared_after_the_fact,
                 "reports as prospective", id="capacity-axis"),
    pytest.param(matched.PANEL, another_resampling_unit, "not over one kind of unit",
                 id="resampling-unit"),
    pytest.param(matched.PANEL, a_protein_arm_that_is_a_bounded_negative,
                 "are themselves bounded negatives", id="protein-arm"),
])
def test_an_artifact_that_disagrees_with_what_a_row_would_claim_is_refused(
        tmp_path, artifact, mutate, message):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, artifact, mutate)
    assert message in str(refusal.value)


# --------------------------------------------------------------------------- #
# The verdict counts.
# --------------------------------------------------------------------------- #

def contrast_verdicts(panel: dict, tier: str) -> dict:
    return {arm: record["contrasts"][f"{tier}/R_seqprof_minus_R_like"]["across_seeds"]
            for arm, record in panel["E22"]["arms"].items()}


def test_a_verdict_count_is_read_off_the_verdicts_and_resolved_up_is_not_down(retained):
    panel = load(matched.PANEL)
    emitted = rows(retained)
    matched_tier = contrast_verdicts(panel, "T_match")
    unresolved = [arm for arm, record in matched_tier.items() if record["verdict"] == "unresolved"]
    downward = [arm for arm, record in matched_tier.items()
                if record["verdict"] == "supported" and record["direction"] == "below"]
    # The matched tier holds an arm that resolves downward, so the count of arms
    # that resolve no gain is not the count of arms that fail to resolve upward.
    assert downward and len(unresolved) + len(downward) == len(matched_tier)
    open_tier = contrast_verdicts(panel, "T_open")
    upward = [arm for arm, record in open_tier.items()
              if record["verdict"] == "supported" and record["direction"] == "above"]
    bounded = [arm for arm, record in panel["E21"]["arms"].items()
               if record["D_open"]["across_seeds"]["verdict"] == "bounded_negative"]
    expected = {
        "T_match/seqprof_minus_likelihood/unresolved_arms": len(unresolved),
        "T_open/seqprof_minus_likelihood/resolved_above_arms": len(upward),
        "T_open/native_likelihood/bounded_negative_arms": len(bounded),
    }
    for stem, count in expected.items():
        row = emitted[f"{COUNTS}/{stem}"]
        assert row.value == count and row.kind == "count" and row.unit == "checkpoints", stem
        # The row points at one of the verdicts it counts, which keeps it from drifting.
        assert resolve(panel, row.source_pointer) == (
            "unresolved" if "unresolved" in stem else
            "supported" if "resolved_above" in stem else "bounded_negative")
    pointer = emitted[f"{COUNTS}/T_open/seqprof_minus_likelihood/resolved_above_arms"].source_pointer
    assert resolve(panel, pointer[:-1] + ("direction",)) == "above"


def test_an_arm_resolving_downward_is_not_counted_as_resolving_upward(retained, tmp_path):
    panel = load(matched.PANEL)
    arms = panel["E22"]["arms"]
    open_tier = contrast_verdicts(panel, "T_open")
    upward = [arm for arm, record in open_tier.items()
              if record["verdict"] == "supported" and record["direction"] == "above"]

    def floor(arm):
        points = arms[arm]["levels"]["T_open/R_like"]["across_seeds"]["points"]
        return min(float(point) for point in points.values())

    # The arm with the strongest native likelihood among those that resolve a gain:
    # turning it downward leaves the separation the module also guards intact, so
    # the only thing that can move is the count.
    flipped = max(upward, key=floor)

    def resolve_downward(payload):
        record = payload["E22"]["arms"][flipped]["contrasts"]["T_open/R_seqprof_minus_R_like"]
        assert record["across_seeds"]["verdict"] == "supported"
        record["across_seeds"]["direction"] = "below"

    ledger = stage(tmp_path, matched.PANEL, resolve_downward)
    emitted = rows(ledger)
    assert emitted[f"{COUNTS}/T_open/seqprof_minus_likelihood/resolved_above_arms"].value == \
        len(upward) - 1
    # The arm is still a resolved contrast and so is in no unresolved count.
    unresolved = f"{COUNTS}/T_match/seqprof_minus_likelihood/unresolved_arms"
    assert emitted[unresolved].value == rows(retained)[unresolved].value


def test_a_gain_the_native_likelihood_does_not_bind_is_refused(tmp_path):
    panel = load(matched.PANEL)
    arms = panel["E22"]["arms"]
    held = [arm for arm, record in contrast_verdicts(panel, "T_open").items()
            if record["verdict"] != "supported"]
    # The held arm whose native likelihood is strongest, made to resolve a gain
    # too: the arms supervision helps are then no longer the weaker ones.
    strongest = max(held, key=lambda arm: min(float(point) for point in arms[arm]["levels"][
        "T_open/R_like"]["across_seeds"]["points"].values()))

    def resolves_a_gain(payload):
        record = payload["E22"]["arms"][strongest]["contrasts"]["T_open/R_seqprof_minus_R_like"]
        record["across_seeds"].update(verdict="supported", direction="above")

    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, matched.PANEL, resolves_a_gain)
    assert "lower native-likelihood accuracy" in str(refusal.value)
