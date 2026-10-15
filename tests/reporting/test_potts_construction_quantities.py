"""The two construction diagnostics Supplementary Note 14 quotes, and their refusals.

Note 14 reads both diagnostics as reductions over a declared set: the gauge
transformation's size over every qualified family fit, and the distance between
the two samplers over the development domains. Neither artifact names its set, so
a diagnostic that does not cover its declared roster is a refusal rather than a
quieter number, and an aggregate the artifact records is read against the
per-domain records it reduces --- including the direction of the reduction, which
is what the Note's "at most" and "at least" state.

The sampler comparison records three readings of one comparison: a gap between
two means, that gap in units of its own standard error, and the largest
per-position distance between the two sampling distributions. Each is recorded
beside what it reduces, so each is checked against it: a standard-error count
taken over another standard error would make the Note's two quoted spans spans of
two different comparisons.

The last test runs on the retained artifacts and asserts the numbers the Note
quotes, at the precision it quotes them to.
"""
import json

import pytest

from scripts.capability.reporting import potts_construction_quantities as potts
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError

#: Per-fit gauge readings, chosen so that each aggregate is attained by a
#: different fit: a reduction read off the wrong record would then disagree.
FITS = {
    "1E0L": (0.0020, 0.99933, 7.1e-15),
    "1F0M": (0.0013, 0.99948, 7.1e-15),
    "1IFY": (0.0014, 0.99965, 8.8e-15),
    "1K1V": (0.0017, 0.99960, 7.9e-15),
    "1UFM": (0.0013, 0.99959, 1.0658e-14),
    "1V1C": (0.0067, 0.99959, 1.7e-15),
    "2K1B": (0.0012, 0.99954, 8.2e-15),
    "2KWH": (0.0028, 0.99944, 5.3e-15),
    "2LGW": (0.0067714, 0.9988330, 4.4e-15),
    "2LVN": (0.0024, 0.99953, 7.1e-15),
    "3DKM": (0.0014, 0.99937, 6.2e-15),
    "3L1X": (0.0009, 0.99964, 7.9e-15),
}
#: Per-domain sampler readings as (joint mean, joint se, prefix mean, prefix se,
#: largest per-position distance). The gap and its standard-error count are
#: derived from them, so the staged payload is internally consistent by
#: construction and a test can break exactly one tie.
DRAWS = {
    "1K1V": (155.47247678193898, 0.12615908305617135, 144.04193672255155, 0.15422480561279445,
             0.3165),
    "1UFM": (200.0, 0.2, 157.693469794615424, 0.21, 0.55825),
    "2KWH": (180.0, 0.17, 163.39741117938189, 0.18, 0.25975),
    "3L1X": (260.0, 0.2, 210.42556289975221, 0.2, 0.679),
}


def gauge_payload(*, fits: dict | None = None, aggregate: dict | None = None) -> dict:
    fits = FITS if fits is None else fits
    records = [{"domain": domain, "fraction_of_J_norm_removable_by_gauge": fraction,
                "coupling_bias_cosine_median": cosine, "energy_shift_spread": spread}
               for domain, (fraction, cosine, spread) in sorted(fits.items())]
    recorded = {
        "max_fraction_of_J_norm_removable_by_gauge":
            max(entry["fraction_of_J_norm_removable_by_gauge"] for entry in records),
        "min_coupling_bias_cosine_median":
            min(entry["coupling_bias_cosine_median"] for entry in records),
        "max_energy_shift_spread": max(entry["energy_shift_spread"] for entry in records),
    }
    recorded.update(aggregate or {})
    return {"amino_acid_order": "ACDEFGHIKLMNPQRSTVWY", "domains": records,
            "aggregate": recorded}


def sampler_payload(*, draws: dict | None = None, schema: str = potts.SAMPLER_SCHEMA,
                    samplers: tuple[str, ...] = potts.SAMPLERS,
                    overrides: dict | None = None) -> dict:
    draws = DRAWS if draws is None else draws
    records = []
    for domain, (joint, joint_se, prefix, prefix_se, distance) in sorted(draws.items()):
        gap = joint - prefix
        combined = (joint_se ** 2 + prefix_se ** 2) ** 0.5
        record = {
            "domain": domain,
            "energy": {"potts_joint": {"mean": joint, "standard_error": joint_se},
                       "prefix_chain": {"mean": prefix, "standard_error": prefix_se}},
            "energy_gap_joint_minus_prefix": gap,
            "energy_gap_in_standard_errors": gap / combined,
            "between_sampler_tv": {"max": distance,
                                   "per_column": [distance / 2, distance, distance / 3]},
        }
        record.update((overrides or {}).get(domain, {}))
        records.append(record)
    return {"schema": schema, "samplers": {name: f"stub {name}" for name in samplers},
            "domains": records, "draws_per_sampler_per_replicate": 2000, "replicates": 2,
            "gibbs": {"sweeps": 200}, "seed": 20261013,
            "seed_scheme": "SeedSequence([seed, domain_index, sampler_index, replicate])"}


def stage(tmp_path, *, gauge: dict | None = None, sampler: dict | None = None) -> Ledger:
    payloads = {potts.GAUGE: gauge if gauge is not None else gauge_payload(),
                potts.SAMPLER: sampler if sampler is not None else sampler_payload()}
    for relative, payload in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    ledger = Ledger(Artifacts(tmp_path))
    potts.add_potts_construction_quantities(ledger)
    return ledger


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


def ledger_supports(ledger: Ledger) -> dict:
    return {key: value.description for key, value in ledger.supports.items()}


def test_every_quoted_reading_is_a_bound_and_the_two_spans_are_pairs(tmp_path):
    emitted = rows(stage(tmp_path))
    # Neither diagnostic resamples anything, so no row may claim to be a point
    # estimate the manuscript owes an interval for.
    for quantity in emitted.values():
        assert quantity.kind == "range_bound"
        assert quantity.interval is None and quantity.level == "generation"
    # The spans the Note writes as ``a--b`` are the min/max pairs the consistency
    # checker matches a quoted range against.
    for prefix in ("potts_sampler/energy_gap_joint_minus_prefix",
                   "potts_sampler/energy_gap_in_standard_errors"):
        low, high = emitted[f"{prefix}/min"], emitted[f"{prefix}/max"]
        assert low.value < high.value
        assert low.unit == high.unit and low.support_id == high.support_id


def test_the_procedural_constants_and_the_unquoted_share_are_not_emitted(tmp_path):
    emitted = rows(stage(tmp_path))
    values = {quantity.value for quantity in emitted.values()}
    # Draws per sampler and replicate, replicates, Gibbs sweeps and the seed.
    assert not values & {2000.0, 2.0, 200.0, 20261013.0}
    # The largest gauge-carried first-order share is recorded and the Note states
    # only its direction, so no row quotes a value for it.
    assert not [identifier for identifier in emitted if "first_order" in identifier]
    # Those constants define the populations, so they travel on the supports.
    support = ledger_supports(stage(tmp_path))["e20_potts_sampler_comparison_development"]
    for constant in ("2000", "200", "20261013"):
        assert constant in support


def test_a_diagnostic_short_of_its_declared_roster_refuses(tmp_path):
    thinned = {domain: values for domain, values in FITS.items() if domain != "3L1X"}
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, gauge=gauge_payload(fits=thinned))
    assert "not the 12 qualified fits" in str(refusal.value)

    short = {domain: values for domain, values in DRAWS.items() if domain != "2KWH"}
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, sampler=sampler_payload(draws=short))
    assert "not the 4 development domains" in str(refusal.value)


def test_an_aggregate_that_is_not_the_reduction_it_records_refuses(tmp_path):
    # A largest share read off a fit that is not the largest.
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, gauge=gauge_payload(
            aggregate={"max_fraction_of_J_norm_removable_by_gauge": 0.0067}))
    assert "so there is no one reading to quote" in str(refusal.value)
    # A smallest cosine recorded as the largest one: the direction of the
    # reduction is what "at least" states, so reversing it is a refusal and not
    # a different number.
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, gauge=gauge_payload(aggregate={
            "min_coupling_bias_cosine_median":
                max(cosine for _, cosine, _ in FITS.values())}))
    assert "so there is no one reading to quote" in str(refusal.value)


def test_another_schema_or_another_pair_of_samplers_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, sampler=sampler_payload(schema="potts_sampler_comparison_v2"))
    assert f"not a {potts.SAMPLER_SCHEMA} comparison" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, sampler=sampler_payload(samplers=("potts_joint", "gibbs_chain")))
    assert "a difference between two other samplers" in str(refusal.value)


def test_the_three_sampler_readings_have_to_be_readings_of_one_comparison(tmp_path):
    # A gap that is not the difference of the two recorded means.
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, sampler=sampler_payload(
            overrides={"3L1X": {"energy_gap_joint_minus_prefix": 44.0}}))
    assert "so there is no one difference to quote" in str(refusal.value)
    # A standard-error count taken over some other standard error.
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, sampler=sampler_payload(
            overrides={"1K1V": {"energy_gap_in_standard_errors": 90.0}}))
    assert "not readings of one comparison" in str(refusal.value)
    # A largest per-position distance that is not the largest of them.
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, sampler=sampler_payload(
            overrides={"2KWH": {"between_sampler_tv": {"max": 0.4,
                                                       "per_column": [0.1, 0.25975]}}}))
    assert "and the largest of its" in str(refusal.value)


def test_the_retained_diagnostics_carry_the_numbers_the_note_quotes():
    artifacts = Artifacts()
    for relative in (potts.GAUGE, potts.SAMPLER):
        if not artifacts.exists(relative):
            pytest.skip("the E20 gauge and sampler diagnostics are not staged in this checkout")
    ledger = Ledger(artifacts)
    potts.add_potts_construction_quantities(ledger)
    emitted = rows(ledger)

    # "at most 0.68\% of the coupling norm is removable by gauge": the diagnostic
    # records a fraction and the Note quotes it as a percent.
    share = emitted["potts_gauge/largest_coupling_norm_share_removable_by_gauge"]
    assert share.unit == "percent" and round(share.value, 2) == 0.68
    # "a cosine of at least 0.9988".
    assert round(emitted["potts_gauge/smallest_coupling_bias_cosine"].value, 4) == 0.9988
    # "energies are preserved to 1.1e-14", quoted to two significant figures.
    spread = emitted["potts_gauge/largest_energy_shift_spread"].value
    assert float(f"{spread:.1e}") == 1.1e-14
    # "the two differ by 11--50 energy units, between 57 and 173 standard errors".
    assert round(emitted["potts_sampler/energy_gap_joint_minus_prefix/min"].value) == 11
    assert round(emitted["potts_sampler/energy_gap_joint_minus_prefix/max"].value) == 50
    assert round(emitted["potts_sampler/energy_gap_in_standard_errors/min"].value) == 57
    assert round(emitted["potts_sampler/energy_gap_in_standard_errors/max"].value) == 173
    # "per-position total-variation distances reaching 0.68".
    assert round(emitted["potts_sampler/largest_per_position_total_variation"].value, 2) == 0.68
