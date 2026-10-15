"""Source mappings for the two construction diagnostics Supplementary Note 14 quotes.

Note 14 declares the guided-decoding protocol, and two of its paragraphs rest on
measurement rather than declaration. The gauge paragraph reports how large the
conversion to the zero-sum gauge is, because the Frobenius norm of a coupling
block is not gauge invariant: in a gauge whose coupling row means do not vanish a
couplings-only arm injects first-order information, and a positive result would
be confounded. The sampler paragraph reports how far the local Potts energy added
during decoding is from the prefix conditional of the Potts joint, because the
two are different objects and the Note separates them with numbers rather than
with an assertion.

Every row here is read out of one of those two retained diagnostics. Nothing is
fitted, nothing is resampled and no interval is copied, because neither
diagnostic carries one: each reading is a reduction over a declared set of family
fits or of development domains, so it is emitted as a ``range_bound`` and never
as a point estimate the manuscript would owe an interval for. An aggregate the
artifact records is read against the per-domain records it reduces and refuses on
disagreement: a reduction over a set is one reading only while it agrees with the
set, and the direction of the reduction --- a largest share, a smallest cosine
--- is what the Note's "at most" and "at least" state.

Two rosters carry what the artifacts do not name. ``QUALIFIED_FITS`` is the
twelve-domain reference set every family fit of this construction is qualified on,
and ``DEVELOPMENT_DOMAINS`` is its four-domain development end, disjoint from the
eight sealed evaluation domains the frozen confirmatory analysis declares. Each
artifact lists the domains it happened to compute and names neither set, so a
diagnostic that does not cover its declared roster is a refusal rather than a
quieter number: "every qualified fit" over eleven fits, and "over the development
domains" over three of four, are different claims.

The spans the Note writes as ``a--b`` are emitted as the ``min``/``max`` pair the
consistency checker matches a quoted range against, so the energy gap and its
standard-error count are verified as spans rather than exempted.

The procedural constants these diagnostics record are not emitted: the draw count
per sampler and replicate, the replicate count, the Gibbs sweep count and the
seed. No claim quotes any of them as a numeral, and each collides with numbers
the manuscript writes for other things, so a row for any of them could only
dilute the attribution of a number it does quote. They travel on the support
descriptions instead, which is where the population they define belongs. The
largest gauge-carried first-order share is recorded and is not emitted either:
the Note states its direction and quotes no value for it.
"""
from __future__ import annotations

from math import isclose

from src.capability.core.evidence_ledger import LedgerError
# The emit contract --- how a row carries its locator, digest and pointer --- is
# the follow-up module's, and the ``<prefix>/min``--``<prefix>/max`` identifier
# pair a quoted range is matched against is the propagation module's. A second
# copy of either here would be a second place for it to drift.
from scripts.capability.reporting.followup_quantities import _emitter
from scripts.capability.reporting.propagation_quantities import _span

GAUGE = "results/R6/e20_potts_gauge_20261013/gauge_diagnostic.json"
SAMPLER = "results/R6/e20_potts_sampler_comparison_20261013/sampler_comparison.json"

#: The twelve-domain reference set every family fit of this construction is
#: qualified on. No artifact field names it: the gauge diagnostic lists the
#: domains it computed, which is the same twelve only while nothing has been
#: dropped.
QUALIFIED_FITS = ("1E0L", "1F0M", "1IFY", "1K1V", "1UFM", "1V1C", "2K1B", "2KWH", "2LGW", "2LVN",
                  "3DKM", "3L1X")
#: The development end of that set, disjoint from the eight sealed evaluation
#: domains the frozen confirmatory analysis declares. The Note's sampler
#: paragraph reads "over the development domains", which is exactly these four.
DEVELOPMENT_DOMAINS = ("1K1V", "1UFM", "2KWH", "3L1X")

#: The two samplers the energy difference is between. The comparison exists to
#: keep them apart, so a record carrying one of them under another name is not
#: the difference the Note quotes.
SAMPLERS = ("potts_joint", "prefix_chain")
SAMPLER_SCHEMA = "potts_sampler_comparison_v1"

#: The scale a full-sequence Potts energy is read on. It is a log-density in nats
#: only up to the model's own normalisation, which this construction never
#: computes, so the unit says what the Note says --- energy units --- rather than
#: borrowing a measured or an information unit.
ENERGY = "Potts energy units"
#: A difference divided by its own standard error. It is dimensionless, and
#: naming it that way would lose what makes "between 57 and 173" readable.
STANDARD_ERRORS = "standard errors"
COSINE = "dimensionless cosine"
TOTAL_VARIATION = "dimensionless total-variation distance"

#: Each aggregate reading the Note quotes, with the per-domain field it reduces
#: and the direction of the reduction. The direction is what "at most" and "at
#: least" state, so it is read against the records rather than trusted.
GAUGE_AGGREGATES = {
    "max_fraction_of_J_norm_removable_by_gauge": ("fraction_of_J_norm_removable_by_gauge", max),
    "min_coupling_bias_cosine_median": ("coupling_bias_cosine_median", min),
    "max_energy_shift_spread": ("energy_shift_spread", max),
}


def add_potts_construction_quantities(ledger) -> None:
    emit = _emitter(ledger)
    _gauge_transformation(ledger, emit)
    _sampler_difference(ledger, emit)


# --------------------------------------------------------------------------- #
# The gauge: how much of a couplings-only intervention the gauge could carry.
# --------------------------------------------------------------------------- #

def _gauge_fits(ledger) -> dict[str, dict]:
    """The qualified fits, keyed by domain, or a refusal."""
    payload = ledger.artifacts.json(GAUGE)
    fits = {record["domain"]: record for record in payload["domains"]}
    if len(fits) != len(payload["domains"]):
        raise LedgerError(f"{GAUGE}: the fit roster repeats a domain")
    if tuple(sorted(fits)) != tuple(sorted(QUALIFIED_FITS)):
        raise LedgerError(
            f"{GAUGE}: the diagnostic covers {sorted(fits)} and not the {len(QUALIFIED_FITS)} "
            f"qualified fits {sorted(QUALIFIED_FITS)}; a reduction over a different set of fits is "
            "not the one the Note reports")
    return fits


def _gauge_support(ledger, fits: dict[str, dict]) -> str:
    payload = ledger.artifacts.json(GAUGE)
    return ledger.declare_support(
        "e20_potts_gauge_qualified_fits",
        f"the {len(fits)} qualified family Potts fits of the guided-decoding construction "
        f"({', '.join(sorted(fits))}), one per domain of the twelve-domain reference set, over the "
        f"{len(payload['amino_acid_order'])}-letter alphabet the fits are parameterised in; each "
        "fit is converted to the zero-sum gauge and the size of that transformation is recorded, so "
        "every row is a property of the parameterisation and not a measurement of any generated "
        "candidate")


def _agreeing_aggregate(fits: dict[str, dict], field: str) -> tuple[str, float]:
    """One recorded aggregate, read against the per-domain records it reduces.

    The aggregate and the reduction over the records are one reading only while
    they agree, so a disagreement refuses rather than picking one, and the
    direction of the reduction is read rather than assumed: "at most" and "at
    least" are the Note's own words for it.
    """
    per_domain, reduce = GAUGE_AGGREGATES[field]
    values = {domain: float(record[per_domain]) for domain, record in fits.items()}
    chosen = reduce(values, key=values.get)
    return chosen, values[chosen]


def _gauge_transformation(ledger, emit) -> None:
    fits = _gauge_fits(ledger)
    support = _gauge_support(ledger, fits)
    payload = ledger.artifacts.json(GAUGE)
    aggregate = payload["aggregate"]
    for field in GAUGE_AGGREGATES:
        chosen, value = _agreeing_aggregate(fits, field)
        if not isclose(float(aggregate[field]), value, rel_tol=1e-12, abs_tol=0.0):
            raise LedgerError(
                f"{GAUGE}: the recorded {field} is {aggregate[field]} and the same reduction over "
                f"the {len(fits)} fits is {value} ({chosen}), so there is no one reading to quote")

    chosen, fraction = _agreeing_aggregate(fits, "max_fraction_of_J_norm_removable_by_gauge")
    emit(GAUGE, ("aggregate", "max_fraction_of_J_norm_removable_by_gauge", "<as a percentage>"),
         identifier="potts_gauge/largest_coupling_norm_share_removable_by_gauge",
         claim="largest share of a fit's coupling-block Frobenius norm that the conversion to the "
               f"zero-sum gauge removes, over the {len(fits)} qualified fits ({chosen}); the "
               "diagnostic records it as a fraction and this is that fraction written as a percent "
               "(multiplied by 100), which is how the Note quotes it. It bounds how much of a "
               "couplings-only intervention's magnitude the gauge could have carried",
         family="potts_gauge", support=support, unit="percent", kind="range_bound",
         value=100.0 * fraction, level="generation")

    chosen, cosine = _agreeing_aggregate(fits, "min_coupling_bias_cosine_median")
    emit(GAUGE, ("aggregate", "min_coupling_bias_cosine_median"),
         identifier="potts_gauge/smallest_coupling_bias_cosine",
         claim="smallest per-fit median cosine between the realised coupling bias before and after "
               f"the conversion to the zero-sum gauge, over the {len(fits)} qualified fits "
               f"({chosen}): how far the conversion turns the direction a couplings-only "
               "intervention pushes in. It is a minimum over the per-fit medians, not over the "
               "per-position cosines, whose own minimum the diagnostic records separately and "
               "lower",
         family="potts_gauge", support=support, unit=COSINE, kind="range_bound", value=cosine,
         level="generation")

    chosen, spread = _agreeing_aggregate(fits, "max_energy_shift_spread")
    emit(GAUGE, ("aggregate", "max_energy_shift_spread"),
         identifier="potts_gauge/largest_energy_shift_spread",
         claim="largest spread, across sequences, of the full-sequence energy shift the conversion "
               f"to the zero-sum gauge introduces, over the {len(fits)} qualified fits ({chosen}); "
               "the conversion preserves energy differences by construction, so this is the "
               "floating-point residual of that identity and the sense in which full-sequence "
               "energies are preserved",
         family="potts_gauge", support=support, unit=ENERGY, kind="range_bound", value=spread,
         level="generation")


# --------------------------------------------------------------------------- #
# The sampler: a prefix chain is not the Potts joint's prefix conditional.
# --------------------------------------------------------------------------- #

def _sampler_domains(ledger) -> dict[str, dict]:
    """The development domains the comparison is run on, or a refusal."""
    payload = ledger.artifacts.json(SAMPLER)
    if payload["schema"] != SAMPLER_SCHEMA:
        raise LedgerError(f"{SAMPLER}: not a {SAMPLER_SCHEMA} comparison")
    if tuple(sorted(payload["samplers"])) != tuple(sorted(SAMPLERS)):
        raise LedgerError(
            f"{SAMPLER}: the comparison is between {sorted(payload['samplers'])} and not the "
            f"declared {sorted(SAMPLERS)}; a difference between two other samplers is a different "
            "statement")
    records = {record["domain"]: record for record in payload["domains"]}
    if len(records) != len(payload["domains"]):
        raise LedgerError(f"{SAMPLER}: the domain roster repeats a domain")
    if tuple(sorted(records)) != tuple(sorted(DEVELOPMENT_DOMAINS)):
        raise LedgerError(
            f"{SAMPLER}: the comparison covers {sorted(records)} and not the "
            f"{len(DEVELOPMENT_DOMAINS)} development domains {sorted(DEVELOPMENT_DOMAINS)}; a "
            "reduction over a different set of domains is not the one the Note reports")
    for domain, record in sorted(records.items()):
        _consistent_difference(domain, record)
    return records


def _consistent_difference(domain: str, record: dict) -> None:
    """The recorded difference, read against the draws it is a difference of.

    The gap, its standard-error count and the between-sampler distance are each
    recorded beside what they reduce --- the two samplers' means and standard
    errors, and the per-position distances --- so each is one reading only while
    it agrees with that. The Note quotes the gap and the standard-error count as
    two readings of one comparison, which holds only if the count is that gap
    over its own standard error and not over another.
    """
    energy = record["energy"]
    joint, prefix = energy["potts_joint"], energy["prefix_chain"]
    gap = joint["mean"] - prefix["mean"]
    if not isclose(gap, record["energy_gap_joint_minus_prefix"], rel_tol=1e-9, abs_tol=0.0):
        raise LedgerError(
            f"{SAMPLER}: {domain} records an energy gap of "
            f"{record['energy_gap_joint_minus_prefix']} and the two samplers' recorded means "
            f"differ by {gap}, so there is no one difference to quote")
    standard_error = (joint["standard_error"] ** 2 + prefix["standard_error"] ** 2) ** 0.5
    if not isclose(gap / standard_error, record["energy_gap_in_standard_errors"],
                   rel_tol=1e-9, abs_tol=0.0):
        raise LedgerError(
            f"{SAMPLER}: {domain} records the gap as {record['energy_gap_in_standard_errors']} "
            f"standard errors and the recorded gap over the recorded standard errors is "
            f"{gap / standard_error}, so the two quoted readings are not readings of one "
            "comparison")
    distances = record["between_sampler_tv"]
    if not isclose(max(distances["per_column"]), distances["max"], rel_tol=1e-12, abs_tol=0.0):
        raise LedgerError(
            f"{SAMPLER}: {domain} records a largest per-position total-variation distance of "
            f"{distances['max']} and the largest of its {len(distances['per_column'])} per-position "
            f"distances is {max(distances['per_column'])}")


def _sampler_support(ledger, records: dict[str, dict]) -> str:
    payload = ledger.artifacts.json(SAMPLER)
    return ledger.declare_support(
        "e20_potts_sampler_comparison_development",
        f"the {len(records)} development domains of the guided-decoding construction "
        f"({', '.join(sorted(records))}), disjoint from the eight sealed evaluation domains; per "
        f"domain, {payload['draws_per_sampler_per_replicate']} chains per sampler per replicate "
        f"over {payload['replicates']} replicates, drawn from the prefix-truncated conditionals and "
        f"from the Potts joint by Gibbs sampling at {payload['gibbs']['sweeps']} sweeps, seeded as "
        f"{payload['seed_scheme']} from {payload['seed']}; the endpoints are the full-sequence "
        "Potts energy of a drawn sequence and the per-position amino-acid distribution the draws "
        "induce, both properties of the sampler and not of any generated candidate")


def _sampler_difference(ledger, emit) -> None:
    records = _sampler_domains(ledger)
    support = _sampler_support(ledger, records)
    _span(emit, SAMPLER, ("domains", "<per domain>", "energy_gap_joint_minus_prefix"),
          prefix="potts_sampler/energy_gap_joint_minus_prefix",
          claim=f"mean full-sequence Potts energy of a Gibbs draw from the Potts joint minus that "
                f"of a draw from the prefix-truncated conditionals, across the {len(records)} "
                "development domains: what sampling a chain from the truncated conditionals costs "
                "against sampling the joint it is not the prefix conditional of",
          family="potts_sampler", support=support, unit=ENERGY,
          values={domain: float(record["energy_gap_joint_minus_prefix"])
                  for domain, record in records.items()},
          level="generation")
    _span(emit, SAMPLER, ("domains", "<per domain>", "energy_gap_in_standard_errors"),
          prefix="potts_sampler/energy_gap_in_standard_errors",
          claim=f"the same energy gap in units of its own standard error, across the "
                f"{len(records)} development domains: the gap over the combined standard error of "
                "the two samplers' means, which is what makes the difference between the two "
                "samplers large rather than merely nonzero",
          family="potts_sampler", support=support, unit=STANDARD_ERRORS,
          values={domain: float(record["energy_gap_in_standard_errors"])
                  for domain, record in records.items()},
          level="generation")

    distances = {domain: float(record["between_sampler_tv"]["max"])
                 for domain, record in records.items()}
    chosen = max(distances, key=distances.get)
    emit(SAMPLER, ("domains", "<per domain>", "between_sampler_tv", "max", "<max>"),
         identifier="potts_sampler/largest_per_position_total_variation",
         claim="largest total-variation distance between the two samplers' amino-acid "
               f"distributions at any one position, over the {len(records)} development domains "
               f"({chosen}); it is a distance between two sampling distributions at a position, "
               "not a distance between sequences",
         family="potts_sampler", support=support, unit=TOTAL_VARIATION, kind="range_bound",
         value=distances[chosen], level="generation")
