"""Single- and double-mutation landscapes of *generated* proteins, and the
model-side companion of the natural-protein contact endpoint.

Three questions share this module because they share one measurement. All three
read a per-position log likelihood out of
:mod:`src.capability.position.position_likelihood` -- the project's single
producer of position-resolved likelihood -- and all three reduce it with the
estimators this project already declared. Nothing here opens a model, and nothing
here re-implements a bootstrap, a contact definition, a separation stratum or a
matched contrast.

E15 -- do the mutation-related patterns a model shows on natural proteins also
hold on proteins it generated? The measurement is a seeded single-substitution
scan of a generated sequence and of a *length-matched natural* sequence, scored
by the same arm in the same run, and the comparison is the one E01 made: the
site's own term, the downstream propagation profile, and the log-linear decay of
that profile with sequence separation.

E16 -- does a four-state likelihood interaction appear in a model's own products?
The measurement is ``log p(AB) - log p(Ab) - log p(aB) + log p(ab)`` on residue
pairs of a generated sequence, selected as contacting and non-contacting with
*exact one-to-one matching on sequence separation*, from the folded structure of
that sequence.

E07 -- does a model's own four-state interaction term concentrate at structural
contacts of natural proteins, where an experimental non-additivity measurement
exists? The measurement is the same four-state term, on the frozen MegaScale
double-mutant cohort, aggregated to the site pairs the R3 contact annotation
already classified, and contrasted with the same coarsened-exact matching, the
same weighting and the same two-stage bootstrap that the published measured
endpoint used.

Three things have to be said once, here, because every number this module
produces inherits them.

**On a generated protein there is no experimental measurement.** The four-state
term of E16 is the model's *internal* non-additivity: how far its own joint
likelihood departs from the sum of its own single-mutation likelihoods. It is not
an estimate of real epistasis and carries no accuracy claim. Only E07 has a
measured endpoint and therefore only E07 can speak to accuracy.

**Length is the confounder that matters for E15.** The downstream propagation
quantities are sums and means over receivers, and the number of receivers of a
mutation at site ``i`` is ``L - i``; the decay half-distance E01 measured is of
the same order as the sequence lengths here. A generated/natural comparison that
did not match on length would be a comparison of lengths. Matching is therefore
one-to-one with a declared caliper, and the realised imbalance is reported beside
every estimate rather than asserted away.

**A merged-piece tokenisation costs position resolution, not scalars.** For an
arm whose segmentation of a residue depends on its neighbours -- the ProLLaMA
lineage's SentencePiece among them -- a substitution can change the token grid,
and a mutation whose grid moved has no per-position correspondence at all. Those
mutations are dropped from the position-resolved analysis and counted, and the
retained fraction travels with the estimate. The *scalar* quantities (the
likelihood difference, the four-state term) are unaffected: they are differences
of whole-sequence sums and need no alignment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..extensions.responses import DISTANCE_EDGES, DISTANCE_NAMES
from ..generation.near_duplicates import NEAR_DUPLICATE_CONTAINMENT, RESIDUE_SHINGLE, near_duplicate_groups
from ..position.contact_response import (
    CONTACT_ANGSTROM,
    MIN_SEQUENCE_SEPARATION,
    rebuild_states,
    receiver_census,
    separation_stratum,
)
from ..position.position_likelihood import read_archive
from ..position.position_terms import alignment
from .pairwise_epistasis import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, interval

COHORT_SCHEMA = "generated_mutation_cohort_v1"

#: Every selection in this module is a function of this one seed and of the
#: sequence itself, so a cohort is reproducible from the sequence file alone and
#: no selection can depend on a score.
SELECTION_SEED = 20261008

#: One-to-one length matching caliper: a natural partner may differ from its
#: generated sequence by at most this many residues, or by this fraction of the
#: generated length, whichever is larger. A generated sequence with no partner
#: inside the caliper leaves the matched design and is counted, because a wider
#: caliper for the hard cases is the imbalance the design exists to prevent.
LENGTH_CALIPER_RESIDUES = 2
LENGTH_CALIPER_FRACTION = 0.02

#: Exact separation matching for the E16 pair design: a non-contacting control
#: may differ from its contacting partner by at most this many residues of
#: sequence separation, *inside* the same declared separation stratum.
SEPARATION_CALIPER = 2

#: Predicted-structure confidence floor, in pLDDT units. A position below it has
#: no admitted coordinate and enters neither the contacting nor the
#: non-contacting class, exactly as an unmapped experimental position does in
#: ``position.contact_response``.
CONFIDENCE_FLOOR = 70.0

#: Keys a predicted-structure archive may carry the CB-CB (CA for glycine)
#: distance matrix and the per-residue confidence under. Tried in order; an
#: archive carrying none of them is refused with its own key list in the message,
#: because a missing distance matrix is a different product and not a weaker one.
DISTANCE_KEYS = ("cb_distance_angstrom", "cb_distance", "distance_cb", "cb_distances")
CONFIDENCE_KEYS = ("plddt", "plddt_ca", "confidence")

#: Declared degeneracy stratum of a generated product, from the properties the
#: frozen generated set already carries. A homopolymeric run or a near-zero
#: composition entropy makes a sequence trivially predictable, which moves every
#: likelihood quantity for reasons that have nothing to do with being generated;
#: the stratum is reported, and the primary estimate is repeated inside the
#: non-degenerate stratum as a sensitivity rather than the degenerate products
#: being dropped from an outcome-blind set.
DEGENERATE_ENTROPY_NATS = 1.5
DEGENERATE_RUN_RESIDUES = 10

ORIGINS = ("generated", "natural")


# --------------------------------------------------------------- the sequences


def _rng(*parts: Any) -> np.random.Generator:
    """A generator seeded by the declared seed and the content it acts on."""

    material = "\x1f".join(str(part) for part in parts).encode()
    return np.random.default_rng(
        [SELECTION_SEED, int.from_bytes(hashlib.sha256(material).digest()[:8], "big")]
    )


def read_generated(path: Path, *, stages: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The frozen generated set, re-verified against its own digests.

    Every record's sequence is rehashed and its length recounted. The file is the
    primary key for this whole programme, so a record that does not describe
    itself is a refusal here rather than a surprise three stages later.
    """

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        sequence = str(row["sequence"])
        if hashlib.sha256(sequence.encode()).hexdigest() != row["sequence_sha256"]:
            raise ValueError(f"{path}:{number}: sequence does not match its own sha256")
        if len(sequence) != int(row["length"]):
            raise ValueError(f"{path}:{number}: sequence length disagrees with its own field")
        if set(sequence) - set(AA20):
            raise ValueError(f"{path}:{number}: {row['id']} carries a non-AA20 symbol")
        if row["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate identity {row['id']!r}")
        seen.add(row["id"])
        if stages is None or row["stage"] in stages:
            records.append(row)
    if not records:
        raise ValueError(f"{path}: no record selected")
    return sorted(records, key=lambda row: row["id"])


def degenerate(record: Mapping[str, Any]) -> bool:
    """Whether this generated product falls in the declared degeneracy stratum."""

    properties = record["properties"]
    return bool(
        float(properties["composition_entropy_nats"]) < DEGENERATE_ENTROPY_NATS
        or int(properties["longest_single_residue_run"]) >= DEGENERATE_RUN_RESIDUES
    )


def stratified_subsample(
    records: Sequence[Mapping[str, Any]], *, per_stage: int
) -> list[dict[str, Any]]:
    """A seeded draw of ``per_stage`` products per stage, equally over streams.

    The stratification is over the two declarations the generated set already
    carries and nothing else -- stage and random stream -- so the draw is blind
    to length, composition and any outcome. Length balance is not imposed: it
    follows from drawing uniformly inside a stream, and the realised length
    distribution is reported.
    """

    if per_stage < 1:
        raise ValueError("a subsample needs at least one product per stage")
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        by_cell.setdefault((str(row["stage"]), str(row["stream"])), []).append(row)
    stages = sorted({stage for stage, _ in by_cell})
    chosen: list[dict[str, Any]] = []
    for stage in stages:
        streams = sorted(stream for found, stream in by_cell if found == stage)
        if not streams:
            continue
        base, extra = divmod(per_stage, len(streams))
        for index, stream in enumerate(streams):
            pool = sorted(by_cell[(stage, stream)], key=lambda row: row["id"])
            want = base + (1 if index < extra else 0)
            if want > len(pool):
                raise ValueError(
                    f"{stage}/{stream} holds {len(pool)} products and {want} were requested"
                )
            order = _rng("subsample", stage, stream).permutation(len(pool))
            chosen.extend(dict(pool[int(position)]) for position in order[:want])
    return sorted(chosen, key=lambda row: row["id"])


def natural_identity(sequence: str) -> str:
    """A content identity for a natural partner, so a cohort row is reproducible.

    Content-derived, which makes it reproducible and makes two byte-identical
    natural records *the same* row rather than two rows. That is the right
    semantics and it is why :func:`match_natural` draws from distinct sequences:
    Swiss-Prot is non-redundant per entry and not per sequence, so a pool of
    387,931 eligible records in the 39-408 band carries only 316,900 distinct
    sequences, one of them 63 times. Drawing records would hand two generated
    products the same protein under one identity.
    """

    return "nat_" + hashlib.sha256(sequence.encode()).hexdigest()[:20]


def match_natural(
    generated: Sequence[Mapping[str, Any]], pool: Sequence[str]
) -> dict[str, Any]:
    """One natural partner per generated product, matched on length, without reuse.

    The pool is reduced to its **distinct sequences**, permuted once under the
    declared seed and bucketed by length, so which natural protein fills a length
    is a function of the seed and not of the file order. Distinct sequences rather
    than records because the comparator is a protein, not a database entry: two
    byte-identical Swiss-Prot entries are one protein sequenced in two strains,
    ``near_duplicate_groups`` would put them in one independence group anyway, and
    drawing both would give two generated products the same comparator under one
    content identity. The two invariants this function owns -- one natural protein
    serves at most one generated product, and the partners carry distinct
    identities -- are asserted before it returns.

    Partners are assigned shortest-generated-first, which is where the caliper
    binds: a 40-residue product has far fewer admissible partners than a
    400-residue one, and serving the scarce end first keeps a failure to match
    from being pushed onto the short products as a group.
    """

    distinct: list[str] = list(dict.fromkeys(str(record) for record in pool))
    buckets: dict[int, list[int]] = {}
    order = _rng("natural-pool", len(distinct)).permutation(len(distinct))
    for position in order:
        buckets.setdefault(len(distinct[int(position)]), []).append(int(position))
    used: set[int] = set()
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for row in sorted(generated, key=lambda row: (int(row["length"]), str(row["id"]))):
        length = int(row["length"])
        caliper = max(LENGTH_CALIPER_RESIDUES, int(round(LENGTH_CALIPER_FRACTION * length)))
        partner: int | None = None
        for delta in range(caliper + 1):
            for candidate_length in ({length - delta, length + delta} if delta else {length}):
                for position in buckets.get(candidate_length, ()):
                    if position not in used:
                        partner = position
                        break
                if partner is not None:
                    break
            if partner is not None:
                break
        if partner is None:
            unmatched.append({"id": row["id"], "length": length, "caliper": caliper})
            continue
        used.add(partner)
        sequence = distinct[partner]
        matched.append(
            {
                "id": natural_identity(sequence),
                "sequence": sequence,
                "length": len(sequence),
                "matched_to": row["id"],
                "length_delta": len(sequence) - length,
                "pool_position": partner,
            }
        )
    sequences = [row["sequence"] for row in matched]
    identities = [row["id"] for row in matched]
    if len(set(sequences)) != len(sequences) or len(set(identities)) != len(identities):
        raise AssertionError(
            "match_natural returned a reused natural protein; the pool is drawn from "
            "distinct sequences without replacement, so this cannot happen and a cohort "
            "must not be written from it"
        )
    deltas = [abs(int(row["length_delta"])) for row in matched]
    return {
        "matched": sorted(matched, key=lambda row: row["matched_to"]),
        "unmatched": unmatched,
        "balance": {
            "pairs": len(matched),
            "unmatched_generated": len(unmatched),
            "pool_records": len(pool),
            "pool_distinct_sequences": len(distinct),
            "pool_duplicate_records_removed": len(pool) - len(distinct),
            "draw": "distinct eligible sequences, without replacement",
            "mean_absolute_length_delta_residues": float(np.mean(deltas)) if deltas else None,
            "max_absolute_length_delta_residues": int(max(deltas)) if deltas else None,
            "caliper_rule": (
                f"max({LENGTH_CALIPER_RESIDUES} residues, "
                f"{LENGTH_CALIPER_FRACTION:g} x generated length)"
            ),
        },
    }


def independence_groups(sequences: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    """Near-duplicate components of the union, as the bootstrap unit.

    A corpus record is not an independent unit and neither is a generated
    product: a model that emits two homopolymeric runs has emitted one thing
    twice. ``generation.near_duplicates`` is this project's declaration of what
    one unit is, and it is reused rather than restated.
    """

    labels, record = near_duplicate_groups(list(sequences), unit="residues")
    names = [f"g{int(label):05d}" for label in labels]
    return names, {
        "unit": "near-duplicate component at residue shingles",
        "containment": NEAR_DUPLICATE_CONTAINMENT,
        "shingle": RESIDUE_SHINGLE,
        "groups": len(set(names)),
        "records": len(names),
        "detail": record,
    }


# --------------------------------------------------------------- the mutations


def substitution_label(sequence: str, site: int, residue: str) -> str:
    return f"{sequence[site]}{site + 1}{residue}"


def state_label(wildtype: str, state: str) -> str:
    """The one- or two-substitution label of a state, read off the wild type.

    One spelling, used by the builder that writes a cohort and by every analysis
    that joins a measured cycle back to it; two spellings of the same label would
    be a silent join failure rather than an error.
    """

    if len(state) != len(wildtype):
        raise ValueError("state length differs from the wild type")
    sites = [index for index in range(len(wildtype)) if wildtype[index] != state[index]]
    if not 1 <= len(sites) <= 2:
        raise ValueError(f"{len(sites)} substituted positions is not a cycle state")
    return ":".join(substitution_label(wildtype, site, state[site]) for site in sites)


def apply_substitutions(sequence: str, changes: Mapping[int, str]) -> str:
    out = list(sequence)
    for site, residue in changes.items():
        if out[site] == residue:
            raise ValueError(f"position {site} is already {residue}")
        out[site] = residue
    return "".join(out)


def scan_mutations(sequence: str, *, sites: int, subs: int) -> list[dict[str, Any]]:
    """A seeded single-substitution scan: ``sites`` positions, ``subs`` each.

    Positions are drawn uniformly without replacement over the whole sequence.
    Uniform, rather than weighted toward the N-terminus where a causal arm has
    more downstream receivers, because the anchor cohorts this comparison extends
    are site-saturation scans and a weighted draw would make the generated
    profile and the published natural one different measurements.
    """

    if sites < 1 or subs < 1:
        raise ValueError("a scan needs at least one site and one substitution")
    length = len(sequence)
    generator = _rng("scan", sequence, sites, subs)
    positions = sorted(
        int(value)
        for value in generator.choice(length, size=min(sites, length), replace=False)
    )
    rows: list[dict[str, Any]] = []
    for site in positions:
        alternatives = sorted(set(AA20) - {sequence[site]})
        picked = generator.choice(
            len(alternatives), size=min(subs, len(alternatives)), replace=False
        )
        for index in sorted(int(value) for value in picked):
            residue = alternatives[index]
            rows.append(
                {
                    "label": substitution_label(sequence, site, residue),
                    "sequence": apply_substitutions(sequence, {site: residue}),
                    "site": site,
                    "wild_residue": sequence[site],
                    "mutant_residue": residue,
                }
            )
    return rows


def pair_mutations(sequence: str, pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The four-state states of a pair design, with one substitution per position.

    A position that appears in several selected pairs carries the *same*
    substitution in all of them. That is not an economy: it makes the single-state
    term of every cycle through that position literally the same number, so a
    difference between two cycles cannot come from having drawn a different
    replacement residue.
    """

    positions = sorted({int(pair["i"]) for pair in pairs} | {int(pair["j"]) for pair in pairs})
    replacement: dict[int, str] = {}
    for site in positions:
        alternatives = sorted(set(AA20) - {sequence[site]})
        index = int(_rng("pair-substitution", sequence, site).integers(len(alternatives)))
        replacement[site] = alternatives[index]
    states: dict[str, str] = {}
    for site in positions:
        label = substitution_label(sequence, site, replacement[site])
        states[label] = apply_substitutions(sequence, {site: replacement[site]})
    cycles = []
    for pair in sorted(pairs, key=lambda row: (int(row["i"]), int(row["j"]))):
        low, high = int(pair["i"]), int(pair["j"])
        single_low = substitution_label(sequence, low, replacement[low])
        single_high = substitution_label(sequence, high, replacement[high])
        double = f"{single_low}:{single_high}"
        states[double] = apply_substitutions(
            sequence, {low: replacement[low], high: replacement[high]}
        )
        cycles.append(
            {
                "i": low,
                "j": high,
                "separation": high - low,
                "stratum": separation_stratum(high - low),
                "contact": bool(pair["contact"]),
                "structure_distance_angstrom": float(pair["distance"]),
                "single_low": single_low,
                "single_high": single_high,
                "double": double,
            }
        )
    ordered = sorted(states)
    return {
        "mutants": ordered,
        "sequences": [states[label] for label in ordered],
        "cycles": cycles,
        "substitutions": {str(site): replacement[site] for site in positions},
    }


# ------------------------------------------------------- the predicted structure


def read_predicted_structure(
    path: Path, *, sequence: str, confidence_key: str = "auto", confidence_floor: float = CONFIDENCE_FLOOR
) -> dict[str, Any]:
    """CB-CB distances and admitted positions of one folded generated sequence.

    ``confidence_key`` is ``"auto"`` to take the first declared confidence array
    the archive carries, a literal key name, or ``"none"`` to declare explicitly
    that no confidence filter is applied. There is no silent third behaviour: an
    archive without a confidence array and without ``"none"`` is refused, because
    a contact map read at unknown confidence is not the same measurement as one
    read at pLDDT 70.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        present = list(data.files)
        distance_key = next((key for key in DISTANCE_KEYS if key in present), None)
        if distance_key is None:
            raise ValueError(
                f"{path}: no CB-CB distance matrix under any of {list(DISTANCE_KEYS)}; "
                f"the archive carries {present}"
            )
        distance = np.asarray(data[distance_key], dtype=np.float64)
        if confidence_key == "none":
            found_key, confidence = None, None
        else:
            candidates = CONFIDENCE_KEYS if confidence_key == "auto" else (confidence_key,)
            found_key = next((key for key in candidates if key in present), None)
            if found_key is None:
                raise ValueError(
                    f"{path}: no confidence array under any of {list(candidates)}; the archive "
                    f"carries {present}. Pass --confidence-key none to declare that this "
                    "analysis applies no confidence filter."
                )
            confidence = np.asarray(data[found_key], dtype=np.float64)
    length = len(sequence)
    if distance.shape != (length, length):
        raise ValueError(
            f"{path}: distance matrix {distance.shape} is not {length}x{length} for this sequence"
        )
    if confidence is None:
        admitted = np.ones(length, dtype=bool)
    else:
        if confidence.shape != (length,):
            raise ValueError(f"{path}: confidence {confidence.shape} is not one value per residue")
        admitted = np.asarray(confidence >= float(confidence_floor), dtype=bool)
    return {
        "distance": distance,
        "admitted": admitted,
        "distance_key": distance_key,
        "confidence_key": found_key,
        "confidence_floor": None if confidence is None else float(confidence_floor),
        "admitted_positions": int(admitted.sum()),
        "length": length,
        "keys_present": present,
    }


def eligible_pairs(
    structure: Mapping[str, Any],
    *,
    cutoff: float = CONTACT_ANGSTROM,
    min_separation: int = MIN_SEQUENCE_SEPARATION,
) -> list[dict[str, Any]]:
    """Every admitted position pair above the separation floor, classified.

    The cutoff and the floor are the ones ``position.contact_response`` declares,
    so a contact on a generated protein is the same relation a contact is on a
    natural one. A non-finite distance is a refusal and never a non-contact.
    """

    distance = np.asarray(structure["distance"], dtype=np.float64)
    admitted = np.flatnonzero(np.asarray(structure["admitted"], dtype=bool))
    rows: list[dict[str, Any]] = []
    for first in range(len(admitted)):
        for second in range(first + 1, len(admitted)):
            low, high = int(admitted[first]), int(admitted[second])
            if high - low < min_separation:
                continue
            value = float(distance[low, high])
            if not np.isfinite(value):
                raise ValueError(
                    f"positions {low} and {high} have no finite CB-CB distance; a pair this "
                    "coordinate cannot be formed from is a refusal, not a non-contact"
                )
            rows.append(
                {
                    "i": low,
                    "j": high,
                    "separation": high - low,
                    "stratum": separation_stratum(high - low),
                    "distance": value,
                    "contact": bool(value < cutoff),
                }
            )
    return rows


def matched_pair_design(
    pairs: Sequence[Mapping[str, Any]],
    sequence: str,
    *,
    per_stratum: int,
    caliper: int = SEPARATION_CALIPER,
) -> dict[str, Any]:
    """Contacting pairs and separation-matched non-contacting controls.

    Matching is exact one-to-one inside a declared separation stratum and within
    ``caliper`` residues of separation. Sequence separation is the dominant
    confounder of every contact reading -- contacting residues are closer along
    the chain, and the likelihood response decays with chain distance for reasons
    that are not structural -- so a contact that cannot be matched is dropped
    rather than compared against a more distant control.
    """

    if per_stratum < 1:
        raise ValueError("a pair design needs at least one contact per stratum")
    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for stratum in DISTANCE_NAMES:
        contacts = sorted(
            (row for row in pairs if row["contact"] and row["stratum"] == stratum),
            key=lambda row: (row["i"], row["j"]),
        )
        controls = sorted(
            (row for row in pairs if not row["contact"] and row["stratum"] == stratum),
            key=lambda row: (row["i"], row["j"]),
        )
        if not contacts:
            continue
        if not controls:
            # A stratum with contacts and no control at all cannot be matched. The
            # contacts are reported as dropped rather than skipped silently: an
            # unmatchable cell is support the design did not use.
            dropped.extend(
                {"i": row["i"], "j": row["j"], "reason": f"no non-contact in stratum {stratum}"}
                for row in contacts
            )
            continue
        order = _rng("pair-design", sequence, stratum).permutation(len(contacts))
        available = list(controls)
        taken = 0
        for position in order:
            if taken >= per_stratum:
                break
            contact = contacts[int(position)]
            best, best_gap = None, None
            for index, control in enumerate(available):
                gap = abs(int(control["separation"]) - int(contact["separation"]))
                if gap > caliper:
                    continue
                if best_gap is None or gap < best_gap:
                    best, best_gap = index, gap
            if best is None:
                dropped.append(
                    {"i": contact["i"], "j": contact["j"], "reason": "no control inside caliper"}
                )
                continue
            control = available.pop(best)
            selected.append(dict(contact, matched_separation_gap=int(best_gap)))
            selected.append(dict(control, matched_separation_gap=int(best_gap)))
            taken += 1
    contacts = [row for row in selected if row["contact"]]
    controls = [row for row in selected if not row["contact"]]
    imbalance = (
        float(np.mean([row["separation"] for row in contacts]))
        - float(np.mean([row["separation"] for row in controls]))
        if contacts and controls
        else None
    )
    return {
        "pairs": sorted(selected, key=lambda row: (row["i"], row["j"])),
        "dropped_contacts": dropped,
        "balance": {
            "contacts": len(contacts),
            "controls": len(controls),
            "eligible_pairs": len(pairs),
            "eligible_contacts": sum(1 for row in pairs if row["contact"]),
            "separation_imbalance_residues": imbalance,
            "strata": sorted({row["stratum"] for row in selected}),
            "caliper_residues": int(caliper),
        },
    }


# ------------------------------------------------------------ the cohort record


def cohort_assay(
    *,
    assay: str,
    wildtype: str,
    mutants: Sequence[str],
    sequences: Sequence[str],
    cluster: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One row in the shape ``extract_position_likelihood.py`` already reads.

    The cohort format is not new here. It is the one the position-likelihood
    stage declares, so the extractor consumes a generated-protein scan, a
    generated-protein pair design and the frozen natural double-mutant cohort
    without a second producer and without a second convention.
    """

    if set(wildtype) - set(AA20):
        raise ValueError(f"{assay}: wild type carries a non-AA20 symbol")
    if len(mutants) != len(sequences) or not mutants:
        raise ValueError(f"{assay}: one sequence per mutant is required")
    if len(set(mutants)) != len(mutants):
        raise ValueError(f"{assay}: duplicate mutant identity")
    digest = hashlib.sha256(
        json.dumps(list(mutants), separators=(",", ":")).encode()
    ).hexdigest()
    row = {
        "assay": assay,
        "wildtype": wildtype,
        "mutants": list(mutants),
        "sequences": list(sequences),
        "cluster": cluster,
        "mutant_digest": digest,
    }
    row.update(dict(extra or {}))
    return row


def require_unique_assays(assays: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a cohort whose consumer would refuse it, at the producer.

    ``extract_position_likelihood.select_assays`` rejects a duplicate assay
    identity, and it is right to. But a builder that can emit such a cohort is
    the defect and the consumer's guard is only the symptom: the cell dies after
    the campaign has been frozen, pushed and scheduled, which is the most
    expensive place to find it. This is the producer-side half of that contract,
    called by every mode before a cohort is written, and it names the colliding
    identities and the sequences behind them so the cause is readable without
    reopening the file.
    """

    seen: dict[str, list[str]] = {}
    for row in assays:
        seen.setdefault(str(row["assay"]), []).append(str(row["wildtype"]))
    collisions = {
        assay: wildtypes for assay, wildtypes in seen.items() if len(wildtypes) > 1
    }
    if collisions:
        detail = ", ".join(
            f"{assay} x{len(wildtypes)}"
            f"{' (identical sequences)' if len(set(wildtypes)) == 1 else ' (different sequences)'}"
            for assay, wildtypes in sorted(collisions.items())[:8]
        )
        raise ValueError(
            f"{len(collisions)} assay identities are not unique: {detail}. A cohort with a "
            "repeated identity would be refused by the extraction stage, and the duplicate "
            "would also mean one comparator serving two rows."
        )


# --------------------------------------------------------------- stage plumbing


def prepare_output_directory(out: Path, completion: str) -> Path:
    """Accept an empty existing ``--out``; refuse one that already holds work.

    The campaign queue creates the output directory before it launches the cell,
    so refusing a directory that merely exists refuses every queued run. What must
    be refused is *prior work*: a completion record, or any other content. The
    distinction is the whole of this function and it is why it is written once.
    """

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / completion).exists():
        raise SystemExit(f"{out}: {completion} is already present; this cell is complete")
    existing = sorted(item.name for item in out.iterdir())
    if existing:
        raise SystemExit(
            f"{out}: refusing to write into a directory that already holds {existing[:8]}"
        )
    return out


# ------------------------------------------------------------------- the reading


@dataclass(frozen=True)
class Extraction:
    """One arm's completed extraction directory, indexed by assay."""

    arm: str
    paradigm: str
    root: Path
    files: dict[str, str]
    reasons: dict[str, str]
    misaligned: dict[str, frozenset[str]]
    completion: dict[str, Any]

    def payload(self, assay: str) -> dict[str, Any]:
        return read_archive(self.root / "archives" / self.files[assay])

    def states(self, payload: Mapping[str, Any], sequences: Sequence[str]):
        """The packing sidecars of one assay, read back out of its own archive."""

        return rebuild_states(payload, list(sequences))

    def agrees_on_alignment(self, assay: str, found: Sequence[str]) -> None:
        """Refuse if this reader's alignment verdicts differ from the producer's.

        The producing stage recorded, per assay, which single substitutions it
        could align. Recomputing them here from the archive's own packing is a
        check, not a copy -- and a disagreement means one of the two is reading a
        different token partition, in which case neither reading may be published.
        """

        declared = self.misaligned.get(assay)
        if declared is None:
            return
        if frozenset(found) != declared:
            raise ValueError(
                f"{assay}: the recomputed misaligned substitutions "
                f"{sorted(set(found) ^ set(declared))[:8]} disagree with the producing "
                "stage's own record"
            )

    def covered(self, assays: Sequence[str]) -> tuple[list[str], list[dict[str, str]]]:
        """Which requested assays this extraction scored, and why the rest are absent.

        An assay the producing stage recorded as excluded or skipped -- outside the
        residue band, past the packed-token budget, without a structural mapping --
        is a measurement outcome and is dropped with its own reason. An assay that
        is absent with no recorded reason is a defect, and the caller is expected
        to refuse on it.
        """

        present = [assay for assay in assays if assay in self.files]
        absent = [
            {"assay": assay, "reason": self.reasons.get(assay, "absent without a recorded reason")}
            for assay in assays
            if assay not in self.files
        ]
        return present, absent


def open_extraction(root: Path) -> Extraction:
    """Index a completed extraction by its own completion record.

    The archive filenames are derived by the producing stage, so they are read
    out of its receipt rather than recomputed here: one naming rule, in one
    place, and a cell that did not finish has no receipt to read.
    """

    root = Path(root)
    completion = json.loads((root / "position_likelihood.json").read_text())
    if completion.get("status") != "complete":
        raise ValueError(f"{root}: extraction is not complete")
    identity = completion["identity"]
    files = {str(row["assay"]): str(row["file"]) for row in completion["assays"]}
    if len(files) != len(completion["assays"]):
        raise ValueError(f"{root}: duplicate assay in the completion record")
    worst = float(completion["totals"]["retention_max_abs_nats"])
    if identity["paradigm"] == "causal_next_token" and worst != 0.0:
        raise ValueError(
            f"{root}: retention residual {worst} is not identically zero; the retained "
            "vectors are not the ones the arm's own scalars reduce"
        )
    reasons = {
        str(row["assay"]): str(row["reason"])
        for key in ("excluded_assays", "skipped_assays")
        for row in completion.get(key, ())
    }
    misaligned = {
        str(row["assay"]): frozenset(
            str(item["mutation"]) for item in row["misaligned_single_substitutions"]
        )
        for row in completion["assays"]
        if "misaligned_single_substitutions" in row
    }
    return Extraction(
        arm=str(identity["arm"]),
        paradigm=str(identity["paradigm"]),
        root=root,
        files=files,
        reasons=reasons,
        misaligned=misaligned,
        completion=completion,
    )


def state_likelihoods(payload: Mapping[str, Any]) -> dict[str, float]:
    """``log p(mutant) - log p(wild type)`` per mutant, as the archive holds it."""

    mutants = [str(value) for value in payload["mutants"]]
    values = np.asarray(payload["likelihood"], dtype=np.float64)
    if len(mutants) != len(values):
        raise ValueError("the archive does not hold one likelihood per mutant")
    return dict(zip(mutants, (float(value) for value in values)))


def mutation_receivers(
    payload: Mapping[str, Any], *, states: Sequence[Mapping[str, Any]], paradigm: str,
    index: int, site: int
) -> tuple[dict[str, Any] | None, str | None]:
    """One single substitution's per-residue response, or why it has none.

    The alignment verdict comes first, from ``position_terms.alignment`` -- the
    rule the producing stage itself applies. A merged-piece interface can
    retokenise a sequence when one residue changes, and it can do so *without*
    changing the token count: the two arrays then have the same shape while their
    boundaries differ, so a token-shape check alone would silently compare the
    likelihood of different residues and report the mismatch as an upstream
    invariance failure. An unaligned mutation has no per-position correspondence
    at all; it is reported as an absence with its reason and never repaired by
    padding or by aligning on residue index.
    """

    verdict = alignment(dict(states[0]), dict(states[index + 1]), site)
    if not verdict["aligned"]:
        return None, str(verdict["reason"])
    try:
        return receiver_census(payload, index, site=site, paradigm=paradigm), None
    except ValueError as error:
        if "token grid" in str(error):
            return None, str(error)
        raise


def four_state_interaction(
    likelihoods: Mapping[str, float], cycle: Mapping[str, Any]
) -> float:
    """``log p(AB) - log p(Ab) - log p(aB) + log p(ab)``, the wild type cancelling.

    Every term is already expressed relative to the wild type by the archive, so
    the wild-type term is identically zero and the interaction is a difference of
    three retained scalars. Spelling it this way keeps the reported quantity the
    arm's own published reduction rather than a re-summation of the position
    vector.
    """

    return float(
        likelihoods[cycle["double"]]
        - likelihoods[cycle["single_low"]]
        - likelihoods[cycle["single_high"]]
    )


def additive_prediction(likelihoods: Mapping[str, float], cycle: Mapping[str, Any]) -> float:
    """The model's own additive prediction for the double state."""

    return float(likelihoods[cycle["single_low"]] + likelihoods[cycle["single_high"]])


# ------------------------------------------------------------- the aggregations


def nested_mean(entries: Iterable[tuple[Any, float]]) -> float | None:
    """Mean over distinct keys, each key's rows averaged first."""

    grouped: dict[Any, list[float]] = {}
    for key, value in entries:
        if value is None or not np.isfinite(float(value)):
            continue
        grouped.setdefault(key, []).append(float(value))
    if not grouped:
        return None
    return float(np.mean([float(np.mean(values)) for values in grouped.values()]))


def paired_origin_contrast(
    rows: Sequence[Mapping[str, Any]], *, value: str,
    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Generated minus matched natural, with the independence group as the unit.

    Each row carries a ``group``, an ``origin`` and a ``sequence_id``. Rows are
    averaged within a sequence, sequences within an origin inside a group, and the
    group's generated-minus-natural difference is the resampled quantity. A
    natural partner is assigned the group of the generated sequence it was matched
    to, so the pairing survives the resample: a draw takes a matched pair or
    neither member of it.
    """

    by_group: dict[Any, dict[str, list[tuple[Any, float]]]] = {}
    for row in rows:
        if row[value] is None or not np.isfinite(float(row[value])):
            continue
        cell = by_group.setdefault(row["group"], {origin: [] for origin in ORIGINS})
        cell[row["origin"]].append((row["sequence_id"], float(row[value])))
    differences: dict[Any, float] = {}
    sides: dict[str, dict[Any, float]] = {origin: {} for origin in ORIGINS}
    for group, cell in by_group.items():
        means = {origin: nested_mean(cell[origin]) for origin in ORIGINS}
        for origin in ORIGINS:
            if means[origin] is not None:
                sides[origin][group] = means[origin]
        if means["generated"] is not None and means["natural"] is not None:
            differences[group] = means["generated"] - means["natural"]
    record = {
        "value": value,
        "unit": "near-duplicate independence group, generated and matched natural paired",
        "difference": interval(
            [differences[group] for group in sorted(differences)], draws=draws, seed=seed
        ),
        "paired_groups": len(differences),
    }
    for origin in ORIGINS:
        record[origin] = interval(
            [sides[origin][group] for group in sorted(sides[origin])], draws=draws, seed=seed
        )
    return record


def profile_contrast(
    rows: Sequence[Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Generated minus natural mean absolute response, inside separation strata.

    Separation is the axis the response decays along, so the comparison is made
    inside a stratum and never pooled across them. The strata are the frozen
    distance names this project already uses for residue separation.

    Rows are per-sequence aggregates inside one stratum -- one row per
    (sequence, stratum) carrying that cell's mean absolute response and mean
    separation -- because a full-cohort receiver table runs to millions of rows
    and the nesting averages within a sequence anyway. The reported separation
    imbalance is therefore sequence-equal, matching the estimate's own weighting.
    """

    out = []
    for stratum in DISTANCE_NAMES:
        selected = [row for row in rows if row["stratum"] == stratum]
        if not selected:
            continue
        record = paired_origin_contrast(selected, value="absolute_response", draws=draws, seed=seed)
        record["stratum"] = stratum
        record["receivers"] = len(selected)
        record["separation_imbalance_residues"] = _separation_imbalance(selected)
        out.append(record)
    return {"strata": out, "edges": list(DISTANCE_EDGES), "names": list(DISTANCE_NAMES)}


def _separation_imbalance(rows: Sequence[Mapping[str, Any]]) -> float | None:
    generated = [float(row["separation"]) for row in rows if row["origin"] == "generated"]
    natural = [float(row["separation"]) for row in rows if row["origin"] == "natural"]
    if not generated or not natural:
        return None
    return float(np.mean(generated) - np.mean(natural))


def precision_record(estimate: Mapping[str, Any]) -> dict[str, Any]:
    """Half-width and the effect a stratum of this precision could resolve.

    An interval that includes zero says nothing until its width is read: the
    audit lesson this programme is operating under is that an unresolved stratum
    is unresolved and not zero. ``minimum_resolvable`` is the half-width itself --
    the smallest effect whose two-sided interval at this precision would exclude
    zero -- and ``minimum_detectable_80`` scales it to the conventional
    80%-power effect size for the same standard error.
    """

    bounds = estimate.get("interval")
    if not bounds or bounds[0] is None or bounds[1] is None:
        return {
            "half_width": None,
            "minimum_resolvable": None,
            "minimum_detectable_80": None,
            "reason": estimate.get("undefined", "no interval was formed"),
        }
    half = (float(bounds[1]) - float(bounds[0])) / 2.0
    standard_error = half / 1.959963984540054
    return {
        "half_width": half,
        "minimum_resolvable": half,
        "minimum_detectable_80": float(2.801585 * standard_error),
        "groups": estimate.get("groups"),
    }
