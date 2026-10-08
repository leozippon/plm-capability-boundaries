"""Is the length-matched natural *fragment* an adequate comparator for whole-protein generation?

The concern
===========

Section 6's generative endpoint is a generated attempt's complete-domain
recognition minus that of a **contiguous substring of a real protein cut to the
attempt's exact length**. A fragment of a real protein is an odd thing to ask a
whole-protein generator to beat: cut a 300-residue enzyme to 60 residues and the
curated profile that recognises the enzyme may no longer clear its own gathering
threshold, so the comparator can be easier to beat than the biology it stands
for -- or, if the fragment happens to span a short domain, harder.

What is already measured, and what is not
=========================================

The generation-and-control gate
(``results/R6/generative_control_20260924/gate_endpoints.json``) already carries
a **whole-record** natural comparator beside the fragment: one complete UniRef50
record in the attempt's length band, on the same oracle, the same attempt ledger
and the same near-duplicate-group bootstrap. :func:`comparator_table` reads both
and states, per cell, whether the published verdict survives the substitution.
That is the adequacy question, answered without new sampling.

What is **not** measured anywhere is a natural comparator matched on the
requested *family* as well as the length. :func:`family_matched_draw` builds it,
and only where it is well defined: a conditional cell requests a class, so the
class's frozen Pfam referent supplies the family before any generated outcome is
read. For an unconditional cell there is no requested family, and matching on
the families the generations happened to hit would select the comparator on the
outcome -- the one thing the campaign's shape rules forbid. That cell therefore
gets no family-matched arm, and the artefact says so rather than leaving the
reader to infer it.

The family-matched draw is a **ceiling, not a control the model is asked to
beat**: real full-length proteins of the requested family are what a perfect
conditional generator would produce, so their complete-domain rate prices the
instrument on exactly the families being requested.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "d1_expanded_generation_controls_v1"

#: The gate's two comparators that stand for a real protein. Everything else it
#: measures (shuffle, markov_k, hydropathy) carries declared sequence statistics
#: and is not a natural-protein comparator.
NATURAL_COMPARATORS: tuple[str, str] = ("fragment", "natural")

#: The gate's endpoints, in the order the manuscript reads them.
ENDPOINTS: tuple[str, str] = ("complete_domain", "any_family")

#: Length-match tolerance for the family-matched draw, as a fraction of the
#: generated attempt length it stands against. Declared before any natural
#: recognition rate was read, and the same tolerance the conditional campaign's
#: random anchor side already uses.
LENGTH_MATCH_TOLERANCE: float = 0.10

#: Natural full-length records per class in the family-matched draw.
DRAW_PER_CLASS: int = 100

#: The draw seed. One value reproduces the whole draw from the corpus.
DRAW_SEED: int = 20261008

CEILING: dict[str, str] = {
    "a_comparator_is_not_a_null": (
        "neither the fragment nor the whole natural record is a null hypothesis. They "
        "are reference populations, and the endpoint is a difference against a named "
        "one"
    ),
    "family_matched_is_a_ceiling": (
        "real full-length proteins of the requested family are what a perfect "
        "conditional generator would emit; their rate prices the instrument and is "
        "never a control the model is asked to beat"
    ),
    "family_matching_needs_a_request": (
        "an unconditional cell requests no family, so matching on the families its "
        "generations happened to carry would select the comparator on the outcome. No "
        "family-matched arm is reported for those cells"
    ),
    "gate_support_is_one_stream": (
        "the gate's cells are one decoding configuration and one batch-seed stream; "
        "its verdict counts are not the manuscript's three-stream replication counts "
        "and the two must not be quoted as one another"
    ),
    "novelty_is_an_alignment_statement": (
        "maximum identity to a searched corpus is a covariate. An alignment screen "
        "that finds nothing does not exclude profile-level homology and is not novelty"
    ),
}


def _state(interval: Sequence[float] | None) -> str:
    if interval is None:
        return "unresolved"
    low, high = float(interval[0]), float(interval[1])
    if low > 0.0:
        return "positive"
    if high < 0.0:
        return "negative"
    return "unresolved"


def comparator_table(
    gate: Mapping[str, Any], *, endpoint: str = "complete_domain"
) -> dict[str, Any]:
    """Per cell, the published fragment difference beside the whole-record one.

    The two comparators are measured on the same attempts, the same oracle and
    the same resampling unit, so the only thing that changes between the two
    columns is what a real protein is taken to be.
    """

    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown gate endpoint {endpoint!r}; declared: {list(ENDPOINTS)}")
    cells = gate.get("cells")
    if not cells:
        raise ValueError("the gate artefact carries no cell")
    rows: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda value: value["cell"]):
        block = cell["endpoints"][endpoint]
        record: dict[str, Any] = {
            "cell": cell["cell"],
            "arm": cell["arm"],
            "condition": cell["condition"],
            "n_attempts": int(cell["n_attempts"]),
            "n_near_duplicate_groups": int(cell["n_clusters"]),
            "model_rate": float(block["model_rate"]),
            "denominator_rule": block.get("denominator_rule"),
        }
        for comparator in NATURAL_COMPARATORS:
            arm = block["controls"][comparator]
            record[comparator] = {
                "control_rate": float(arm["control_rate"]),
                "difference": float(arm["difference"]),
                "ci95": [float(value) for value in arm["ci95"]],
                "verdict": _state(arm["ci95"]),
                "qualified": bool(arm.get("qualified", True)),
                "unit": arm.get("unit"),
            }
        record["verdict_changes"] = record["fragment"]["verdict"] != record["natural"]["verdict"]
        record["fragment_minus_natural_difference"] = (
            record["fragment"]["difference"] - record["natural"]["difference"]
        )
        record["distinct_families"] = cell.get("distinct_families")
        rows.append(record)
    counts = {
        comparator: dict(Counter(row[comparator]["verdict"] for row in rows))
        for comparator in NATURAL_COMPARATORS
    }
    changed = [row["cell"] for row in rows if row["verdict_changes"]]
    shifts = np.asarray([row["fragment_minus_natural_difference"] for row in rows], dtype=float)
    return {
        "schema_version": SCHEMA_VERSION,
        "endpoint": endpoint,
        "n_cells": len(rows),
        "cells": rows,
        "verdict_counts": counts,
        "cells_whose_verdict_changes": changed,
        "n_cells_whose_verdict_changes": len(changed),
        "fragment_minus_natural_difference": {
            "mean": float(shifts.mean()),
            "median": float(np.median(shifts)),
            "min": float(shifts.min()),
            "max": float(shifts.max()),
            "n_cells_fragment_easier_to_beat": int((shifts > 0).sum()),
            "reading": (
                "positive means the published fragment comparator reports a LARGER "
                "model-minus-control difference than the whole natural record does, so "
                "the fragment is the more generous comparator on that cell"
            ),
        },
        "sign_changes": {
            "n_positive_lost": sum(
                1
                for row in rows
                if row["fragment"]["verdict"] == "positive" and row["natural"]["verdict"] != "positive"
            ),
            "n_positive_gained": sum(
                1
                for row in rows
                if row["fragment"]["verdict"] != "positive" and row["natural"]["verdict"] == "positive"
            ),
        },
        "ceiling": dict(CEILING),
    }


# ------------------------------------------------- duplication, termination, novelty


#: An attempt this short is an immediate stop, not a product. Declared because a
#: native termination at zero or one residue is a decoder that refused to start,
#: and pooling it with finished products makes truncation look like the only way
#: generation fails.
IMMEDIATE_STOP_RESIDUES: int = 1

#: Below this a natively terminated attempt is a finished *fragment* rather than a
#: domain-sized product. Reported, never used to drop an attempt.
SHORT_NATIVE_PRODUCT_RESIDUES: int = 50


def _censored(row: Mapping[str, Any]) -> bool | None:
    """Whether one attempt hit the token budget, from the stop accounting alone.

    Deliberately blind to composition. Classifying an attempt by its residues
    first files a censored attempt that happens to carry a non-canonical residue
    under its composition instead, which undercounts censoring; the stop reason
    and the token count are the only evidence about termination.
    """

    tokens = row.get("generated_tokens")
    budget = row.get("effective_max_new_tokens")
    if tokens is not None and budget is not None:
        return int(tokens) >= int(budget)
    stop = row.get("decoder_stop")
    if stop is None:
        return None
    return str(stop) not in ("eos", "native_terminal", "stop_string")


def _termination(rows: Sequence[Mapping[str, Any]], stops: Counter) -> dict[str, Any]:
    """Termination behaviour, counted on the stop reason and never on composition."""

    flags = [_censored(row) for row in rows]
    censored = [row for row, flag in zip(rows, flags) if flag is True]
    native = [row for row, flag in zip(rows, flags) if flag is False]
    unknown = sum(1 for flag in flags if flag is None)
    native_lengths = [int(row["length"]) for row in native]
    censored_lengths = [int(row["length"]) for row in censored]
    return {
        "decoder_stop_counts": dict(sorted(stops.items())),
        "n_native_delimiter_observed": sum(
            1 for row in rows if row.get("native_delimiter_observed")
        ),
        "n_at_token_budget": len(censored),
        "n_natively_terminated": len(native),
        "n_termination_unknown": unknown,
        "classification_rule": (
            "an attempt is censored when its generated token count reaches the "
            "effective budget, otherwise when its stop reason is not a native stop. "
            "Composition is never consulted: classifying by residues first files a "
            "censored attempt carrying a non-canonical residue under its composition "
            "and undercounts censoring"
        ),
        "native_products": {
            "n_immediate_stop_at_or_below": IMMEDIATE_STOP_RESIDUES,
            "n_immediate_stop": sum(
                1 for value in native_lengths if value <= IMMEDIATE_STOP_RESIDUES
            ),
            "n_below_short_threshold": sum(
                1 for value in native_lengths if value < SHORT_NATIVE_PRODUCT_RESIDUES
            ),
            "short_threshold_residues": SHORT_NATIVE_PRODUCT_RESIDUES,
            "median_length_residues": (
                float(np.median(native_lengths)) if native_lengths else None
            ),
            "note": (
                "a native termination at or below one residue is a decoder that "
                "refused to start, not a product; pooling it with finished products "
                "makes truncation look like the only failure mode"
            ),
        },
        "censored_products": {
            "median_length_residues": (
                float(np.median(censored_lengths)) if censored_lengths else None
            ),
        },
        "length_matched_native_versus_censored": {
            "supported": False,
            "reason": (
                "for pure-protein arms a natively finished product has median length "
                "about 150-230 residues while a censored one sits at the token budget, "
                "so the two strata barely overlap in length and a length-matched "
                "native-versus-censored contrast has essentially no support. No "
                "endpoint here is stratified that way, and none may be"
            ),
        },
        "note": (
            "a native end delimiter in the decoded text is not the same event as the "
            "decoder stopping on its end token; both are counted"
        ),
    }


def cell_census(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Duplication, termination and novelty of one annotated attempt ledger.

    All three are properties the complete-domain rate cannot express. A cell can
    reach a high yield on one repeated sequence (duplication), can never finish a
    product at all (termination), or can reproduce a corpus record it was trained
    on (novelty), and each reading changes what the yield means.
    """

    if not rows:
        raise ValueError("a census needs at least one attempt")
    digests = [row["sequence_sha256"] for row in rows]
    counts = Counter(digests)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=float)
    identities = [
        float(row["reference_identity"])
        for row in rows
        if row.get("reference_identity") is not None
    ]
    searched = sum(
        1 for row in rows if row.get("reference_search_status") not in (None, "not_searched")
    )
    stops = Counter(str(row.get("decoder_stop")) for row in rows)
    census: dict[str, Any] = {
        "n_attempts": len(rows),
        "duplication": {
            "n_distinct_sequences": len(counts),
            "exact_duplication_rate": 1.0 - len(counts) / len(rows),
            "largest_exact_duplicate_share": max(counts.values()) / len(rows),
            "n_sequences_seen_more_than_once": sum(1 for value in counts.values() if value > 1),
        },
        "termination": _termination(rows, stops),
        "length_residues": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "n_zero_length": int((lengths == 0).sum()),
        },
        "novelty_covariate": {
            "n_searched": searched,
            "n_with_reported_identity": len(identities),
            "max_identity_percent": max(identities) if identities else None,
            "median_identity_percent": float(np.median(identities)) if identities else None,
            "n_at_or_above_95_percent": sum(1 for value in identities if value >= 95.0),
            "status": "measured" if identities else "not_run",
            "note": CEILING["novelty_is_an_alignment_statement"],
        },
    }
    profiles = [row.get("profile") for row in rows if isinstance(row.get("profile"), Mapping)]
    if profiles:
        census["recognition"] = {
            cohort: {
                "any_family_rate": float(
                    np.mean([1.0 if block[cohort]["any_family"] else 0.0 for block in profiles])
                ),
                "complete_domain_rate": float(
                    np.mean([1.0 if block[cohort]["complete_domain"] else 0.0 for block in profiles])
                ),
                "n_scored": len(profiles),
            }
            for cohort in ("generated", "fragment")
            if all(cohort in block for block in profiles)
        }
    return census


def census_over_cells(ledgers: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """The census of every supplied cell, plus what it says across cells."""

    per_cell = {name: cell_census(rows) for name, rows in sorted(ledgers.items())}
    duplication = np.asarray(
        [block["duplication"]["exact_duplication_rate"] for block in per_cell.values()],
        dtype=float,
    )
    censored = np.asarray(
        [block["termination"]["n_at_token_budget"] for block in per_cell.values()], dtype=float
    )
    attempts = np.asarray([block["n_attempts"] for block in per_cell.values()], dtype=float)
    return {
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "across_cells": {
            "exact_duplication_rate": {
                "mean": float(duplication.mean()),
                "max": float(duplication.max()),
                "cells_above_one_tenth": sorted(
                    name
                    for name, block in per_cell.items()
                    if block["duplication"]["exact_duplication_rate"] > 0.10
                ),
            },
            "token_budget_censoring": {
                "mean_attempts_per_cell": float(censored.mean()),
                "mean_share_per_cell": float((censored / attempts).mean()),
                "share_of_attempts_pooled": float(censored.sum() / attempts.sum()),
                "n_cells": len(per_cell),
                "n_attempts": int(attempts.sum()),
                "scope_note": (
                    "the pooled share is dominated by the conditional cells, which "
                    "carry 3,200 attempts against 800 for an unconditioned cell, so "
                    "the mean per-cell share is the comparable figure and the cell "
                    "set is named with it"
                ),
                "reading": (
                    "the share of the published yield's denominator that consists of "
                    "budget-censored continuations rather than finished products"
                ),
            },
            "native_products": {
                "n_immediate_stop": sum(
                    block["termination"]["native_products"]["n_immediate_stop"]
                    for block in per_cell.values()
                ),
                "n_below_short_threshold": sum(
                    block["termination"]["native_products"]["n_below_short_threshold"]
                    for block in per_cell.values()
                ),
                "cells_whose_native_terminations_are_all_immediate": sorted(
                    name
                    for name, block in per_cell.items()
                    if block["termination"]["n_natively_terminated"] > 0
                    and block["termination"]["native_products"]["n_immediate_stop"]
                    == block["termination"]["n_natively_terminated"]
                ),
            },
        },
    }


# -------------------------------------------------- the family-matched natural draw


def pfam_members(path: Path, families: Iterable[str]) -> dict[str, set[str]]:
    """``accession -> matched families`` from the staged InterPro Pfam residue map.

    Read once over the whole table for every requested family, because the table
    is 840k rows and the alternative is one pass per class.
    """

    wanted = {str(value).split(".", 1)[0] for value in families}
    if not wanted:
        raise ValueError("a membership lookup needs at least one family")
    members: dict[str, set[str]] = defaultdict(set)
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().split("\t")
        if [field.strip() for field in header] != ["uniprot", "start", "end", "pfam_id"]:
            raise ValueError(f"{path} is not the expected Pfam residue table: {header}")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                continue
            family = fields[3].split(".", 1)[0]
            if family in wanted:
                members[fields[0]].add(family)
    if not members:
        raise ValueError(
            f"{path} carries no protein for any of {sorted(wanted)}; the family-matched "
            "draw would be empty and is refused rather than reported as a zero rate"
        )
    return dict(members)


def swissprot_accession(header: str) -> str | None:
    """The accession of a Swiss-Prot FASTA header, or None if it is not one."""

    fields = header.split("|")
    return fields[1] if len(fields) >= 3 and fields[0] in ("sp", "tr") else None


def family_matched_draw(
    *,
    classes: Mapping[str, Mapping[str, Any]],
    length_bands: Mapping[str, tuple[int, int]],
    records: Iterable[tuple[str, str]],
    members: Mapping[str, set[str]],
    per_class: int = DRAW_PER_CLASS,
    seed: int = DRAW_SEED,
) -> dict[str, Any]:
    """Full-length Swiss-Prot proteins of each requested class, in its length band.

    ``classes`` maps a class key to its frozen record (its Pfam ``referent``).
    ``length_bands`` gives the inclusive residue band the class's own generated
    attempts occupied, so the comparator is matched on length as well as family.
    The draw is a seeded permutation of the canonically sorted eligible list, as
    every other draw in this package is; it is never the head of the corpus file.
    """

    if per_class < 1:
        raise ValueError("a draw needs at least one record per class")
    wanted: dict[str, set[str]] = {}
    without_referent: list[str] = []
    for class_key, record in classes.items():
        referent = {str(value).split(".", 1)[0] for value in record["referent"]}
        if not referent:
            # A class whose referent draw carried no Pfam family at the declared share
            # is an unmeasurable class: there is no profile set a family-matched
            # comparator could be drawn against. It is reported, not approximated.
            without_referent.append(class_key)
            continue
        wanted[class_key] = referent
    if not wanted:
        raise ValueError(
            "no supplied class carries a Pfam referent, so no family-matched draw "
            "exists for any of them"
        )
    missing = sorted(set(wanted) - set(length_bands))
    if missing:
        raise ValueError(f"no generated length band for classes {missing}")

    eligible: dict[str, list[tuple[str, str]]] = {key: [] for key in wanted}
    n_records = 0
    for header, sequence in records:
        n_records += 1
        accession = swissprot_accession(header)
        if accession is None:
            continue
        carried = members.get(accession)
        if not carried:
            continue
        length = len(sequence)
        for class_key, referent in wanted.items():
            if not (carried & referent):
                continue
            low, high = length_bands[class_key]
            if low <= length <= high:
                eligible[class_key].append((accession, sequence))

    rng_root = int(seed)
    draw: list[dict[str, Any]] = []
    per_class_record: dict[str, Any] = {}
    for class_key in sorted(wanted):
        candidates = sorted(eligible[class_key])
        offset = int.from_bytes(hashlib.sha256(class_key.encode("utf-8")).digest()[:4], "big")
        rng = np.random.default_rng((rng_root + offset) % (2**31 - 1))
        order = rng.permutation(len(candidates)) if candidates else np.asarray([], dtype=int)
        taken = [candidates[int(index)] for index in order[:per_class]]
        per_class_record[class_key] = {
            "referent": sorted(wanted[class_key]),
            "length_band_residues": list(length_bands[class_key]),
            "n_eligible": len(candidates),
            "n_drawn": len(taken),
            "shortfall": max(0, per_class - len(taken)),
            "shortfall_reason": (
                None
                if len(taken) >= per_class
                else (
                    f"only {len(candidates)} Swiss-Prot records carry a referent family "
                    f"inside the generated length band {length_bands[class_key]}; the "
                    "realised count is reported and the band is never widened after a "
                    "rate has been seen"
                )
            ),
        }
        for index, (accession, sequence) in enumerate(taken):
            draw.append(
                {
                    "id": f"nfm_{class_key}_{index:04d}".replace(".", "_"),
                    "class_key": class_key,
                    # The arm travels on the record so a recognition run can join the
                    # frozen class-to-referent map without re-reading the draw.
                    "arm": str(classes[class_key].get("arm") or ""),
                    "accession": accession,
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                    "length": len(sequence),
                    "declared_families": sorted(members[accession] & wanted[class_key]),
                    "role": "family_matched_natural",
                }
            )
    if not draw:
        raise ValueError(
            "the family-matched natural draw is empty for every class; it is refused "
            "rather than reported as a zero comparator rate"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "family_matched_natural",
        "ceiling": CEILING["family_matched_is_a_ceiling"],
        "draw_seed": int(seed),
        "per_class_target": int(per_class),
        "length_match_tolerance": LENGTH_MATCH_TOLERANCE,
        "corpus_records_read": n_records,
        "classes_without_referent": sorted(without_referent),
        "classes_without_referent_reason": (
            "the class's referent draw carried no Pfam family at the declared share, so "
            "there is no profile set a family-matched comparator could be drawn against; "
            "the class is unmeasurable for this arm, not failing"
        ),
        "per_class": per_class_record,
        "n_drawn": len(draw),
        "records": draw,
    }


def length_band(
    rows: Sequence[Mapping[str, Any]], *, tolerance: float = LENGTH_MATCH_TOLERANCE
) -> tuple[int, int]:
    """The inclusive residue band a cell's non-empty attempts occupy, widened by ``tolerance``.

    A band rather than a per-attempt match because the comparator is a whole
    natural protein and cannot be cut to an exact length without becoming the
    fragment comparator this analysis exists to question.
    """

    lengths = [int(row["length"]) for row in rows if int(row["length"]) > 0]
    if not lengths:
        raise ValueError("a length band needs at least one non-empty attempt")
    low = max(1, int(round(min(lengths) * (1.0 - tolerance))))
    high = int(round(max(lengths) * (1.0 + tolerance)))
    return low, high


def normalise_recognition(
    rows: Iterable[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """One recognition shape, whatever spelling the producer used.

    The oracle stage's sidecar writes ``pfam_families`` / ``any_profile_hit``,
    while the in-memory result of ``family_oracle.recognise`` uses ``families`` /
    ``any_family``. Reading one shape and being handed the other silently yields
    all-zero rates -- a comparator that looks measured and is not -- so the two
    are reconciled in one place and a row carrying neither is refused.
    """

    normalised: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row["id"])
        if "families" in row:
            families = row["families"]
        elif "pfam_families" in row:
            families = row["pfam_families"]
        else:
            raise ValueError(
                f"recognition record {identifier!r} carries neither 'families' nor "
                "'pfam_families'; it is not a recognition record"
            )
        families = list(families or ())
        normalised[identifier] = {
            "families": families,
            "any_family": bool(
                row.get("any_family", row.get("any_profile_hit", bool(families)))
            ),
            "complete_domain": bool(row.get("complete_domain", False)),
            "best_profile_coverage": row.get("best_profile_coverage"),
        }
    if not normalised:
        raise ValueError("no recognition record was supplied")
    return normalised


def natural_recognition_rates(
    draw: Mapping[str, Any],
    recognition: Mapping[str, Mapping[str, Any]],
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The family-matched ceiling: per class, the rate on real full-length proteins.

    ``records`` carries the drawn sequences. The draw's completion JSON keeps only
    their accounting -- the records themselves live in the sibling
    ``natural_family_matched.jsonl`` -- so a caller that has read the JSON must
    pass them, and a call with neither is refused rather than reporting a ceiling
    over nothing.

    A drawn record the oracle never searched is a defect, not a non-hit, and is
    reported: the whole point of the arm is that these sequences are real proteins
    of the requested family, so an unexplained miss means the recognition run and
    the draw are not the same set.
    """

    if records is None:
        records = draw.get("records")
    if not records:
        raise ValueError(
            "the family-matched draw carried no records. The draw's completion JSON "
            "keeps only the accounting; pass the records from the sibling "
            "natural_family_matched.jsonl"
        )
    # The caller keys by identifier; the record itself need not repeat it.
    recognition = normalise_recognition(
        {**row, "id": key} for key, row in recognition.items()
    )
    unscored = sorted(
        record["id"] for record in records if record["id"] not in recognition
    )
    per_class: dict[str, Any] = {}
    for record in records:
        per_class.setdefault(record["class_key"], []).append(record)
    rates: dict[str, Any] = {}
    for class_key, group in sorted(per_class.items()):
        hits = [recognition.get(record["id"], {}) for record in group]
        referent = {
            str(value).split(".", 1)[0] for value in draw["per_class"][class_key]["referent"]
        }
        target = [
            any(family in referent for family in block.get("families", ())) for block in hits
        ]
        rates[class_key] = {
            "n_drawn": len(group),
            "any_family_rate": float(
                np.mean([1.0 if block.get("any_family") else 0.0 for block in hits])
            ),
            "complete_domain_rate": float(
                np.mean([1.0 if block.get("complete_domain") else 0.0 for block in hits])
            ),
            "target_family_rate": float(np.mean([1.0 if value else 0.0 for value in target])),
            "median_length_residues": float(
                np.median([record["length"] for record in group])
            ),
        }
    return {
        "status": "measured",
        "n_unscored_records": len(unscored),
        "unscored_records": unscored[:10],
        "per_class": rates,
        "over_classes": {
            field: {
                "mean": float(np.mean([block[field] for block in rates.values()])),
                "min": float(np.min([block[field] for block in rates.values()])),
                "max": float(np.max([block[field] for block in rates.values()])),
            }
            for field in ("any_family_rate", "complete_domain_rate", "target_family_rate")
        },
        "ceiling": CEILING["family_matched_is_a_ceiling"],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} carries no record")
    return rows
