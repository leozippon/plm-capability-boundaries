#!/usr/bin/env python3
"""ProGen3 robustness campaign: achieved upper bounds on the pairwise contrasts, reported descriptively.

This is a sibling of ``analyse_e20_confirmatory.py`` and reuses its machinery wherever a function does not
depend on that module's ``MODELS`` table. It is not a second copy of E20's verdict machinery and does not
pretend to be one: the five-arm design (N, Pr, M, C, Cx) has no F cell, E20's verdict requires ``C-Pr`` and
``F-M`` both below margin, so that verdict can only ever be ``unestimable`` here. Adding an F arm after
learning the verdict cannot form would be outcome-adjacent protocol modification, so it was declined and the
pairwise question is answered pairwise instead. **No verdict and no pass/fail flag is issued anywhere.**

What is reported, per rung and pooled within a measure line:

* achieved upper 95% BCa bounds on ``C-Pr``, ``C-M``, ``C-Cx``, ``Pr-N`` and ``M-N``, on E20's primary TM
  endpoint and on its co-primary admissible-share endpoint;
* the captured fraction of the selection-oracle headroom per selector at k = 10, on E20's definitions;
* pool diversity, from E20's own cell summary;
* the precision control this campaign has and E20 does not: ``progen3-112m`` in float16 against the same
  checkpoint in bfloat16, at the same arms and the same doses.

``C-M`` is coupling against fields-only at matched measured intervention. It is **not** E20's ``F-M`` and is
never a substitute for it; the artifact labels it as its own contrast and says so.

The independence unit is the domain. Eight domains; a contrast resting on fewer carries no interval, exactly
as in the frozen script, whose BCa coverage note is carried forward unchanged. Precision is part of the
native contract, so the two bfloat16 rungs are one measure line and the float16 rung is another; strata never
pool across a measure line.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse_e20_confirmatory as e20  # noqa: E402
from src.capability.context.family_potts import load_family_model  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402

SCHEMA = "progen3_robustness_bounds_v1"
INPUTS_SCHEMA = "progen3_robustness_inputs_v1"

#: The campaign's five arms, in report order. F and Cd are outside this design by specification.
ARMS = ("N", "Pr", "M", "C", "Cx")

#: ``(name, left, right, what it reads)``. Descriptive; no contrast here carries a flag.
CONTRASTS = (
    ("C-Pr", "C", "Pr", "coupling guidance against conventional first-order (profile) guidance; the same "
                        "contrast E20 reports"),
    ("C-M", "C", "M", "coupling against fields-only at matched measured intervention: the pairwise-specific "
                      "contrast a five-arm design admits. It is NOT E20's F-M and is not a substitute for it. "
                      "F combines fields and coupling and has no cell in this design, so F-M cannot be formed; "
                      "C-M differences a coupling-only arm against a fields-only arm, which is a different "
                      "question about a different pair of arms."),
    ("C-Cx", "C", "Cx", "coupling against its column-permuted control: the matched incorrect-constraint "
                        "comparison, at the dose that matches its measured intervention"),
    ("Pr-N", "Pr", "N", "first-order guidance against unguided decoding"),
    ("M-N", "M", "N", "fields-only guidance against unguided decoding"),
)

#: The two endpoints, both E20's own. ``feasibility`` is the frozen field name for the second one and is a
#: misnomer: it is the share of candidates clearing the 20% identity restriction and no predicted confidence
#: enters it, so the artifact names it for what it measures and records the frozen field name beside it.
ENDPOINTS = (
    ("primary_tm", "tm", "mean TM-score over candidates with identity to the wild type strictly below "
                         f"{e20.ADMISSIBLE_IDENTITY:g}%, at least {e20.MIN_ADMISSIBLE} such candidates or the cell "
                         "has no value and is never imputed"),
    ("novelty_admissibility_rate", "feasibility", "share of the pool clearing the 20% identity restriction "
                                                  "(n_admissible / n). Carried under the frozen field name "
                                                  "`feasibility`, which is a misnomer: no predicted confidence "
                                                  "enters this endpoint."),
)

STATEMENT = (
    "Descriptive robustness report. This script issues NO VERDICT and NO PASS/FAIL FLAG on any contrast, "
    "endpoint, rung or measure line. It reports achieved upper 95% BCa bounds and the numbers behind them. "
    "E20's 0.05 TM margin is quoted as a reference scale only, because it is the scale E20 declared; nothing "
    "here is compared against it to reach a conclusion, and no p-value is issued (the p-value fields of the "
    "frozen record shape are present and always null). E20's verdict machinery is deliberately absent: its "
    "verdict requires C-Pr and F-M both below margin, this five-arm design has no F cell, so that verdict "
    "could only be 'unestimable' and reproducing it would say nothing. C-M is reported instead as its own "
    "contrast and is not a substitute for F-M. The independence unit is the domain; a contrast resting on "
    f"fewer than {MINIMUM_BOOTSTRAP_UNITS} domains carries no interval."
)

REFERENCE_SCALE = (
    f"E20 declared a {e20.MARGIN} TM equivalence margin. It is reproduced here as a reference scale so a bound "
    "can be read against something, and for no other purpose: no bound is tested against it, no flag is "
    "derived from it, and a bound below it is not a conclusion."
)

#: The second co-primary's own caveat, carried forward from the corrected E20 reading.
ADMISSIBILITY_NOTE = (
    "A novelty-admissibility rate, not a feasibility or confidence rate. Guidance that moves candidates toward "
    "the reference family leaves fewer of them novel, so a negative bound on this endpoint is the "
    "quality-novelty trade-off measured on the novelty axis and is not a statement about predicted structure."
)

#: E20's CAPTURE_DEVELOPMENT_ONLY names E20's own cell depths (200 primary, 100 frontier) and its frontier
#: cells, neither of which exists here, so this campaign states its own equivalent rather than importing that
#: sentence. Everything else in the captured-fraction block is the frozen module's.
CAPTURE_DEPTH_NOTE = (
    "Budget scaling -- whether a selector's gain grows with the candidate budget -- is development-only and is "
    "not measured here. Every cell in this campaign carries the same 200 candidates, so only the captured "
    "fraction at that one depth can be read."
)

DOSE_FLOOR_STATEMENT = (
    "Declared context, not a finding. A dose cell is listed when the calibration's own per-domain floor margin "
    "falls below the margin it declared as required, i.e. the intervention that domain's family reference "
    "admits at the calibrated dose is at or near the declared noise floor. These are properties of those "
    "family references -- their column structure and the strength the fit can express -- not of ProGen3: "
    "ProGen2-medium showed the same weakness at the same columns in E20. Any per-domain number for a listed "
    "cell must be read next to this list."
)

INTERVENTION_BASIS_NOTE = (
    "The records of this campaign carry no per-candidate guidance_strength field, because ProGen3's forward is "
    "not prefix-exact and a teacher-forced replay would not measure what the sampler observed. The measured "
    "intervention is therefore the calibration's wild-type-basis per-residue KL, solved before any endpoint "
    "existed to match ProGen2-medium's measured KL per arm, and it lives in the calibration artifacts rather "
    "than in the records. The per-cell kl_per_residue field of the frozen cell summary is consequently null "
    "throughout; it is not a missing measurement."
)

PRECISION_STATEMENT = (
    "The precision control. progen3-112m is run in float16 at the doses solved for it in bfloat16, so the "
    "kernel precision is the only thing that moves between the two rungs. The contrast is float16 minus "
    "bfloat16, per arm, over the same eight domains, on both endpoints. It tests directly whether a bfloat16 "
    "numerical floor could drive a conclusion rather than leaving that as a caveat. No verdict and no flag."
)

POOLED_PREFIX = "pooled_"


# ------------------------------------------------------------------ inputs


def parse_rung(label: str, precisions) -> tuple[str, str]:
    """``'progen3-112m-float16' -> ('progen3-112m', 'float16')``, refusing anything that does not parse.

    The label is the one ``collect_guided_decoding_pool.recorded_model_label`` writes: the checkpoint's arm
    name, a hyphen, and the amended dtype. ``precisions`` is the set of dtypes the passed protocols declare,
    so the vocabulary comes from the campaign's own declarations rather than from a constant here.
    """

    base, _, precision = label.rpartition("-")
    if not base or precision not in precisions:
        raise ValueError(
            f"model label {label!r} does not parse as '<checkpoint>-<precision>' with precision in "
            f"{sorted(precisions)}; the amended labels carry the precision and an unparsed label cannot be "
            "assigned to a measure line"
        )
    return base, precision


def load_line(records_path: Path, protocol_path: Path, precisions) -> dict:
    """One measure line: its records, its protocol's dose declaration, and the rungs it holds.

    Every record in the file must carry a label of this protocol's declared precision. That is what keeps the
    two bfloat16 rungs together and the float16 rung apart: a measure line is a precision, so a file holding
    two precisions is refused rather than silently merged.
    """

    records = _records(records_path)
    document = json.loads(fr.require_asset(protocol_path, "--protocol").read_text())
    declared = fr.load_strengths(protocol_path, "--protocol")
    declaration = fr.parse_strength_declaration(declared)
    precision = document["native_contract"]["dtype"]
    if precision not in precisions:
        raise ValueError(f"{protocol_path}: declared dtype {precision!r} is not among {sorted(precisions)}")
    # The label is parsed before the strength contract is checked, so a label this campaign cannot place is
    # reported as an unparsable label rather than as a (model, arm) with no declared dose.
    bases = {}
    for label in sorted({r["model"] for r in records}):
        base, found = parse_rung(label, precisions)
        if found != precision:
            raise ValueError(
                f"{records_path}: record label {label!r} is {found} but the protocol declares {precision}; a "
                "measure line is one precision and these records must be measured on their own line"
            )
        bases[label] = base
    if not bases:
        raise ValueError(f"{records_path} holds no records")
    fr.require_frozen_strength(records, declared)
    return {"precision": precision, "records_path": records_path, "protocol_path": protocol_path,
            "records": records, "declaration": declaration, "roles": fr.load_roles(protocol_path, "--protocol"),
            "domains_declared": document["domains"], "rungs": tuple(bases), "bases": bases,
            "candidates_per_cell": document["sequences_per_cell"]}


def _records(path: Path) -> list[dict]:
    """The per-sequence records of one measure line, through the sibling loader the frozen script uses."""

    return e20.sf.load_records(fr.require_asset(path, "--records"))


def declared_precisions(protocol_paths) -> frozenset[str]:
    """The dtype vocabulary, taken from the passed protocols' own ``native_contract.dtype`` declarations."""

    return frozenset(json.loads(fr.require_asset(p, "--protocol").read_text())["native_contract"]["dtype"]
                     for p in protocol_paths)


def load_campaign(records_paths, protocol_paths) -> dict:
    """Every measure line, with the cross-line invariants checked and nothing merged across a precision."""

    if len(records_paths) != len(protocol_paths):
        raise ValueError(f"{len(records_paths)} --records and {len(protocol_paths)} --protocol; pass one "
                         "protocol per records file, in the same order, one pair per measure line")
    precisions = declared_precisions(protocol_paths)
    lines = [load_line(r, p, precisions) for r, p in zip(records_paths, protocol_paths)]
    if len({line["precision"] for line in lines}) != len(lines):
        raise ValueError("two measure lines declare the same precision; a precision is a measure line")
    reference = lines[0]["declaration"]
    for line in lines[1:]:
        if line["declaration"] != reference:
            raise ValueError(f"{line['protocol_path']} declares different strengths from "
                             f"{lines[0]['protocol_path']}; the rungs share one dose declaration and a "
                             "divergence means the doses are no longer matched")
    # Report order is the protocol's own declaration order, not alphabetical.
    declared_rungs = tuple(dict.fromkeys(model for model, _arm in reference))
    present = {label for line in lines for label in line["rungs"]}
    if present != set(declared_rungs):
        raise ValueError(f"rungs present {sorted(present)} but the protocol declares {list(declared_rungs)}; "
                         "a missing rung is refused, not analysed around")
    for line in lines:
        line["rungs"] = tuple(label for label in declared_rungs if label in line["rungs"])
    ids = [r["id"] for line in lines for r in line["records"]]
    if len(set(ids)) != len(ids):
        raise ValueError("a candidate id appears in more than one measure line; the lines must be disjoint")
    doses = {key: declaration[0] for key, declaration in reference.items()}
    records = [r for line in lines for r in line["records"]]
    return {"lines": lines, "records": records, "doses": doses, "rungs": declared_rungs,
            "bases": {label: base for line in lines for label, base in line["bases"].items()},
            "precision_of": {label: line["precision"] for line in lines for label in line["rungs"]}}


def check_design(campaign: dict, cells: "Cells") -> dict:
    """Refuse a hole: every rung, arm, domain and cell depth the campaign declares must be present.

    Reported as the shapes block of the inputs artifact and checked identically in a full run, so a dry run
    that passes is evidence the full run will read the same cells.
    """

    domains = sorted({r["domain"] for r in campaign["records"]})
    if domains != sorted(e20.SEALED_DOMAINS):
        raise ValueError(f"domains {domains} are not the sealed eight {sorted(e20.SEALED_DOMAINS)}")
    shapes = {}
    for line in campaign["lines"]:
        evaluation = sorted(d for d, role in line["roles"].items() if role == "evaluation")
        if evaluation != sorted(e20.SEALED_DOMAINS):
            raise ValueError(f"{line['protocol_path']}: evaluation domains {evaluation} are not the sealed eight")
        for record in line["records"]:
            declared_length = line["domains_declared"][record["domain"]]["L"]
            if len(record["sequence"]) != declared_length:
                raise ValueError(f"{record['id']}: sequence of length {len(record['sequence'])} against the "
                                 f"{declared_length} columns the protocol declares for {record['domain']}")
        for label in line["rungs"]:
            arms = sorted({arm for model, arm in campaign["doses"] if model == label})
            if arms != sorted(ARMS):
                raise ValueError(f"{label}: declared arms {arms} are not this design's {sorted(ARMS)}")
            per_cell = {}
            for arm in ARMS:
                for domain in e20.SEALED_DOMAINS:
                    summary, reason = cells.cell(label, domain, arm)
                    if summary is None:
                        raise ValueError(f"{label}/{domain}/{arm}: {reason}")
                    if summary["n"] != line["candidates_per_cell"]:
                        raise ValueError(f"{label}/{domain}/{arm}: {summary['n']} candidates against the declared "
                                         f"{line['candidates_per_cell']}; the campaign is incomplete")
                    per_cell[f"{arm}/{domain}"] = summary["n"]
            shapes[label] = {
                "measure_line": line["precision"], "base_checkpoint": campaign["bases"][label],
                "n_records": sum(1 for r in campaign["records"] if r["model"] == label),
                "arms": arms, "n_arms": len(arms), "domains": list(e20.SEALED_DOMAINS),
                "n_domains": len(e20.SEALED_DOMAINS), "n_cells": len(per_cell),
                "candidates_per_cell": sorted(set(per_cell.values())),
                "doses": {arm: list(campaign["doses"][(label, arm)]) for arm in arms},
            }
    return {"domains": domains, "rungs": shapes}


# ------------------------------------------------------------------ cells


class Cells:
    """Per ``(rung, domain, arm)`` cell summary at the protocol-declared frozen dose.

    Reimplemented from ``analyse_e20_confirmatory.Sources``, which cannot be reused: it resolves a cell
    through that module's ``FROZEN_DOSES`` and ``REFERENCE_MODEL_LABELS`` tables, both keyed on E20's two
    model labels and the latter carrying P and H arms this design does not have. Only the lookup is rewritten;
    the dose table is the campaign protocol's own ``strengths`` declaration so the doses keep one source, and
    the cell summary itself -- the admissibility rule, the 25-candidate floor, the identity companions, the
    composition controls and the pool diversity -- is the frozen ``summarise_cell``.
    """

    def __init__(self, records, doses):
        self.doses = doses
        self.cells = {key: e20.summarise_cell(rows) for key, rows in e20.group_cells(records).items()}

    def cell(self, rung, domain, arm):
        """``(summary, None)`` or ``(None, reason)``, the getter shape ``model_contrast`` expects."""

        if (rung, arm) not in self.doses:
            return None, f"no declared frozen dose for {rung}/{arm}"
        dose = self.doses[(rung, arm)]
        key = (rung, domain, arm, *dose)
        if key in self.cells:
            return self.cells[key], None
        return None, f"no {arm} cell for {rung} at dose {list(dose)}"


# ------------------------------------------------------------------ contrasts


def pool_over(by_token: dict, tokens) -> tuple[dict, dict]:
    """Per-domain mean over ``tokens``; a domain missing under any token is missing and never imputed.

    Reimplemented from ``analyse_e20_confirmatory.pooled_contrast``, which iterates that module's ``MODELS``.
    """

    domains = set().union(*(set(per) | set(missing) for per, missing in by_token.values()))
    per, missing = {}, {}
    for domain in sorted(domains):
        parts = [by_token[t][0].get(domain) for t in tokens]
        if any(part is None for part in parts):
            missing[domain] = "; ".join(f"{t}: {by_token[t][1][domain]}" for t in tokens
                                        if domain in by_token[t][1])
            continue
        per[domain] = {k: float(np.mean([part[k] for part in parts])) for k in parts[0]}
    return per, missing


def contrast_table(domains, spec, tokens, metric, getter, pooled_label: str | None) -> dict:
    """``{stratum: {contrast: estimate}}`` over ``tokens``, optionally with a pooled stratum.

    Reimplemented from ``analyse_e20_confirmatory.contrast_table``, which fans out over that module's
    ``MODELS`` and ``STRATA``. The per-token differencing is the frozen ``model_contrast`` (its ``model``
    parameter is an opaque token handed to ``getter``) and the reduction over domains is the frozen
    ``estimate``, so the arithmetic, the eight-domain floor and the BCa interval are all the frozen ones.
    """

    strata = {token: {} for token in tokens}
    if pooled_label is not None:
        strata[pooled_label] = {}
    for name, left, right, _reads in spec:
        by_token = {t: e20.model_contrast(domains, t, left, right, metric, getter) for t in tokens}
        for token in tokens:
            strata[token][name] = e20.estimate(*by_token[token])
        if pooled_label is not None:
            strata[pooled_label][name] = e20.estimate(*pool_over(by_token, tokens))
    return {stratum: e20.strip_p(records) for stratum, records in strata.items()}


def upper_bound(record: dict) -> float | None:
    """The achieved upper 95% bound, or None where the contrast is unestimable. No flag is derived from it."""

    return record["interval95_bca"][1] if record["status"] == "estimated" else None


# ------------------------------------------------------------------ captured fraction of the oracle headroom


def captured_fraction(records, rungs, doses, family_models, permutation_seed: int) -> dict:
    """E20's captured fraction of the selection-oracle headroom, fanned out over ``rungs`` instead of MODELS.

    Reimplemented from ``analyse_e20_confirmatory.captured_fraction`` for that one reason: it loops over that
    module's ``MODELS`` and indexes its ``FROZEN_DOSES``. Every definition inside is the frozen module's --
    ``group_cells``, the admissibility rule and its 25-candidate floor, ``selector_view``, ``selector_scores``,
    ``tm_oracle_gain``, ``bca_interval``, the selector roster, k, the declared expectation and the statement
    that the TM oracle is a ceiling measurement and not a selector. Within a ``(rung, domain)`` cell the pool
    is the admissible candidates; a domain's gains are the means over the rungs of the stratum and its
    captured fraction is the ratio of those mean gains; the interval is BCa over domains, withheld below the
    unit floor.
    """

    cells = e20.group_cells(records)
    domains = sorted({r["domain"] for r in records} & set(e20.SEALED_DOMAINS))
    names = list(e20.SELECTOR_NAMES)
    out = {"label": e20.CAPTURE_LABEL, "rungs": list(rungs), "k": e20.SELECTION_K,
           "expectation": e20.CAPTURE_EXPECTATION, "oracle_statement": e20.CAPTURE_ORACLE_STATEMENT,
           "depth_note": CAPTURE_DEPTH_NOTE, "interval_coverage_note": e20.BCA_NOTE,
           "no_verdict": "outside any decision; no threshold and no flag is attached", "arms": {}}
    for arm in e20.CAPTURE_ARMS:
        per, missing = {}, {}
        for domain in domains:
            selector_gain = {name: [] for name in names}
            oracle, sizes = [], {}
            for rung in rungs:
                rows = cells.get((rung, domain, arm, *doses[(rung, arm)]))
                admissible = [r for r in rows or [] if r["identity_to_wild_type"] < e20.ADMISSIBLE_IDENTITY]
                if len(admissible) < e20.MIN_ADMISSIBLE or domain not in family_models:
                    missing[domain] = (f"{rung}: no {arm} pool with {e20.MIN_ADMISSIBLE} admissible candidates "
                                       "and a family model")
                    break
                sizes[rung] = {"pool": len(rows), "admissible": len(admissible)}
                scores = e20.selector_scores(names, e20.selector_view(admissible), family_models[domain],
                                             permutation_seed)
                pool_mean = float(np.mean([r["tm_score"] for r in admissible]))
                for name in names:
                    top = np.argsort(-scores[name], kind="stable")[:e20.SELECTION_K]
                    selector_gain[name].append(
                        float(np.mean([admissible[i]["tm_score"] for i in top])) - pool_mean)
                oracle.append(e20.tm_oracle_gain(admissible))
            else:
                headroom = float(np.mean(oracle))
                per[domain] = {"oracle_gain": headroom, "pool_sizes": sizes, "selectors": {
                    name: {"gain": float(np.mean(selector_gain[name])),
                           "captured_fraction": (float(np.mean(selector_gain[name])) / headroom
                                                 if headroom > 0 else None)}
                    for name in names}}
        floor = bootstrap_unit_floor(len(per))
        block = {"n_domains": len(per), "per_domain": per, "missing_domains": missing,
                 "mean_oracle_gain": e20._mean([v["oracle_gain"] for v in per.values()]), "selectors": {}}
        for name in names:
            fractions = [v["selectors"][name]["captured_fraction"] for v in per.values()]
            entry = {"mean_gain": e20._mean([v["selectors"][name]["gain"] for v in per.values()]),
                     "mean_captured_fraction": e20._mean(fractions)}
            if floor["degenerate"] or None in fractions:
                entry.update(status="unestimable", interval95_bca=None,
                             reason=(f"{len(per)} domains have a value, below the {MINIMUM_BOOTSTRAP_UNITS}-domain "
                                     "floor") if floor["degenerate"]
                             else "an oracle gain was not positive in some domain")
            else:
                entry.update(status="estimated", interval95_bca=e20.bca_interval(fractions)[0])
            block["selectors"][name] = entry
        out["arms"][arm] = block
    return out


# ------------------------------------------------------------------ pool diversity


def pool_diversity(cells: Cells, rungs, domains) -> dict:
    """Per rung and arm, the mean over domains of the frozen cell summary's within-cell diversity fields."""

    out = {}
    for rung in rungs:
        out[rung] = {}
        for arm in ARMS:
            found = [cells.cell(rung, domain, arm)[0] for domain in domains]
            present = [summary for summary in found if summary is not None]
            out[rung][arm] = {
                "n_cells": len(present),
                "mean_pairwise_identity": e20._mean([s["mean_pairwise_identity"] for s in present]),
                "near_duplicate_rate": e20._mean([s["near_duplicate_rate"] for s in present]),
                "per_domain": {domain: {"mean_pairwise_identity": summary["mean_pairwise_identity"],
                                        "near_duplicate_rate": summary["near_duplicate_rate"]}
                               for domain, summary in zip(domains, found) if summary is not None},
            }
    return {"definition": f"within-cell mean pairwise identity and the share of candidates with a "
                          f"{fr.NEAR_DUPLICATE_IDENTITY:g}%-or-closer neighbour, from the frozen cell summary; "
                          "the rung value is the mean over the eight domains",
            "rungs": out}


# ------------------------------------------------------------------ precision control


def precision_control(campaign: dict, cells: Cells, domains) -> dict:
    """float16 against bfloat16 for each checkpoint run at both precisions, per arm, on both endpoints."""

    by_base = defaultdict(dict)
    for rung in campaign["rungs"]:
        by_base[campaign["bases"][rung]][campaign["precision_of"][rung]] = rung
    pairs = {base: rungs for base, rungs in by_base.items() if len(rungs) > 1}
    out = {"statement": PRECISION_STATEMENT, "interval_coverage_note": e20.BCA_NOTE, "checkpoints": {}}
    if not pairs:
        out["checkpoints"] = None
        out["withheld_reason"] = ("no checkpoint in this campaign is present at more than one precision, so the "
                                  "precision control cannot be formed")
        return out
    for base, rungs in sorted(pairs.items()):
        if sorted(rungs) != ["bfloat16", "float16"]:
            raise ValueError(f"{base}: precisions {sorted(rungs)}; the declared control is float16 against "
                             "bfloat16 and no other pair is defined")
        test, reference = rungs["float16"], rungs["bfloat16"]
        declared = {arm: (campaign["doses"][(test, arm)], campaign["doses"][(reference, arm)]) for arm in ARMS}
        differing = {arm: [list(a), list(b)] for arm, (a, b) in declared.items() if a != b}
        if differing:
            raise ValueError(f"{base}: the two precisions are not at the same doses ({differing}); precision is "
                             "then not the only thing that moves and the control is void")
        entry = {"float16_rung": test, "bfloat16_rung": reference, "contrast": f"{test} - {reference}",
                 "doses": {arm: list(campaign["doses"][(test, arm)]) for arm in ARMS}, "endpoints": {}}
        getter = _by_rung(cells.cell)
        for endpoint, metric, definition in ENDPOINTS:
            by_arm = {arm: e20.estimate(*e20.model_contrast(domains, arm, test, reference, metric, getter))
                      for arm in ARMS}
            entry["endpoints"][endpoint] = {
                "definition": definition, "by_arm": e20.strip_p(by_arm),
                "achieved_upper_bound": {arm: upper_bound(by_arm[arm]) for arm in ARMS}}
        out["checkpoints"][base] = entry
    return out


def _by_rung(getter):
    """Adapt ``cells.cell`` for ``model_contrast`` when the differenced names are rungs, not arms.

    ``model_contrast(domains, token, a, b, metric, getter)`` calls ``getter(token, domain, name)``. For an arm
    contrast the token is the rung and ``name`` is the arm. The precision control swaps the roles -- the token
    is the arm, held fixed, and the two differenced names are the two rungs -- so this adapter re-orders the
    arguments. Nothing else about the frozen differencing changes, and the reason string it builds then names
    the rung that was short of admissible candidates, which is what a reader needs there.
    """

    return lambda arm, domain, rung: getter(rung, domain, arm)


# ------------------------------------------------------------------ declared context: the dose floor


def dose_floor_context(calibrations, campaign: dict) -> dict:
    """Per rung, arm and domain: the calibration's own floor margin, and which cells fall below what it required.

    A rung without its own calibration -- the float16 control, whose doses were solved in bfloat16 -- is
    carried over from the calibration of the same checkpoint, and only when every declared dose is identical.
    """

    solved = {}
    for path in calibrations:
        document = json.loads(fr.require_asset(path, "--calibration").read_text())
        label = f"{document['target']['model']}-{document['target']['dtype']}"
        if document["status"] != "solved" or document["refusals"]:
            raise ValueError(f"{path}: calibration status {document['status']!r} with refusals "
                             f"{document['refusals']}; a dose that was not solved has no declared context")
        solved[label] = {"path": str(path), "checkpoint": document["target"]["model"],
                         "sha256": sha256_file(Path(path)), "basis": document["basis"],
                         "decision_statistic": document["decision_statistic"], "arms": document["solved"]}
    out = {"statement": DOSE_FLOOR_STATEMENT, "intervention_basis": INTERVENTION_BASIS_NOTE,
           "calibrations": {label: {k: v for k, v in block.items() if k != "arms"}
                            for label, block in solved.items()},
           "rungs": {}, "below_declared_floor_margin": []}
    for rung in campaign["rungs"]:
        source, carried = solved.get(rung), None
        if source is None:
            base = campaign["bases"][rung]
            candidates = sorted(label for label, block in solved.items() if block["checkpoint"] == base)
            if len(candidates) != 1:
                raise ValueError(f"{rung}: {len(candidates)} calibrations for checkpoint {base!r}; pass the "
                                 "calibration whose doses this rung was scheduled at with --calibration")
            carried = candidates[0]
            source = solved[carried]
        for arm in ARMS:
            if arm == "N":
                continue
            declared = campaign["doses"][(rung, arm)]
            block = source["arms"].get(arm)
            if block is None:
                raise ValueError(f"{rung}/{arm}: the calibration declares no solved dose for this arm")
            component = block["component"]
            index = 0 if component == "fields" else 1
            if abs(block["dose"] - declared[index]) > 1e-12 or declared[1 - index] != 0.0:
                raise ValueError(f"{rung}/{arm}: the protocol dose {list(declared)} does not match the "
                                 f"calibrated {component} dose {block['dose']}; the declared context would "
                                 "describe a dose the campaign did not run")
            required, floor = block["floor_margin_required"], block["noise_floor_nats"]
            margins = block["per_domain_floor_margin"]
            entry = {"dose": list(declared), "component": component, "noise_floor_nats": floor,
                     "floor_margin_required": required, "median_floor_margin_achieved": block["floor_margin_achieved"],
                     "per_domain_floor_margin": margins,
                     "per_domain_achieved_kl_per_residue": block["achieved"]["per_domain"],
                     "calibration_source": carried or rung,
                     "carried_over": None if carried is None else {
                         "from": carried,
                         "reason": "this rung has no calibration of its own: its doses were solved for the same "
                                   "checkpoint at the other precision and it runs at those identical doses, so "
                                   "the floor margin -- a property of the family reference and the dose -- is the "
                                   "one solved there. Verified dose-for-dose before being carried over."},
                     "domains_below_required_margin": sorted(d for d, m in margins.items() if m < required)}
            out["rungs"].setdefault(rung, {})[arm] = entry
            for domain in entry["domains_below_required_margin"]:
                out["below_declared_floor_margin"].append(
                    {"rung": rung, "arm": arm, "domain": domain, "floor_margin": margins[domain],
                     "floor_margin_required": required})
    return out


# ------------------------------------------------------------------ driver


def analyse(campaign: dict, family_models: dict, permutation_seed: int, calibrations) -> dict:
    cells = Cells(campaign["records"], campaign["doses"])
    shapes = check_design(campaign, cells)
    domains = shapes["domains"]
    lines, strata_of_line = {}, {}
    for line in campaign["lines"]:
        pooled = f"{POOLED_PREFIX}{line['precision']}" if len(line["rungs"]) > 1 else None
        lines[line["precision"]] = {
            "records": str(line["records_path"]), "protocol": str(line["protocol_path"]),
            "rungs": list(line["rungs"]), "n_records": len(line["records"]),
            "candidates_per_cell": line["candidates_per_cell"],
            "pooled_stratum": pooled,
            "pooled_withheld_reason": None if pooled else
            "one rung on this measure line; a pooled stratum would only duplicate it",
        }
        strata_of_line[line["precision"]] = [*line["rungs"], *([pooled] if pooled else [])]

    bounds, achieved = {}, {}
    for endpoint, metric, definition in ENDPOINTS:
        table = {}
        for line in campaign["lines"]:
            pooled = lines[line["precision"]]["pooled_stratum"]
            table.update(contrast_table(domains, CONTRASTS, line["rungs"], metric, cells.cell, pooled))
        bounds[endpoint] = {"definition": definition, "strata": table}
        if endpoint == "novelty_admissibility_rate":
            bounds[endpoint]["note"] = ADMISSIBILITY_NOTE
        achieved[endpoint] = {name: {stratum: upper_bound(table[stratum][name]) for stratum in table}
                              for name, *_rest in CONTRASTS}

    capture = {}
    for line in campaign["lines"]:
        for rung in line["rungs"]:
            capture[rung] = captured_fraction(campaign["records"], (rung,), campaign["doses"], family_models,
                                              permutation_seed)
        pooled = lines[line["precision"]]["pooled_stratum"]
        if pooled:
            capture[pooled] = captured_fraction(campaign["records"], line["rungs"], campaign["doses"],
                                                family_models, permutation_seed)

    return {
        "schema": SCHEMA, "statement": STATEMENT, "issues_a_verdict": False, "issues_a_pass_fail_flag": False,
        "reference_scale": {"margin_tm": e20.MARGIN, "role": REFERENCE_SCALE},
        "design": {
            "arms": list(ARMS), "arms_absent": {"F": "fields and coupling together; not in this five-arm design, "
                                                     "so E20's F-M contrast cannot be formed here",
                                                "Cd": "cross-domain coupling control; outside this design"},
            "domains": domains, "matches_sealed_eight": domains == sorted(e20.SEALED_DOMAINS),
            "measure_lines": lines, "strata_by_measure_line": strata_of_line,
            "measure_line_rule": "precision is part of the native contract, so a measure line is a precision: "
                                 "the bfloat16 rungs are measured together and the float16 rung on its own. No "
                                 "stratum pools across a measure line.",
            "rungs": shapes["rungs"],
            "contrast_definitions": {name: {"left": left, "right": right, "reads": reads}
                                     for name, left, right, reads in CONTRASTS},
        },
        "discipline": {
            "independence_unit": "domain",
            "minimum_bootstrap_units": MINIMUM_BOOTSTRAP_UNITS,
            "interval": f"{int(e20.CONFIDENCE * 100)}% BCa bootstrap over domains, {e20.N_RESAMPLES} resamples, "
                        f"seed {e20.BOOTSTRAP_SEED}",
            "interval_coverage_note": e20.BCA_NOTE,
            "p_values": "none issued. The p-value fields of the frozen record shape are present and null.",
            "admissibility": f"identity to the wild type strictly below {e20.ADMISSIBLE_IDENTITY:g}%, at least "
                             f"{e20.MIN_ADMISSIBLE} admissible candidates or the cell has no TM value and is "
                             "never imputed",
        },
        "declared_context_dose_floor": dose_floor_context(calibrations, campaign),
        "bounds": bounds, "achieved_upper_bounds": achieved,
        e20.CAPTURE_LABEL: {"by_stratum": capture},
        "pool_diversity": pool_diversity(cells, campaign["rungs"], domains),
        "precision_control": precision_control(campaign, cells, domains),
        "cells": [{"model": m, "domain": d, "arm": a, "lambda_fields": lf, "lambda_coupling": lc, **summary}
                  for (m, d, a, lf, lc), summary in sorted(cells.cells.items(), key=str)],
    }


def inputs_report(campaign: dict, family_models: dict, shapes: dict) -> dict:
    """What a dry run reports: shapes and counts of the validated inputs, and no contrast of any kind."""

    return {
        "schema": INPUTS_SCHEMA,
        "statement": "Dry run. Inputs were read and validated and their shapes counted; no contrast, bound, "
                     "interval or endpoint value was computed.",
        "domains": shapes["domains"], "matches_sealed_eight": shapes["domains"] == sorted(e20.SEALED_DOMAINS),
        "n_records_total": len(campaign["records"]),
        "measure_lines": {line["precision"]: {
            "records": str(line["records_path"]), "protocol": str(line["protocol_path"]),
            "n_records": len(line["records"]), "rungs": list(line["rungs"]),
            "candidates_per_cell": line["candidates_per_cell"]} for line in campaign["lines"]},
        "rungs": shapes["rungs"],
        "columns_by_domain": {
            domain: {"declared_by_protocol": campaign["lines"][0]["domains_declared"][domain]["L"],
                     "family_reference": family_models[domain].length,
                     "sequence_lengths_in_records": sorted(
                         {len(r["sequence"]) for r in campaign["records"] if r["domain"] == domain})}
            for domain in shapes["domains"]},
    }


def render_inputs(report: dict) -> str:
    lines = [report["statement"],
             f"records {report['n_records_total']} over {len(report['measure_lines'])} measure line(s); "
             f"domains {report['domains']} (sealed eight: {report['matches_sealed_eight']})"]
    for precision, line in report["measure_lines"].items():
        lines.append(f"== measure line {precision}: {line['n_records']} records, rungs {line['rungs']} ==")
        lines.append(f"   records  {line['records']}")
        lines.append(f"   protocol {line['protocol']}")
    for rung, shape in report["rungs"].items():
        lines.append(f"[{shape['measure_line']}] {rung} (checkpoint {shape['base_checkpoint']}): "
                     f"{shape['n_records']} records, {shape['n_arms']} arms {shape['arms']}, "
                     f"{shape['n_domains']} domains, {shape['n_cells']} cells, "
                     f"candidates/cell {shape['candidates_per_cell']}")
        lines.append("   doses " + "  ".join(f"{arm}:{dose}" for arm, dose in shape["doses"].items()))
    lines.append("columns by domain (protocol / family reference / sequence lengths seen):")
    for domain, block in report["columns_by_domain"].items():
        lines.append(f"   {domain}: {block['declared_by_protocol']} / {block['family_reference']} / "
                     f"{block['sequence_lengths_in_records']}")
    calibration = report["calibration"]
    lines.append(f"calibration: {len(calibration['files'])} file(s) validated against the declared doses; "
                 f"carried over for {calibration['carried_over_rungs'] or 'no rung'}; "
                 f"{calibration['n_dose_cells_below_required_floor_margin']} dose cells below the required "
                 "floor margin")
    for rung, source in calibration["source_by_rung"].items():
        lines.append(f"   {rung} <- {source}")
    return "\n".join(lines)


def render(result: dict) -> str:
    lines = [result["statement"], "",
             f"domains {result['design']['domains']} (sealed eight: {result['design']['matches_sealed_eight']})",
             f"measure lines {list(result['design']['measure_lines'])}; strata "
             f"{result['design']['strata_by_measure_line']}"]
    for endpoint, block in result["bounds"].items():
        lines.append(f"== ACHIEVED UPPER 95% BCa BOUND: {endpoint} (descriptive; reference scale "
                     f"{result['reference_scale']['margin_tm']:+.2f} TM; no verdict) ==")
        for name in result["achieved_upper_bounds"][endpoint]:
            cells = "  ".join(f"{stratum}: " + ("unestimable" if value is None else f"{value:+.4f}")
                              for stratum, value in result["achieved_upper_bounds"][endpoint][name].items())
            lines.append(f"{name:<6}{cells}")
        for stratum, records in block["strata"].items():
            for name, record in records.items():
                per = " ".join(f"{d}:{v['diff']:+.4f}" for d, v in record["per_domain"].items())
                head = f"[{stratum}] {name:<6} n={record['n_domains']} "
                if record["status"] == "unestimable":
                    lines.append(f"{head}UNESTIMABLE ({record['reason']}) [{per}]")
                    continue
                low, high = record["interval95_bca"]
                lines.append(f"{head}mean {record['mean']:+.4f} CI95 [{low:+.4f}, {high:+.4f}] "
                             f"+{record['favouring']['positive']}/-{record['favouring']['negative']} "
                             f"d_id_wt {record['mean_identity_diff']:+.1f} [{per}]")
    lines.append(result["discipline"]["interval_coverage_note"])
    capture = result[e20.CAPTURE_LABEL]["by_stratum"]
    lines.append(f"== {e20.CAPTURE_LABEL} (k={e20.SELECTION_K}; no threshold, no flag) ==")
    for stratum, block in capture.items():
        for arm, arm_block in block["arms"].items():
            for name, entry in arm_block["selectors"].items():
                interval = ("unestimable" if entry["interval95_bca"] is None
                            else f"CI95 [{entry['interval95_bca'][0]:+.3f}, {entry['interval95_bca'][1]:+.3f}]")
                fraction = ("n/a" if entry["mean_captured_fraction"] is None
                            else f"{entry['mean_captured_fraction']:+.3f}")
                lines.append(f"[{stratum}/{arm}] {name:<18} n={arm_block['n_domains']} captured {fraction} {interval}")
    diversity = result["pool_diversity"]["rungs"]
    lines.append("== pool diversity (within-cell mean pairwise identity / near-duplicate rate) ==")
    for rung, arms in diversity.items():
        lines.append(f"[{rung}] " + "  ".join(
            f"{arm}:{block['mean_pairwise_identity']:.1f}/{block['near_duplicate_rate']:.3f}"
            for arm, block in arms.items()))
    precision = result["precision_control"]
    lines.append("== PRECISION CONTROL (float16 - bfloat16, same doses; no verdict) ==")
    if precision["checkpoints"] is None:
        lines.append(precision["withheld_reason"])
    else:
        for base, entry in precision["checkpoints"].items():
            for endpoint, block in entry["endpoints"].items():
                lines.append(f"[{base}] {endpoint}: " + "  ".join(
                    f"{arm}: " + ("unestimable" if value is None else f"{value:+.4f}")
                    for arm, value in block["achieved_upper_bound"].items()))
    floor = result["declared_context_dose_floor"]
    lines.append("== DECLARED CONTEXT: dose cells below the calibration's required floor margin ==")
    for item in floor["below_declared_floor_margin"]:
        lines.append(f"   {item['rung']}/{item['arm']}/{item['domain']}: floor margin {item['floor_margin']:.2f} "
                     f"against the required {item['floor_margin_required']:.1f}")
    lines.append(floor["statement"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", type=Path, action="append", required=True, metavar="PER_SEQUENCE_JSONL",
                    help="per_sequence.jsonl of one measure line; repeat once per line, paired with --protocol")
    ap.add_argument("--protocol", type=Path, action="append", required=True,
                    help="the measure line's guided_generation_protocol JSON; repeat in the same order as --records")
    ap.add_argument("--references", type=Path, required=True,
                    help="directory of family-reference <DOMAIN>.npz/.json")
    ap.add_argument("--calibration", type=Path, action="append", required=True,
                    help="progen3_dose_calibration.json of a rung; repeat for each calibrated checkpoint")
    ap.add_argument("--permutation-seed", type=int, required=True,
                    help="the declared permuted_coupling seed of the generation protocol")
    ap.add_argument("--out", type=Path, help="output JSON artifact; required unless --dry-run")
    ap.add_argument("--dry-run", action="store_true",
                    help="read and validate the inputs and report shapes and counts only; compute no contrast")
    args = ap.parse_args(argv)
    if args.out is None and not args.dry_run:
        ap.error("--out is required for a full run")
    fr.require_asset(args.references, "--references", directory=True)
    campaign = load_campaign(args.records, args.protocol)
    family_models = {domain: load_family_model(domain, args.references)
                     for domain in sorted({r["domain"] for r in campaign["records"]})}
    if args.dry_run:
        cells = Cells(campaign["records"], campaign["doses"])
        report = inputs_report(campaign, family_models, check_design(campaign, cells))
        # The calibration is validated here too -- every declared dose matched against the protocol's -- so a
        # clean dry run means the full run will not refuse on its inputs. Only counts are reported.
        floor = dose_floor_context(args.calibration, campaign)
        report["calibration"] = {
            "files": sorted(floor["calibrations"]),
            "source_by_rung": {rung: sorted({entry["calibration_source"] for entry in arms.values()})
                               for rung, arms in floor["rungs"].items()},
            "carried_over_rungs": sorted(rung for rung, arms in floor["rungs"].items()
                                         if any(entry["carried_over"] for entry in arms.values())),
            "n_dose_cells_below_required_floor_margin": len(floor["below_declared_floor_margin"]),
        }
        report["inputs"] = {"records": [str(p) for p in args.records], "protocol": [str(p) for p in args.protocol],
                            "references": str(args.references),
                            "calibration": [str(p) for p in args.calibration],
                            "permutation_seed": args.permutation_seed}
        if args.out is not None:
            write_json(args.out, report)
        print(render_inputs(report))
        return 0
    result = analyse(campaign, family_models, args.permutation_seed, args.calibration)
    frozen = Path(e20.__file__).resolve()
    result["provenance"] = {
        "frozen_module": str(frozen.relative_to(Path(__file__).resolve().parents[3])),
        "frozen_module_sha256": sha256_file(frozen),
        "inputs": {"records": [str(p) for p in args.records], "protocol": [str(p) for p in args.protocol],
                   "references": str(args.references), "calibration": [str(p) for p in args.calibration],
                   "permutation_seed": args.permutation_seed},
    }
    write_json(args.out, result)
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
