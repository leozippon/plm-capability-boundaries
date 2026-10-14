#!/usr/bin/env python3
"""Pre-specified confirmatory analysis of E20: does pairwise coupling add design benefit over first-order information?

Written before the confirmatory records exist; the decision rules below are frozen and must not be
weakened, broadened or extended to make a result come out.

Endpoints, per domain per cell: the primary is the mean TM-score over candidates with
identity_to_wild_type < 20 (at least 25 such candidates, otherwise the value is missing and is never
imputed); the co-primary is the feasibility rate, the admissible fraction of the pool. The independence
unit is the domain. A contrast with fewer than MINIMUM_BOOTSTRAP_UNITS domains is "unestimable": no
interval, no p-value. The headline is the achieved
upper 95% bound on C - Pr and F - M, per model and pooled. The verdict flag requires that bound to lie
below the 0.05 TM margin in each model separately, and is only ever "no pairwise-specific design benefit",
"inconclusive at this power" or "unestimable"; it is never "positive" and never "equivalent". The decision
rests on the interval and the margin, not on either p-value.

Selection is structurally blind to structure-derived fields: selectors see a SelectorView holding the
sequences and native_loglik and nothing else, and plddt_mean, ptm and tm_score are refused by name.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse_selection_factorial as sf  # noqa: E402
from src.capability.context.family_potts import load_family_model  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402

SCHEMA = "e20_confirmatory_v1"
SEALED_DOMAINS = ("1E0L", "1F0M", "1IFY", "1V1C", "2K1B", "2LGW", "2LVN", "3DKM")
MODELS = ("progen2-medium", "protgpt2")
POOLED = "pooled"
STRATA = (*MODELS, POOLED)

ADMISSIBLE_IDENTITY = 20.0  # percent identity to the wild type; admissible means strictly below
MIN_ADMISSIBLE = 25
MARGIN = 0.05
SELECTION_K = 10
SELECTION_POOL_ARM = "N"  # selection re-ranks the unguided pool, so it is the only intervention in play
CONFIDENCE = 0.95
N_RESAMPLES = 9999
BOOTSTRAP_SEED = 20261013
LEAVE_OUT_DOMAIN = "1IFY"
EPS = 1e-12

VERDICT_CONCLUDE = "no pairwise-specific design benefit"
VERDICT_INCONCLUSIVE = "inconclusive at this power"
VERDICT_UNESTIMABLE = "unestimable"
EQUIVALENCE_CONTRASTS = ("C-Pr", "F-M")

#: Frozen doses (lambda_fields, lambda_coupling). C, Cx and Cd share one coupling dose per model.
_C = {"progen2-medium": (0.0, 4.0), "protgpt2": (0.0, 8.0)}
FROZEN_DOSES = {
    "N": {m: (0.0, 0.0) for m in MODELS},
    "Pr": {m: (0.25, 0.0) for m in MODELS},
    "M": {m: (0.25, 0.0) for m in MODELS},
    "C": _C, "Cx": _C, "Cd": _C,
    "F": {"progen2-medium": (0.25, 1.0), "protgpt2": (0.5, 0.25)},
}
CELL_CONTRASTS = (("C-Pr", "C", "Pr"), ("F-M", "F", "M"), ("C-Cx", "C", "Cx"), ("C-Cd", "C", "Cd"),
                  ("C-N", "C", "N"), ("Pr-N", "Pr", "N"))
SELECTION_CONTRASTS = (("potts_full-potts_fields", "potts_full", "potts_fields"),
                       ("potts_coupling-coupling_permuted", "potts_coupling", "coupling_permuted"))
REFERENCE_CONTRASTS = (("P-N", "P", "N"), ("H-N", "H", "N"))
FRONTIER_ARMS = ("Pr", "C")

#: Structure-derived fields that may never rank or choose a candidate. Never merged into one score either.
FORBIDDEN_SELECTOR_FIELDS = frozenset({"plddt_mean", "ptm", "tm_score"})
SELECTOR_NAMES = ("native_loglik", "potts_fields", "potts_coupling", "potts_full", "coupling_permuted")

#: Reference arms are keyed on the model label their records carry; a record under any other label is refused.
REFERENCE_MODEL_LABELS = {"P": "potts-joint", "H": "natural"}
BCA_NOTE = ("BCa over 8 domains: realised coverage of a nominal 95% percentile interval is about 0.89 at eight "
            "units (src.capability.core.statistics); the interval is anti-conservative, so read the bound as "
            "optimistic, not exact.")
SIGN_FLIP_NOTE = ("exact two-sided sign-flip p cannot fall below 2/2**n (0.0078 at n=8); after Holm over 8 "
                  "contrasts the smallest attainable adjusted p is 0.0625, so no contrast in this family can "
                  "reach family-wise significance at 0.05 by this test. The decision rests on the interval "
                  "and the margin, not on either p-value.")

STATEMENT = (
    "Pre-specified analysis; decision rules frozen before the data existed. The independence unit is the "
    "domain. Contrasts with fewer than MINIMUM_BOOTSTRAP_UNITS domains carry no interval and no p-value. "
    "The headline is the achieved upper bound on C-Pr and F-M; the margin flag requires it below 0.05 TM "
    "in each model separately. Neither p-value decides anything."
)


# ------------------------------------------------------------------ selector registry


@dataclass(frozen=True)
class SelectorView:
    """All a selector may read: the sequences, their domain and native_loglik."""

    domain: str
    sequences: tuple[str, ...]
    native_loglik: np.ndarray


def selector_view(rows) -> SelectorView:
    """Project records onto the selector-visible fields; nothing structural survives the projection."""

    return SelectorView(rows[0]["domain"], tuple(r["sequence"] for r in rows),
                        np.array([r["native_loglik"] for r in rows], dtype=np.float64))


def selector_scores(names, view: SelectorView, model, permutation_seed: int) -> dict[str, np.ndarray]:
    """Scores of the requested selectors, computed by the shared Potts machinery of the selection analysis."""

    for name in names:
        if name in FORBIDDEN_SELECTOR_FIELDS:
            raise ValueError(f"{name!r} is a structure-derived field and cannot be used as a selector")
        if name not in SELECTOR_NAMES:
            raise ValueError(f"unknown selector {name!r}; the registry holds {SELECTOR_NAMES}")
    codes = sf.encode_pool(list(view.sequences), model.length, view.domain)
    scores = sf.selector_scores(model, None, permutation_seed, codes)
    scores["native_loglik"] = view.native_loglik
    return {name: scores[name] for name in names}


# ------------------------------------------------------------------ cell summaries


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if len(values) else None


def summarise_cell(rows) -> dict:
    """Every per-cell number: endpoints with their identity companions, controls and guidance strength."""

    ident = np.array([r["identity_to_wild_type"] for r in rows], dtype=np.float64)
    msa = np.array([r["max_msa_identity"] for r in rows], dtype=np.float64)
    tm = np.array([r["tm_score"] for r in rows], dtype=np.float64)
    admissible = ident < ADMISSIBLE_IDENTITY
    n_adm = int(admissible.sum())
    strengths = [r["guidance_strength"]["kl_per_residue"] for r in rows if r.get("guidance_strength")]
    if strengths and len(strengths) != len(rows):
        raise ValueError(f"{rows[0]['model']}/{rows[0]['domain']}/{rows[0]['arm']}: only some records carry guidance_strength")
    return {
        "n": len(rows), "n_admissible": n_adm, "feasibility": n_adm / len(rows),
        "tm": float(tm[admissible].mean()) if n_adm >= MIN_ADMISSIBLE else None,
        "identity_admissible": _mean(ident[admissible]), "msa_identity_admissible": _mean(msa[admissible]),
        "identity_pool": float(ident.mean()), "msa_identity_pool": float(msa.mean()),
        "tm_all_candidates": float(tm.mean()),
        "composition_tv_to_wild_type": _mean([r["composition_tv_to_wild_type"] for r in rows]),
        "composition_tv_to_family": _mean([r["composition_tv_to_family"] for r in rows]),
        **(fr.diversity([r["sequence"] for r in rows]) if len(rows) > 1 else {}),
        "kl_per_residue": _mean(strengths) if strengths else None,
    }


def group_cells(records) -> dict[tuple, list]:
    cells = defaultdict(list)
    for r in records:
        cells[(r["model"], r["domain"], r["arm"], r["lambda_fields"], r["lambda_coupling"])].append(r)
    return cells


def selected_summary(admissible_rows, scores, name: str) -> dict:
    """Mean TM and identities of the top-k candidates under one selector, drawn from the admissible pool."""

    top = np.argsort(-scores, kind="stable")[:SELECTION_K]
    picked = [admissible_rows[i] for i in top]
    return {"tm": float(np.mean([r["tm_score"] for r in picked])), "n_admissible": len(admissible_rows),
            "identity_admissible": float(np.mean([r["identity_to_wild_type"] for r in picked])),
            "msa_identity_admissible": float(np.mean([r["max_msa_identity"] for r in picked])), "selector": name}


# ------------------------------------------------------------------ statistics


def holm_adjust(p_values, family_size: int | None = None) -> list[float | None]:
    """Holm step-down adjusted p-values for a family of fixed size; None entries (unestimable) count as p = 1.

    The family size is the number of contrasts in the frozen family, not the number that happened to be
    estimable, so the correction never depends on the data.
    """

    m = family_size if family_size is not None else len(p_values)
    if m < len(p_values):
        raise ValueError("family size smaller than the number of p-values")
    filled = [1.0 if p is None else float(p) for p in p_values]
    order = sorted(range(len(filled)), key=lambda i: filled[i])
    adjusted, running = [None] * len(filled), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * filled[i]))
        adjusted[i] = running
    return [None if p_values[i] is None else adjusted[i] for i in range(len(filled))]


def bca_interval(values) -> tuple[list[float], bool]:
    """95% BCa bootstrap interval of the mean over domains; a constant sample is returned as a point, flagged."""

    v = np.asarray(values, dtype=np.float64)
    if float(np.ptp(v)) == 0.0:
        return [float(v[0]), float(v[0])], True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = stats.bootstrap((v,), np.mean, method="BCa", confidence_level=CONFIDENCE,
                                 n_resamples=N_RESAMPLES, random_state=np.random.default_rng(BOOTSTRAP_SEED))
    low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise RuntimeError("BCa interval is non-finite")
    return [low, high], False


def sign_flip_p(diffs) -> float:
    """Exact two-sided sign-flip p-value of the mean difference: all 2**n sign patterns enumerated."""

    d = np.asarray(diffs, dtype=np.float64)
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * len(d))).reshape(len(d), -1).T
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - EPS))


def estimate(per_domain: dict, missing: dict, *, descriptive_floor: int | None = None, label: str = "") -> dict:
    """Summarise one contrast over domains.

    ``per_domain`` maps domain -> {"diff", "identity_diff", "msa_identity_diff"}. Below the unit floor no
    interval and no p-value are produced; ``descriptive_floor`` lowers the floor for the one pre-declared
    descriptive sensitivity and then withholds the p-value as well.
    """

    diffs = np.array([v["diff"] for v in per_domain.values()], dtype=np.float64)
    n = len(diffs)
    floor_units = MINIMUM_BOOTSTRAP_UNITS if descriptive_floor is None else descriptive_floor
    floor = bootstrap_unit_floor(n, minimum_units=floor_units)
    record = {
        "n_domains": n, "per_domain": per_domain, "missing_domains": missing,
        "mean": float(diffs.mean()) if n else None,
        "mean_identity_diff": _mean([v["identity_diff"] for v in per_domain.values()]),
        "mean_msa_identity_diff": _mean([v["msa_identity_diff"] for v in per_domain.values()]),
        "favouring": {"positive": int((diffs > 0).sum()), "negative": int((diffs < 0).sum()),
                      "tied": int((diffs == 0).sum())},
        "interval95_bca": None, "interval_zero_variance": None, "interval_coverage_note": None,
        "p_holm_ttest": None, "p_holm_signflip": None, "_p": None, "_ps": None,
    }
    if floor["degenerate"]:
        reason = f"{n} domains have a value, below the {floor_units}-domain floor"
        if missing:
            reason += "; without a value: " + "; ".join(f"{d} ({why})" for d, why in missing.items())
        record.update(status="unestimable", reason=reason)
        return record
    record["interval95_bca"], record["interval_zero_variance"] = bca_interval(diffs)
    record["interval_coverage_note"] = BCA_NOTE
    if descriptive_floor is None:
        sd = float(diffs.std(ddof=1))
        record["_p"] = (float(stats.ttest_1samp(diffs, 0.0).pvalue) if sd > 0.0
                        else (1.0 if float(diffs.mean()) == 0.0 else 0.0))
        record["_ps"] = sign_flip_p(diffs)
    record.update(status="descriptive" if descriptive_floor is not None else "estimated", reason=label or None)
    return record


def finish_family(records: dict, family_size: int) -> dict:
    """Holm-adjust the frozen family and drop the uncorrected p-values so only the adjusted ones are reported."""

    names = list(records)
    for key, out in (("_p", "p_holm_ttest"), ("_ps", "p_holm_signflip")):
        for name, p in zip(names, holm_adjust([records[n][key] for n in names], family_size)):
            records[name][out] = p
    return strip_p(records)


def strip_p(records: dict) -> dict:
    for rec in records.values():
        rec.pop("_p")
        rec.pop("_ps")
    return records


# ------------------------------------------------------------------ contrasts


class Sources:
    """Per (model, domain, name) summaries for cell arms and for selectors, with the reason when one is absent."""

    def __init__(self, records, family_models, permutation_seed):
        self.cells = {k: summarise_cell(v) for k, v in group_cells(records).items()}
        for (model, domain, arm, *_dose) in self.cells:
            if arm in REFERENCE_MODEL_LABELS and model != REFERENCE_MODEL_LABELS[arm]:
                raise ValueError(f"{domain}: {arm} records carry model {model!r}; the reference arm must be "
                                 f"labelled {REFERENCE_MODEL_LABELS[arm]!r}, and there is no fallback")
        self.arm_cells = defaultdict(list)
        for (model, domain, arm, *_dose), summary in self.cells.items():
            self.arm_cells[(model, domain, arm)].append(summary)
        by_arm = defaultdict(list)
        for r in records:
            by_arm[(r["model"], r["domain"], r["arm"])].append(r)
        names = sorted({n for _, x, y in SELECTION_CONTRASTS for n in (x, y)})
        self._selected = {}
        for (model, domain, arm), rows in by_arm.items():
            if arm != SELECTION_POOL_ARM or domain not in family_models:
                continue
            admissible = [r for r in rows if r["identity_to_wild_type"] < ADMISSIBLE_IDENTITY]
            if len(admissible) < MIN_ADMISSIBLE:
                for name in names:
                    self._selected[(model, domain, name)] = {"tm": None, "n_admissible": len(admissible)}
                continue
            scores = selector_scores(names, selector_view(admissible), family_models[domain], permutation_seed)
            for name in names:
                self._selected[(model, domain, name)] = selected_summary(admissible, scores[name], name)

    def cell(self, model, domain, arm):
        """The frozen-dose cell summary, or (None, reason). P and H are keyed on their model label, not a dose."""

        if arm in REFERENCE_MODEL_LABELS:
            found = self.arm_cells.get((REFERENCE_MODEL_LABELS[arm], domain, arm), [])
            if len(found) > 1:
                raise ValueError(f"{domain}/{arm}: {len(found)} cells under model {REFERENCE_MODEL_LABELS[arm]!r}")
            return (found[0], None) if found else (None, f"no {arm} cell under model {REFERENCE_MODEL_LABELS[arm]!r}")
        key = (model, domain, arm, *FROZEN_DOSES[arm][model])
        if key in self.cells:
            return self.cells[key], None
        return None, f"no {arm} cell at dose {FROZEN_DOSES[arm][model]}"

    def selected(self, model, domain, name):
        if (model, domain, name) in self._selected:
            return self._selected[(model, domain, name)], None
        return None, f"no {SELECTION_POOL_ARM} pool or family model for selection"


def side_value(summary, reason, name, metric):
    if summary is None:
        return None, reason
    if summary[metric] is None:
        return None, f"{name}: {summary['n_admissible']} admissible < {MIN_ADMISSIBLE}"
    return summary, None


def model_contrast(domains, model, a, b, metric, getter) -> tuple[dict, dict]:
    """Per-domain differences a - b for one model; ``getter(model, domain, name)`` returns (summary, reason)."""

    ident, msa = (("identity_admissible", "msa_identity_admissible") if metric == "tm"
                  else ("identity_pool", "msa_identity_pool"))
    per, missing = {}, {}
    for d in domains:
        sa, ra = side_value(*getter(model, d, a), a, metric)
        sb, rb = side_value(*getter(model, d, b), b, metric)
        if sa is None or sb is None:
            missing[d] = ra or rb
            continue
        per[d] = {"diff": sa[metric] - sb[metric], "identity_diff": sa[ident] - sb[ident],
                  "msa_identity_diff": sa[msa] - sb[msa]}
    return per, missing


def pooled_contrast(by_model: dict) -> tuple[dict, dict]:
    """Per-domain mean over models; a domain missing in either model is missing, never imputed."""

    domains = set().union(*(set(p) | set(m) for p, m in by_model.values()))
    per, missing = {}, {}
    for d in sorted(domains):
        parts = [by_model[m][0].get(d) for m in MODELS]
        if any(p is None for p in parts):
            missing[d] = "; ".join(f"{m}: {by_model[m][1][d]}" for m in MODELS if d in by_model[m][1])
            continue
        per[d] = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
    return per, missing


def contrast_table(domains, spec, metric, getter, **kw) -> dict:
    """{stratum: {contrast: estimate}} for contrasts ``spec = ((name, a, b), ...)``."""

    out = {s: {} for s in STRATA}
    for name, a, b in spec:
        by_model = {m: model_contrast(domains, m, a, b, metric, getter) for m in MODELS}
        for m in MODELS:
            out[m][name] = estimate(*by_model[m], **kw)
        out[POOLED][name] = estimate(*pooled_contrast(by_model), **kw)
    return out


def equivalence(primary: dict) -> dict:
    """Headline: achieved upper 95% bounds on C-Pr and F-M. Flag: both models' bounds strictly below +0.05.

    The flag requires each model separately; the pooled stratum is supporting. Unestimable in any model
    gives "unestimable"; otherwise any bound at or above the margin gives "inconclusive at this power".
    """

    out = {"margin_tm": MARGIN, "rule": "conclude only if the upper 95% BCa bound of both C-Pr and F-M lies below "
           "+margin in progen2-medium and in protgpt2 separately; unestimable if any of those four is "
           "unestimable; otherwise inconclusive. The pooled stratum is supporting information only.",
           "interval_coverage_note": BCA_NOTE, "strata": {}}
    for stratum in STRATA:
        recs = {n: primary[stratum][n] for n in EQUIVALENCE_CONTRASTS}
        upper = {n: (r["interval95_bca"][1] if r["status"] == "estimated" else None) for n, r in recs.items()}
        out["strata"][stratum] = {
            "mean": {n: r["mean"] for n, r in recs.items()}, "upper_bound": upper,
            "below_margin": {n: (None if u is None else bool(u < MARGIN - EPS)) for n, u in upper.items()}}
    out["achieved_upper_bound"] = {n: {s: out["strata"][s]["upper_bound"][n] for s in STRATA}
                                   for n in EQUIVALENCE_CONTRASTS}
    bounds = [out["strata"][m]["upper_bound"][n] for m in MODELS for n in EQUIVALENCE_CONTRASTS]
    if any(u is None for u in bounds):
        verdict = VERDICT_UNESTIMABLE
    elif all(u < MARGIN - EPS for u in bounds):
        verdict = VERDICT_CONCLUDE
    else:
        verdict = VERDICT_INCONCLUSIVE
    out["verdict"] = verdict
    out["verdict_is_secondary_flag"] = True
    return out


# ------------------------------------------------------------------ secondary tables


def dose_axis(sources: Sources, arms) -> list[dict]:
    """Per stratum, arm and dose: mean over (model, domain) cells of the endpoints and their identity companions."""

    groups = defaultdict(list)
    for (model, domain, arm, lf, lc), s in sources.cells.items():
        if arms is None or arm in arms:
            for stratum in (model, POOLED) if model in MODELS else (model,):
                groups[(stratum, arm, lf, lc)].append(s)
    rows = []
    for (stratum, arm, lf, lc), cells in sorted(groups.items(), key=str):
        with_tm = [c for c in cells if c["tm"] is not None]
        rows.append({
            "stratum": stratum, "arm": arm, "lambda_fields": lf, "lambda_coupling": lc, "n_cells": len(cells),
            "feasibility": _mean([c["feasibility"] for c in cells]),
            "tm_admissible": _mean([c["tm"] for c in with_tm]), "n_cells_with_tm": len(with_tm),
            "identity_admissible": _mean([c["identity_admissible"] for c in with_tm]),
            "tm_all_candidates": _mean([c["tm_all_candidates"] for c in cells]),
            "identity_pool": _mean([c["identity_pool"] for c in cells]),
            "kl_per_residue": _mean([c["kl_per_residue"] for c in cells if c["kl_per_residue"] is not None]),
        })
    return rows


# ------------------------------------------------------------------ driver


def analyse(records, family_models, permutation_seed: int) -> dict:
    domains = sorted({r["domain"] for r in records})
    sources = Sources(records, family_models, permutation_seed)
    cell_getter = sources.cell
    selection_getter = sources.selected

    def tables(metric, spec, getter, **kw):
        return contrast_table(domains, spec, metric, getter, **kw)

    def primary_family(metric, spec, selection_spec=()):
        table = tables(metric, spec, cell_getter)
        if selection_spec:
            for stratum, recs in tables(metric, selection_spec, selection_getter).items():
                table[stratum].update(recs)
        size = len(spec) + len(selection_spec)
        return {s: {"holm_family_size": size, "contrasts": finish_family(recs, size)} for s, recs in table.items()}

    primary = primary_family("tm", CELL_CONTRASTS, SELECTION_CONTRASTS)
    verdict = equivalence({s: p["contrasts"] for s, p in primary.items()})
    feasibility = primary_family("feasibility", CELL_CONTRASTS)
    reference = {s: strip_p(r) for s, r in tables("tm", REFERENCE_CONTRASTS, cell_getter).items()}

    leave_out = [d for d in domains if d != LEAVE_OUT_DOMAIN]
    sensitivity = None
    if LEAVE_OUT_DOMAIN in domains:
        kw = dict(descriptive_floor=MINIMUM_BOOTSTRAP_UNITS - 1, label=f"descriptive only: {len(leave_out)} domains, "
                  f"below the {MINIMUM_BOOTSTRAP_UNITS}-domain floor; no p-value")
        both = {**contrast_table(leave_out, CELL_CONTRASTS, "tm", cell_getter, **kw)}
        for stratum, recs in contrast_table(leave_out, SELECTION_CONTRASTS, "tm", selection_getter, **kw).items():
            both[stratum].update(recs)
        sensitivity = {s: strip_p(r) for s, r in both.items()}

    return {
        "schema": SCHEMA, "statement": STATEMENT,
        "domains": domains, "matches_sealed_eight": domains == sorted(SEALED_DOMAINS),
        "minimum_bootstrap_units": MINIMUM_BOOTSTRAP_UNITS,
        "frozen": {"admissible_identity_below": ADMISSIBLE_IDENTITY, "minimum_admissible": MIN_ADMISSIBLE,
                   "margin_tm": MARGIN, "selection_k": SELECTION_K, "selection_pool_arm": SELECTION_POOL_ARM,
                   "interval": f"{int(CONFIDENCE * 100)}% BCa bootstrap over domains, {N_RESAMPLES} resamples, "
                               f"seed {BOOTSTRAP_SEED}",
                   "p_value": "two-sided paired t-test and exact sign-flip test over domains, each Holm-adjusted within "
                              "the frozen family; unestimable contrasts count as p = 1 so the family size is "
                              "fixed. " + SIGN_FLIP_NOTE,
                   "interval_coverage_note": BCA_NOTE,
                   "doses": {a: {m: list(d) for m, d in v.items()} for a, v in FROZEN_DOSES.items()},
                   "permutation_seed": permutation_seed},
        "primary_tm": primary, "equivalence": verdict,
        "coprimary_feasibility": feasibility,
        "reference_arms_descriptive": reference,
        "sensitivity_leave_out_1IFY_descriptive": sensitivity,
        "frontier": dose_axis(sources, FRONTIER_ARMS),
        "dose_response_recovery_confounded": {
            "label": "recovery-confounded: mean TM over all candidates, admissibility not applied",
            "rows": dose_axis(sources, None)},
        "cells": [{"model": m, "domain": d, "arm": a, "lambda_fields": lf, "lambda_coupling": lc, **s}
                  for (m, d, a, lf, lc), s in sorted(sources.cells.items(), key=str)],
    }


def render(result: dict) -> str:
    lines = [f"domains {result['domains']} (sealed eight: {result['matches_sealed_eight']})"]
    blocks = [("PRIMARY TM (Holm family)", result["primary_tm"], True),
              ("CO-PRIMARY FEASIBILITY (Holm family)", result["coprimary_feasibility"], True),
              ("REFERENCE ARMS (descriptive, no Holm)", {s: {"contrasts": c} for s, c in result["reference_arms_descriptive"].items()}, False)]
    if result["sensitivity_leave_out_1IFY_descriptive"]:
        blocks.append(("LEAVE-1IFY-OUT (descriptive only)",
                       {s: {"contrasts": c} for s, c in result["sensitivity_leave_out_1IFY_descriptive"].items()}, False))
    for title, family, holm in blocks:
        lines.append(f"== {title} ==")
        for stratum, block in family.items():
            for name, c in block["contrasts"].items():
                per = " ".join(f"{d}:{v['diff']:+.4f}" for d, v in c["per_domain"].items())
                head = f"[{stratum}] {name:<34} n={c['n_domains']} "
                if c["status"] == "unestimable":
                    lines.append(f"{head}UNESTIMABLE ({c['reason']}) per-domain [{per}] mean {c['mean']}")
                    continue
                lo, hi = c["interval95_bca"]
                p = (f" p_holm t={c['p_holm_ttest']:.4f} signflip={c['p_holm_signflip']:.4f}" if holm else "")
                lines.append(f"{head}mean {c['mean']:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}] +{c['favouring']['positive']}/"
                             f"-{c['favouring']['negative']} d_id_wt {c['mean_identity_diff']:+.1f}{p} [{per}]")
    eq = result["equivalence"]
    lines.append(f"== HEADLINE: achieved upper 95% BCa bound on mean TM difference ({eq['interval_coverage_note']}) ==")
    for name, by_stratum in eq["achieved_upper_bound"].items():
        cells = "  ".join(f"{s}: " + ("unestimable" if u is None else f"{u:+.4f}") for s, u in by_stratum.items())
        lines.append(f"{name:<6}{cells}   (margin {eq['margin_tm']:+.2f})")
    lines.append(f"margin flag (secondary; both models required): {eq['verdict']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", type=Path, required=True, help="per_sequence.jsonl of the confirmatory cells")
    ap.add_argument("--references", type=Path, required=True, help="directory of family-reference <DOMAIN>.npz/.json")
    ap.add_argument("--out", type=Path, required=True, help="output JSON artefact")
    ap.add_argument("--permutation-seed", type=int, required=True,
                    help="the declared permuted_coupling seed of the generation protocol")
    args = ap.parse_args(argv)
    fr.require_asset(args.records, "--records")
    fr.require_asset(args.references, "--references", directory=True)
    records = sf.load_records(args.records)
    if not records:
        raise ValueError(f"--records {args.records} holds no records")
    family_models = {d: load_family_model(d, args.references) for d in sorted({r["domain"] for r in records})}
    result = analyse(records, family_models, args.permutation_seed)
    result["inputs"] = {"records": str(args.records), "references": str(args.references)}
    write_json(args.out, result)
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
