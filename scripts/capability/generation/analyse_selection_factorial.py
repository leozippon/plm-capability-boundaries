#!/usr/bin/env python3
"""Descriptive selection analysis: does the pairwise part of a family model help pick folded designs?

Every candidate pool (one per model, domain and generator cell) is re-ranked by
selectors computed from the sequence and the family model alone; the TM-score to
the wild-type reference is read only afterwards, to score the selected sets. The
candidate budget is the whole pool in every case, identical across selectors,
and the pool mean is the expectation of random selection at any k.

Descriptive only. Four domains are below MINIMUM_BOOTSTRAP_UNITS, so no p-value,
confidence interval or bootstrap interval is emitted; the independence unit is
the domain and the plain mean across domains is the only aggregate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.context.family_potts import encode, load_family_model  # noqa: E402
from src.capability.context.potts_gauge import to_zero_sum  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.decoding.guided_decoding import permutation_for, select_cross_domain_donor  # noqa: E402

SCHEMA = "selection_factorial_v2"
KS = (1, 5, 10, 25)
#: Potts scores are computed in the zero-sum gauge; the two ``*_raw`` selectors repeat the
#: component scores on the stored parameters so the gauge sensitivity of the contrast is visible.
GAUGE = "zero_sum"
#: Candidates below this percent identity to the wild type are the novelty-restricted stratum; a cell with
#: fewer than ``MIN_ADMISSIBLE`` of them has no restricted value (the confirmatory analysis uses the same floor).
NOVELTY_IDENTITY = 20.0
MIN_ADMISSIBLE = 25
#: Ranks by identity_to_wild_type, highest first. A sequence-level quantity, but it needs the target sequence:
#: a diagnostic upper reference for "how much of a gain is just preferring near-native sequences", not a method.
ORACLES = ("identity_oracle",)
ORACLE_STATEMENT = (
    "identity_oracle ranks candidates by identity_to_wild_type, highest first. It uses knowledge of the target "
    "sequence that no design method has: a diagnostic upper reference for how much of a selector's gain is "
    "explained by preferring near-native sequences. It is not a usable or proposed selector.")
STRATA_STATEMENT = (
    "Every selector is also scored in a novelty-restricted stratum: candidates with identity_to_wild_type < 20 only, "
    "selection and pool mean both taken within them, reported beside the unrestricted numbers. A cell with fewer "
    "than 25 such candidates has no restricted value and is left out of the restricted summary (counted, not "
    "imputed). The shrinkage is 1 - restricted gain / unrestricted gain on the cross-domain means, where gain is "
    "selected-set TM minus pool-mean TM. shrinkage_matched compares against the unrestricted gain on exactly the "
    "cells that have a restricted value; shrinkage_vs_all_cells compares against the unrestricted summary over all "
    "cells and so also reflects which cells dropped out. A shrinkage ratio is unstable where its unrestricted gain "
    "is near zero (values such as -0.31 and -17.22 occur for coupling_permuted); the field "
    "shrinkage_*_unreliable is true when the unrestricted gain magnitude is below 0.01 TM, and then the gains "
    "themselves, or shrinkage_matched, should be read instead.")
UNRELIABLE_BELOW_GAIN = 0.01
SELECTORS = ("native_loglik", "profile", "identity_oracle", "potts_fields", "potts_coupling", "potts_full",
             "coupling_permuted", "coupling_cross_domain", "potts_fields_raw", "potts_coupling_raw")
#: (name, minuend, subtrahend); each is a difference of selected-set mean TM-scores.
CONTRASTS = (
    ("full_minus_fields", "potts_full", "potts_fields"),
    ("coupling_minus_permuted", "potts_coupling", "coupling_permuted"),
    ("coupling_minus_cross_domain", "potts_coupling", "coupling_cross_domain"),
    ("full_minus_fields_raw_gauge", "potts_full", "potts_fields_raw"),
    ("coupling_raw_minus_permuted", "potts_coupling_raw", "coupling_permuted"),
    ("full_minus_profile", "potts_full", "profile"),
    ("full_minus_native_loglik", "potts_full", "native_loglik"),
)
METRICS = ("tm_score", "identity_to_wild_type", "max_msa_identity")
STATEMENT = (
    "Descriptive selection only. Four development domains are below MINIMUM_BOOTSTRAP_UNITS = 8, "
    "so no p-value, confidence interval or bootstrap interval is reported. The independence unit is "
    "the domain, never the sequence; values are per-domain with their plain mean. Candidate budget "
    "is the whole pool in every cell, identical across selectors; the pool mean is the expectation "
    "of random selection at any k."
)


def pair_matrix(J: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """``J[i, j, x_i, x_j]`` for every sequence: (N, L, L)."""
    idx = np.arange(codes.shape[1])
    return J[idx[None, :, None], idx[None, None, :], codes[:, :, None], codes[:, None, :]]


def field_sum(h: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return h[np.arange(codes.shape[1])[None, :], codes].sum(axis=1)


def coupling_sum(J: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return np.triu(pair_matrix(J, codes), k=1).sum(axis=(1, 2))


def encode_pool(sequences: list[str], length: int, domain: str) -> np.ndarray:
    for s in sequences:
        if len(s) != length:
            raise ValueError(f"{domain}: sequence of length {len(s)} against a {length}-column model")
    return np.stack([encode(s) for s in sequences])  # raises on a non-standard residue


def selector_scores(model, donor, permutation_seed: int, codes: np.ndarray) -> dict[str, np.ndarray]:
    """All model-derived selectors for one domain; ``native_loglik`` comes from the record.

    Potts scores use the zero-sum gauge (``to_zero_sum``); the full score is gauge-invariant up
    to a constant and is checked against the raw parameters, the component scores are not.
    """
    L = model.length
    permutation = permutation_for(model.domain, L, permutation_seed)
    h, J = to_zero_sum(model.h, model.J)
    fields, coupling = field_sum(h, codes), coupling_sum(J, codes)
    full = fields + coupling
    raw_fields, raw_coupling = field_sum(model.h, codes), coupling_sum(model.J, codes)
    shift = full - (raw_fields + raw_coupling)
    if np.ptp(shift) > 1e-6 or not np.array_equal(np.argsort(-full, kind="stable"),
                                                   np.argsort(-(raw_fields + raw_coupling), kind="stable")):
        raise ValueError(f"{model.domain}: potts_full differs between gauges by more than a constant "
                         f"(spread {np.ptp(shift):.3g}); the transform or the scoring is wrong")
    scores = {
        "profile": field_sum(model.log_profile, codes),
        "potts_fields": fields,
        "potts_coupling": coupling,
        "potts_full": full,
        "coupling_permuted": coupling_sum(J[np.ix_(permutation, permutation)], codes),
        "potts_fields_raw": raw_fields,
        "potts_coupling_raw": raw_coupling,
    }
    if donor is not None:
        scores["coupling_cross_domain"] = coupling_sum(to_zero_sum(donor.h, donor.J)[1][:L, :L], codes)
    return scores


def topk_means(score: np.ndarray, values: dict[str, np.ndarray], k: int) -> dict[str, float]:
    order = np.argsort(-score, kind="stable")[:k]
    return {m: float(values[m][order].mean()) for m in METRICS}


def load_records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyse(records, models, permutation_seed):
    lengths = {d: m.length for d, m in models.items() if m.qualified}
    donors = {}
    for domain in models:
        try:
            donors[domain] = models[select_cross_domain_donor(domain, lengths)]
        except ValueError as error:  # the pre-registered refusal: no admissible donor
            donors[domain] = None
            print(f"note: {error}", file=sys.stderr)
    by_cell = defaultdict(list)
    for r in records:
        by_cell[(r["model"], r["domain"], r["arm"], r["lambda_fields"], r["lambda_coupling"])].append(r)
    sizes = {len(v) for v in by_cell.values()}
    if len(sizes) != 1 or max(KS) > min(sizes):
        raise ValueError(f"pools must share one size of at least {max(KS)}, found {sorted(sizes)}")
    cells = []
    for (model, domain, arm, lf, lc), rows in sorted(by_cell.items()):
        codes = encode_pool([r["sequence"] for r in rows], models[domain].length, domain)
        scores = selector_scores(models[domain], donors[domain], permutation_seed, codes)
        scores["native_loglik"] = np.array([r["native_loglik"] for r in rows], dtype=np.float64)
        values = {m: np.array([r[m] for r in rows], dtype=np.float64) for m in METRICS}
        scores["identity_oracle"] = values["identity_to_wild_type"]
        cell = {"model": model, "domain": domain, "arm": arm, "lambda_fields": lf, "lambda_coupling": lc,
                "pool_size": len(rows), "pool_mean": {m: float(values[m].mean()) for m in METRICS},
                "selectors": {}}
        for name in SELECTORS:
            if name in scores:
                cell["selectors"][name] = {str(k): topk_means(scores[name], values, k) for k in KS}
        keep = values["identity_to_wild_type"] < NOVELTY_IDENTITY
        cell["n_admissible"] = int(keep.sum())
        cell["restricted"] = None if cell["n_admissible"] < MIN_ADMISSIBLE else {
            "pool_mean": {m: float(values[m][keep].mean()) for m in METRICS},
            "selectors": {name: {str(k): topk_means(scores[name][keep], {m: values[m][keep] for m in METRICS}, k)
                                 for k in KS} for name in SELECTORS if name in scores}}
        cells.append(cell)
    return cells, {d: (None if donors[d] is None else donors[d].domain) for d in donors}


def summarise(cells, stratum, restricted=False):
    """Per-domain means over the stratum's generator cells, then the plain mean over domains.

    ``selected`` holds the selected-set means of each metric; ``gain`` is the TM-score
    advantage over the pool mean; ``identity_shift`` is the selected-minus-pool change in
    each identity, so a gain bought with near-native sequences is visible beside it.
    """
    total = len(cells)
    if restricted:  # a cell without enough admissible candidates has no restricted value
        cells = [{**c, **c["restricted"]} for c in cells if c["restricted"] is not None]
    domains = sorted({c["domain"] for c in cells})
    per_domain = {d: [c for c in cells if c["domain"] == d] for d in domains}

    def mean_over_cells(rows, fn):
        return float(np.mean([fn(c) for c in rows]))

    def table(name, k):
        out = {}
        for d, rows in per_domain.items():
            if all(name in c["selectors"] for c in rows):
                out[d] = {m: mean_over_cells(rows, lambda c: c["selectors"][name][str(k)][m]) for m in METRICS}
        return out

    summary = {"stratum": stratum, "cells": len(cells), "cells_total": total, "domains": domains,
               "novelty_restricted": restricted, "k": {}}
    for k in KS:
        pool = {d: {m: mean_over_cells(rows, lambda c: c["pool_mean"][m]) for m in METRICS}
                for d, rows in per_domain.items()}
        sel = {name: table(name, k) for name in SELECTORS}
        block = {"pool_mean": {"per_domain": pool, "mean": _mean_metrics(pool)}, "selectors": {}, "contrasts": {}}
        for name, per in sel.items():
            if not per:
                continue
            gain = {d: {"gain_tm": per[d]["tm_score"] - pool[d]["tm_score"],
                        "identity_shift": per[d]["identity_to_wild_type"] - pool[d]["identity_to_wild_type"],
                        "max_msa_identity_shift": per[d]["max_msa_identity"] - pool[d]["max_msa_identity"]}
                    for d in per}
            block["selectors"][name] = {
                "n_domains": len(per), "per_domain_selected": per, "per_domain_gain": gain,
                "mean_selected": _mean_metrics(per), "mean_gain": _mean_metrics(gain)}
        for cname, a, b in CONTRASTS:
            if a in sel and b in sel and sel[a] and sel[b]:
                shared = sorted(set(sel[a]) & set(sel[b]))
                diff = {d: sel[a][d]["tm_score"] - sel[b][d]["tm_score"] for d in shared}
                idiff = {d: sel[a][d]["identity_to_wild_type"] - sel[b][d]["identity_to_wild_type"] for d in shared}
                block["contrasts"][cname] = {
                    "n_domains": len(shared), "per_domain_tm": diff, "mean_tm": float(np.mean(list(diff.values()))),
                    "per_domain_identity": idiff, "mean_identity": float(np.mean(list(idiff.values())))}
        summary["k"][str(k)] = block
    return summary


def shrinkage(unrestricted_all, unrestricted_matched, restricted):
    """Per k and selector: cross-domain mean gains in each stratum and the shrinkage from unrestricted to restricted."""

    out = {}
    for k, block in restricted["k"].items():
        out[k] = {}
        for name, sel in block["selectors"].items():
            gain = sel["mean_gain"]["gain_tm"]
            row = {"n_domains": sel["n_domains"], "gain_restricted": gain}
            for tag, source in (("matched", unrestricted_matched), ("vs_all_cells", unrestricted_all)):
                base = source["k"][k]["selectors"].get(name)
                if base is None or abs(base["mean_gain"]["gain_tm"]) < 1e-12:
                    row[f"gain_unrestricted_{tag}"], row[f"shrinkage_{tag}"] = None, None
                else:
                    row[f"gain_unrestricted_{tag}"] = base["mean_gain"]["gain_tm"]
                    row[f"shrinkage_{tag}"] = 1.0 - gain / base["mean_gain"]["gain_tm"]
                row[f"shrinkage_{tag}_unreliable"] = (row[f"shrinkage_{tag}"] is None
                                                      or abs(base["mean_gain"]["gain_tm"]) < UNRELIABLE_BELOW_GAIN)
            out[k][name] = row
    return out


def _mean_metrics(per_domain):
    keys = next(iter(per_domain.values())).keys()
    return {m: float(np.mean([v[m] for v in per_domain.values()])) for m in keys}


def render(summary):
    lines = [f"== stratum {summary['stratum']} ({summary['cells']} of {summary['cells_total']} cells, domains {summary['domains']}) =="]
    for k, block in summary["k"].items():
        pool = block["pool_mean"]["mean"]
        lines.append(f"k={k}  pool TM {pool['tm_score']:.4f}  id_wt {pool['identity_to_wild_type']:.1f}  "
                     f"id_msa {pool['max_msa_identity']:.1f}")
        lines.append(f"  {'selector':<22}{'n':>2}{'TM':>8}{'gain':>8}{'id_wt':>8}{'d_id_wt':>8}{'id_msa':>8}{'d_id_msa':>9}")
        for name, s in block["selectors"].items():
            m, g = s["mean_selected"], s["mean_gain"]
            lines.append(f"  {name:<22}{s['n_domains']:>2}{m['tm_score']:8.4f}{g['gain_tm']:+8.4f}"
                         f"{m['identity_to_wild_type']:8.1f}{g['identity_shift']:+8.1f}"
                         f"{m['max_msa_identity']:8.1f}{g['max_msa_identity_shift']:+9.1f}")
        for cname, c in block["contrasts"].items():
            per = " ".join(f"{d}:{v:+.4f}" for d, v in c["per_domain_tm"].items())
            lines.append(f"  contrast {cname:<28} n={c['n_domains']} mean {c['mean_tm']:+.4f} "
                         f"(d_id_wt {c['mean_identity']:+.1f})  [{per}]")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", type=Path, required=True, help="per_sequence.jsonl of the measured cells")
    ap.add_argument("--references", type=Path, required=True, help="directory of family-reference <DOMAIN>.npz/.json")
    ap.add_argument("--out", type=Path, required=True, help="output JSON artefact")
    ap.add_argument("--permutation-seed", type=int, required=True,
                    help="the declared permuted_coupling seed of the generation protocol")
    args = ap.parse_args(argv)
    for option, path, is_dir in (("--records", args.records, False), ("--references", args.references, True)):
        if not (path.is_dir() if is_dir else path.is_file()):
            ap.error(f"{option} {path} does not exist")
    records = load_records(args.records)
    domains = sorted({r["domain"] for r in records})
    models = {d: load_family_model(d, args.references) for d in domains}
    cells, donors = analyse(records, models, args.permutation_seed)
    strata = [("all", cells)] + [(f"model={m}", [c for c in cells if c["model"] == m])
                                 for m in sorted({c["model"] for c in cells})]
    summaries, shrinkages = [], {}
    for name, rows in strata:
        full = summarise(rows, name)
        summaries.append(full)
        kept = [c for c in rows if c["restricted"] is not None]
        if kept:
            restricted = summarise(rows, f"{name} | identity < {NOVELTY_IDENTITY:g}", restricted=True)
            summaries.append(restricted)
            shrinkages[name] = shrinkage(full, summarise(kept, name), restricted)
    write_json(args.out, {
        "schema": SCHEMA, "statement": STATEMENT, "ks": list(KS), "selectors": list(SELECTORS),
        "contrasts": [list(c) for c in CONTRASTS], "permutation_seed": args.permutation_seed, "gauge": GAUGE,
        "full_score_gauge_check": "potts_full agrees across gauges (constant shift, identical ranking) in every pool",
        "pool_size": cells[0]["pool_size"], "cross_domain_donors": donors,
        "records": str(args.records), "references": str(args.references),
        "oracles": list(ORACLES), "oracle_statement": ORACLE_STATEMENT, "strata_statement": STRATA_STATEMENT,
        "novelty_identity_below": NOVELTY_IDENTITY, "minimum_admissible": MIN_ADMISSIBLE,
        "shrinkage": shrinkages, "summaries": summaries, "cells": cells})
    for s in summaries:
        print(render(s))
    print(f"sha256 {sha256_file(args.out)}  {args.out}")
    print(ORACLE_STATEMENT)
    for name, per_k in shrinkages.items():
        print(f"== shrinkage, stratum {name}: restricted gain vs unrestricted on matched cells / on all cells ==")
        for k, per in per_k.items():
            for sel, r in per.items():
                f = lambda v: "   n/a" if v is None else f"{v:+.4f}"
                flag = lambda t: " (unreliable)" if r[f"shrinkage_{t}_unreliable"] else ""
                print(f"k={k:<3}{sel:<24} restricted {f(r['gain_restricted'])}  matched {f(r['gain_unrestricted_matched'])} "
                      f"shrink {f(r['shrinkage_matched'])}{flag('matched')}  all-cells {f(r['gain_unrestricted_vs_all_cells'])} "
                      f"shrink {f(r['shrinkage_vs_all_cells'])}{flag('vs_all_cells')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
