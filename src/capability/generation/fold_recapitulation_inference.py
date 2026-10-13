"""E20 arm contrasts: domain is the independence unit.

Sequences within a (model, domain, arm) cell are independent draws given model
and arm and support within-domain description only. Every cross-domain claim
uses one number per domain (the cell mean) and a bootstrap over domains
(``core.statistics.paired_group_bootstrap``), refused below the project's
eight-unit floor (``MINIMUM_BOOTSTRAP_UNITS``). Sequences are never pooled
across domains.

The four main guidance arms are a 2x2 factorial in guidance content::

                 no couplings   couplings
    no fields        N              C
    fields           M              F

so the claim sits on the increments, not on the arms: simple effects C-N, M-N,
F-M, F-C; main effect of couplings ((C-N)+(F-M))/2; main effect of fields
((M-N)+(F-C))/2; interaction (F-M)-(C-N). N is the (0,0) floor reference: a large
C-N or M-N alone is "guidance beats unconditional generation", not the finding.
No composite score and no ranking of arms is produced. Cx is compared against C
(specificity); P and H are external references, contrasted against N only.

Every contrast, for every endpoint (tm_score primary; plddt_mean and ptm
secondary, never combined), is emitted twice: unstratified, and restricted to
sequences with max MSA identity <= the novelty ceiling. A domain enters a
contrast only if every arm that contrast uses has sequences in that stratum; the
factorial family shares one domain set (all of N, M, C, F), and the excluded
domains are listed, never averaged in with partial data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor, paired_group_bootstrap
from .fold_recapitulation import NOVELTY_CEILING

ENDPOINTS = ("tm_score", "plddt_mean", "ptm")  # first is primary
FACTORIAL_ARMS = ("N", "M", "C", "F")

#: Linear contrasts over per-domain arm means.
SIMPLE = {
    "C-N": {"C": 1, "N": -1},
    "M-N": {"M": 1, "N": -1},
    "F-M": {"F": 1, "M": -1},
    "F-C": {"F": 1, "C": -1},
}
FACTORIAL = {
    **SIMPLE,
    "main_couplings": {"C": .5, "N": -.5, "F": .5, "M": -.5},
    "main_fields": {"M": .5, "N": -.5, "F": .5, "C": -.5},
    "interaction": {"F": 1, "M": -1, "C": -1, "N": 1},
}
PAIRWISE = {
    "Cx-N": {"Cx": 1, "N": -1},
    "C-Cx": {"C": 1, "Cx": -1},
    "P-N": {"P": 1, "N": -1},
    "H-N": {"H": 1, "N": -1},
}
READING_NOTE = (
    "The factorial increments carry the claim. N is the floor reference; "
    "guidance beating N is not the finding. No arm ranking or composite is reported."
)


def domain_means(rows: Sequence[Mapping[str, Any]], endpoint: str, *, restrict: bool) -> dict:
    """(model, domain, arm) -> (mean, n). Restricted drops rows above the ceiling."""

    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if restrict and r["max_msa_identity"] > NOVELTY_CEILING:
            continue
        cells[(r["model"], r["domain"], r["arm"])].append(float(r[endpoint]))
    return {k: (float(np.mean(v)), len(v)) for k, v in cells.items()}


def cross_domain_interval(values: Mapping[str, float], *, seed: int, n_bootstrap: int = 2000) -> dict:
    """Bootstrap of the mean over domains; raises below the eight-unit floor."""

    domains = sorted(values)
    v = np.array([values[d] for d in domains])
    if len(domains) < MINIMUM_BOOTSTRAP_UNITS:
        raise ValueError(bootstrap_unit_floor(len(domains))["degenerate_reason"])
    result = paired_group_bootstrap(
        np.zeros(len(v)), v, np.zeros(len(v)), domains, lambda _t, p: float(np.mean(p)),
        seed=seed, n_bootstrap=n_bootstrap,
    )
    return {"mean": result["difference"], "ci95": result["difference_ci95"], "n_domains": result["n_groups"]}


def contrast(means: Mapping, model: str, name: str, coefs: Mapping[str, float], domains: Sequence[str],
             *, seed: int, with_interval: bool) -> dict[str, Any]:
    per_domain, excluded = {}, {}
    for d in domains:
        missing = [a for a in coefs if (model, d, a) not in means]
        if missing:
            excluded[d] = f"no sequences for arm(s) {missing} in this stratum"
        else:
            per_domain[d] = float(sum(c * means[(model, d, a)][0] for a, c in coefs.items()))
    out: dict[str, Any] = {
        "contrast": name, "coefficients": dict(coefs), "per_domain": per_domain,
        "excluded_domains": excluded, "n_domains": len(per_domain),
        "mean_over_domains": float(np.mean(list(per_domain.values()))) if per_domain else None,
    }
    if with_interval:
        try:
            out["interval"] = cross_domain_interval(per_domain, seed=seed)
        except ValueError as error:
            out["interval"] = None
            out["interval_refused"] = str(error)
    return out


def analyse(rows: Sequence[Mapping[str, Any]], *, roles: Mapping[str, str], role: str,
            seed: int = 20261013, with_interval: bool = True) -> dict[str, Any]:
    """All contrasts for one role, per model, endpoint and stratum.

    ``role`` is "development" or "evaluation"; rows of other roles are dropped
    and the output records the role it covers. A contrast with fewer than eight domains yields per-domain values and an
    explicit interval refusal, listed in ``refusals`` (the stage exits non-zero
    on any refusal when role is evaluation).
    """

    if role not in ("development", "evaluation"):
        raise ValueError(f"role must be development or evaluation, not {role!r}")
    rows = [r for r in rows if roles[r["domain"]] == role]
    if not rows:
        raise ValueError(f"no records in role {role}")
    domains = sorted({r["domain"] for r in rows})
    models = sorted({r["model"] for r in rows})
    result: dict[str, Any] = {
        "role": role, "domains": domains, "novelty_ceiling_percent": NOVELTY_CEILING,
        "independence_unit": "domain", "minimum_domains": MINIMUM_BOOTSTRAP_UNITS,
        "reading_note": READING_NOTE, "endpoints": list(ENDPOINTS), "primary_endpoint": ENDPOINTS[0],
        "results": {},
    }
    for model in models:
        for endpoint in ENDPOINTS:
            for stratum, restrict in (("unstratified", False), ("novelty_restricted", True)):
                means = domain_means(rows, endpoint, restrict=restrict)
                complete = [d for d in domains if all((model, d, a) in means for a in FACTORIAL_ARMS)]
                block = {
                    "factorial": {
                        n: contrast(means, model, n, c, complete, seed=seed, with_interval=with_interval)
                        for n, c in FACTORIAL.items()
                    },
                    "pairwise": {
                        n: contrast(means, model, n, c, domains, seed=seed, with_interval=with_interval)
                        for n, c in PAIRWISE.items()
                    },
                    "cell_means": {f"{d}|{a}": means[(model, d, a)] for (m, d, a) in sorted(means) if m == model},
                    "factorial_domains_complete": complete,
                    "factorial_domains_excluded_partial_coverage": [d for d in domains if d not in complete],
                }
                result["results"][f"{model}|{endpoint}|{stratum}"] = block
    result["refusals"] = [
        f"{key} {c['contrast']}: {c['interval_refused']}"
        for key, b in result["results"].items()
        for c in list(b["factorial"].values()) + list(b["pairwise"].values())
        if c.get("interval_refused") and (c["n_domains"] > 0 or c["contrast"] in FACTORIAL)
    ]
    return result
