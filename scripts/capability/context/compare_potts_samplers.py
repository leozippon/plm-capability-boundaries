#!/usr/bin/env python3
"""Measure how far ``sample_prefix_chain`` is from the Potts joint that ``sample_potts_joint`` samples. CPU only.

Per domain, both samplers draw ``--draws`` sequences in each of two independent replicates. Written: mean Potts
energy and its standard error per sampler (replicates pooled); the single-site total-variation distance between
the samplers per column (matched replicate against matched replicate, averaged), its maximum and mean; and the
Monte-Carlo noise floor of the same statistic, the TV between the two replicates of one sampler, which is what
makes the between-sampler distances interpretable. The Gibbs sampler runs the declared schedule (200 sweeps);
the withdrawn half-run-versus-full-run statistic is recorded per replicate (it is Monte-Carlo noise of order
1/sqrt(n), which is why an absolute 0.02 was never attainable at these draw counts) and the declared per-domain
validation (two initialisations against a noise floor, factor 2.0) is recorded as well. Nothing is enforced or adjusted.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context.family_potts import (  # noqa: E402
    load_family_model, potts_score, sample_potts_joint, sample_prefix_chain, MixingError, validate_gibbs_mixing)
from src.capability.core.io import write_json  # noqa: E402

SWEEPS, WITHDRAWN_ABSOLUTE_TOLERANCE, MIXING_FACTOR, VALIDATION_CHAINS = 200, 0.02, 2.0, 4000
SAMPLERS = ("prefix_chain", "potts_joint")


def marginals(sequences: list[str], alphabet: str) -> np.ndarray:
    codes = np.array([[alphabet.index(c) for c in s] for s in sequences])
    return np.stack([np.bincount(codes[:, i], minlength=len(alphabet)) for i in range(codes.shape[1])]) / len(sequences)


def column_tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(axis=1)


def summarise(tv: np.ndarray) -> dict:
    return {"max": float(tv.max()), "mean": float(tv.mean()), "per_column": [float(x) for x in tv]}


def draw(model, sampler: str, n: int, seed_sequence: np.random.SeedSequence):
    rng = np.random.default_rng(seed_sequence)
    if sampler == "prefix_chain":
        return [sample_prefix_chain(model, rng) for _ in range(n)], None
    # tolerance 1.0 disables enforcement; the measured distance and the declared tolerance are recorded instead
    draws = sample_potts_joint(model, n, rng, sweeps=SWEEPS, mixing_tolerance=1.0)  # records the noise-level statistic
    return draws.sequences, draws.mixing_max_tv


def compare_domain(model, index: int, n: int, seed: int) -> dict:
    reps = {s: [] for s in SAMPLERS}
    mixing = []
    for s_i, sampler in enumerate(SAMPLERS):
        for rep in range(2):
            sequences, tv = draw(model, sampler, n, np.random.SeedSequence([seed, index, s_i, rep]))
            reps[sampler].append(sequences)
            if tv is not None:
                mixing.append(tv)
    try:
        validation = validate_gibbs_mixing(model, np.random.default_rng(np.random.SeedSequence([seed, index, 9])),
                                           n=VALIDATION_CHAINS, sweeps=SWEEPS, factor=MIXING_FACTOR)
    except MixingError as error:
        validation = error.numbers
    energy = {s: np.array([potts_score(model, x) for r in reps[s] for x in r]) for s in SAMPLERS}
    stat = {s: {"mean": float(e.mean()), "standard_error": float(e.std(ddof=1) / np.sqrt(len(e))), "n": len(e)}
            for s, e in energy.items()}
    gap = stat["potts_joint"]["mean"] - stat["prefix_chain"]["mean"]
    se = np.hypot(stat["potts_joint"]["standard_error"], stat["prefix_chain"]["standard_error"])
    m = {s: [marginals(r, model.alphabet) for r in reps[s]] for s in SAMPLERS}
    between = np.mean([column_tv(m["prefix_chain"][r], m["potts_joint"][r]) for r in range(2)], axis=0)
    return {
        "domain": model.domain, "length": model.length, "energy": stat,
        "energy_gap_joint_minus_prefix": gap, "energy_gap_in_standard_errors": float(gap / se),
        "between_sampler_tv": summarise(between),
        "noise_floor_tv": {s: summarise(column_tv(*m[s])) for s in SAMPLERS},
        "gibbs_validation": validation,
        "superseded_half_vs_full_mixing_tv": {
            "status": "SUPERSEDED: noise-limited statistic, not evidence about mixing; kept so the figures the record cited stay readable",
            "max_tv_per_replicate": mixing,
            "meets_withdrawn_absolute_tolerance": [t <= WITHDRAWN_ABSOLUTE_TOLERANCE for t in mixing],
            "reason": "its sampling-noise floor at these draw counts is the same size as the values (see noise_floor_tv)"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family-reference-dir", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True, help="development domains only")
    parser.add_argument("--draws", type=int, default=2000, help="draws per sampler per replicate")
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    results = []
    for index, domain in enumerate(args.domains):
        results.append(compare_domain(load_family_model(domain, args.family_reference_dir), index, args.draws, args.seed))
        print(domain, round(results[-1]["energy_gap_in_standard_errors"], 1), round(results[-1]["between_sampler_tv"]["max"], 3), flush=True)
    write_json(args.out, {
        "schema": "potts_sampler_comparison_v1", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "family_reference_dir": str(args.family_reference_dir), "draws_per_sampler_per_replicate": args.draws,
        "replicates": 2, "seed": args.seed, "seed_scheme": "SeedSequence([seed, domain_index, sampler_index, replicate])",
        "gibbs": {"sweeps": SWEEPS, "enforced_here": False,
                  "mixing_factor": MIXING_FACTOR, "validation_chains_per_init": VALIDATION_CHAINS,
                  "withdrawn_absolute_tolerance": WITHDRAWN_ABSOLUTE_TOLERANCE,
                  "withdrawn_reason": "the half-run-versus-full-run TV is Monte-Carlo noise (about 1/sqrt(n)): "
                                      "at 2000 chains it is 0.055-0.076 here, so an absolute 0.02 is unattainable "
                                      "at the declared draw counts whatever the chain does"},
        "provenance_note": "family_potts.py gained MixingError and validate_gibbs_mixing after the 232 development "
                           "cells; new guided cells record a different code_sha256 for it. No guided-decoding path "
                           "calls them and sample_potts_joint draws are bit-identical at a fixed seed.",
        "samplers": {"prefix_chain": "sample_prefix_chain: prefix-truncated conditionals, not the Potts joint",
                     "potts_joint": "sample_potts_joint: Gibbs sampler of P_L(x) proportional to exp(E(x))"},
        "domains": results,
    })


if __name__ == "__main__":
    main()
