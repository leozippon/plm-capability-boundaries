"""E20 model-independent reference arms: ``P`` (Potts joint samples) and ``H`` (held-out natural homologs).

Neither arm has a language model. ``P`` is what the external evolutionary statistics achieve alone and
``H`` is what real homologs achieve, the natural ceiling; both calibrate the scale of the fold endpoint.
Records are the ``e20_generated_sequence_v1`` rows of :mod:`.fold_recapitulation`, whose language-model
fields are null for exactly these two arms and whose ``model`` is ``potts-joint`` / ``natural``.

Arm ``P`` draws from :func:`~..context.family_potts.sample_potts_joint`, the Gibbs sampler of the actual
Potts joint. It never uses ``sample_prefix_chain``, which samples a different distribution. A cell whose
chains fail the per-domain validation (:func:`~..context.family_potts.validate_gibbs_mixing`: two overdispersed
initialisations must agree to within a declared factor of the sampling-noise floor) raises
:class:`~..context.family_potts.MixingError` and yields no records: no retry with more sweeps, no fallback
sampler, no shorter pool. The first declaration, an absolute 0.02 half-run-versus-full-run tolerance, was
withdrawn: at 100 chains that statistic is Monte-Carlo noise (about 1/sqrt(n)), unattainable whatever the chain
does, and blind to chains stuck near their start.

Arm ``H`` takes MSA rows from the whole-cluster holdout recorded at fit time, and only rows that are
gapless across the family model's columns. The fold endpoint compares residue ``i`` with wild-type
residue ``i`` and no aligner is involved, so a gapped row would shift the correspondence and silently
give a wrong TM-score; it is excluded and counted, and a row of the wrong length is a refusal. Fewer
admissible rows than the protocol's minimum is recorded, not an error.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..context.family_potts import FamilyModel, sample_potts_joint, validate_gibbs_mixing
from ..core.amino_acids import AA20
from .fold_recapitulation import MODEL_FREE_ARMS, SCHEMA_VERSION

ATTEMPTS = 1


def contract_row(arm: str, domain: str, index: int, seed: int, sequence: str) -> dict[str, Any]:
    """One ``e20_generated_sequence_v1`` record of a model-independent arm."""

    return {
        "schema_version": SCHEMA_VERSION, "id": f"{MODEL_FREE_ARMS[arm]}|{domain}|{arm}|{seed}|{index:04d}",
        "domain": domain, "model": MODEL_FREE_ARMS[arm], "arm": arm, "lambda": 0.0, "seed": seed,
        "sequence": sequence, "native_loglik": None, "guidance_score": None, "mean_step_entropy": None,
        "attempt_budget": ATTEMPTS,
    }


def potts_joint_cell(model: FamilyModel, *, domain: str, chains: int, sweeps: int, validation_chains: int,
                     mixing_factor: float, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the sampler on this domain, then draw ``chains`` independent chains, one draw each.

    Validation (own seed stream ``[seed, 1]``) raises ``MixingError`` carrying the measured numbers, before any
    record exists. The draw uses the cell's stream seed from uniform-random initialisation. The record ``seed``
    is the cell seed for every draw: the chains share one generator, so no per-draw seed could regenerate a draw.
    """

    validation = validate_gibbs_mixing(model, np.random.default_rng(np.random.SeedSequence([seed, 1])),
                                       n=validation_chains, sweeps=sweeps, factor=mixing_factor)
    draws = sample_potts_joint(model, chains, np.random.default_rng(seed), sweeps=sweeps, mixing_tolerance=None)
    if len(draws.sequences) != chains or {len(s) for s in draws.sequences} != {model.length}:
        raise RuntimeError("the Gibbs sampler returned a pool that is not `chains` sequences of the domain length")
    manifest = {"sampler": "sample_potts_joint", "chains": chains, "draws_per_chain": 1, "sweeps": sweeps,
                "initialisation": "uniform random residues", "validation": validation, "seed": seed}
    return [contract_row("P", domain, i, seed, s) for i, s in enumerate(draws.sequences)], manifest


def natural_cell(msa: Sequence[tuple[str, str]], held_out: np.ndarray, *, domain: str, length: int,
                 max_rows: int, min_admissible: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Up to ``max_rows`` gapless held-out MSA rows; a deterministic subsample when more are admissible.

    ``msa`` is ``(name, row)`` in file order and ``held_out`` flags the rows of the recorded holdout.
    Every row, held out or not, must be ``length`` columns of residues and gaps; anything else is a
    refusal. Gapped held-out rows are excluded, never truncated or padded.
    """

    if len(held_out) != len(msa):
        raise ValueError(f"{domain}: holdout flags {len(held_out)} rows, the MSA has {len(msa)}")
    bad = [name for name, row in msa if len(row) != length or set(row) - set(AA20 + "-")]
    if bad:
        raise ValueError(f"{domain}: {len(bad)} MSA rows (e.g. {bad[0]}) are not {length} columns of "
                         "residues and gaps; identity correspondence needs the family model's columns")
    held = np.flatnonzero(held_out)
    admissible = [int(i) for i in held if "-" not in msa[i][1]]
    chosen = admissible
    if len(admissible) > max_rows:
        chosen = sorted(int(i) for i in np.random.default_rng(seed).choice(admissible, max_rows, replace=False))
    manifest = {
        "held_out_rows": int(len(held)), "gapped_held_out_rows_excluded": int(len(held) - len(admissible)),
        "admissible_rows": len(admissible), "distinct_admissible_sequences": len({msa[i][1] for i in admissible}),
        "max_rows": max_rows, "min_admissible": min_admissible,
        "below_min_admissible": len(admissible) < min_admissible,
        "subsampled": len(admissible) > max_rows, "seed": seed,
        "selected": [{"index": i, "name": msa[i][0]} for i in chosen],
    }
    return [contract_row("H", domain, k, seed, msa[i][1]) for k, i in enumerate(chosen)], manifest
