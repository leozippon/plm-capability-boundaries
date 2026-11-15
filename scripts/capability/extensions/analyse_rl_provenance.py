#!/usr/bin/env python3
"""Assemble the reward-provenance experiment's declared families and read them.

Declared in ``docs/rl-environment-provenance.md``. This producer fits nothing and
chooses nothing: it reads the pool value tables ``sample_rl_provenance_pools.py``
wrote, forms each declared family in its declared order, bands it under the
project's native simultaneous statistic, derives ``k*``, and applies the decision
rules exactly as they were written before any policy was trained. Every estimand,
threshold and seed comes from
``src/capability/extensions/rl_environment_provenance.py``.

A family whose input has not been produced is recorded unresolved with the reason
stated, not dropped. Three of the declared families read tables this producer
does not compute -- the cross-likelihood matrix, the entropy of each pool under
its own model, the held-out natural likelihood, and the identity strata -- each of
which is its own instrument with its own convention, and each of which is passed
in or declared not measured. That is the difference between a column with no
estimate and a family that quietly shrank.

Usage:
    python scripts/capability/extensions/analyse_rl_provenance.py \
        --pools <pool dir> --out <result dir> \
        [--likelihood <cross-nll json>] [--novelty <novelty json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.extensions import rl_environment_provenance as rp  # noqa: E402
from src.capability.extensions import rl_provenance_analysis as ra  # noqa: E402
from src.capability.extensions import sequence_reward as sr  # noqa: E402

RESULT = "rl_provenance_result.json"

#: Which declared family reads which quantity of the pool value tables, and from
#: which folded realisation. The closed-form quantities carry no realisation
#: because no structure enters them.
QUANTITY_FAMILIES: dict[str, tuple[str, str]] = {
    rp.FAMILY_PLDDT: ("mean_ca_plddt", "A"),
    rp.FAMILY_LENGTH: ("length", "closed_form"),
    rp.FAMILY_DISTINCT_5MER: ("distinct_5mer_fraction", "closed_form"),
    rp.FAMILY_LONGEST_RUN: ("longest_single_residue_run", "closed_form"),
    rp.FAMILY_COMPOSITION_ENTROPY: ("composition_entropy_bits", "closed_form"),
    rp.FAMILY_LOW_COMPLEXITY: ("low_complexity_fraction", "closed_form"),
    rp.FAMILY_CHARGE_RATE: ("hit_charge_window", "closed_form"),
    rp.FAMILY_CONJUNCTION_RATE: ("hit_environment_conjunction", "closed_form"),
    rp.FAMILY_HARD_CONJUNCTION_RATE: ("hit_environment_hard_conjunction", "closed_form"),
    rp.FAMILY_MEMORY_CONJUNCTION_RATE: ("hit_memory_conjunction", "A"),
    rp.FAMILY_PLDDT_CRITERION_RATE: ("hit_plddt_criterion", "A"),
}

#: Families whose input is a separate instrument this producer does not run.
#: Named here so the result records them as not measured with their reason
#: rather than leaving a declared family silently absent.
EXTERNAL_FAMILIES: dict[str, str] = {
    rp.FAMILY_CROSS_NLL: "the cross-likelihood matrix, scored under the project's "
                         "own nats-per-residue convention with the stop token excluded",
    rp.FAMILY_ENTROPY: "each pool's token-mean predictive entropy under its own model",
    rp.FAMILY_HELD_OUT_NLL: "the held-out natural-sequence likelihood under each policy",
    rp.FAMILY_IDENTITY_MEAN: "the DIAMOND search over UniRef50, identity over the query",
    rp.FAMILY_IDENTITY_FRACTION: "the same search's fraction below 30% identity",
    rp.FAMILY_SUPPORT_TAIL: "the base model's per-residue NLL of every pool, which the "
                            "support reading is a tail share of",
}


def read_per_sequence(pools: Path) -> dict[str, list[dict[str, Any]]]:
    """Each pool's per-sequence table, which carries length beside every reward.

    The stratified families need length and value on the *same* row, and the
    per-stream value tables carry them as separate lists whose alignment is
    positional. Reading the joined table the collect stage already writes is what
    makes the stratification robust to a quantity that skipped a row.
    """

    out: dict[str, list[dict[str, Any]]] = {}
    for pool in rp.EVALUATED_POOLS:
        path = Path(pools) / f"{pool}_per_sequence.jsonl"
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        out[pool] = rows
    return out


def stratified_streams(rows: Sequence[Mapping[str, Any]], *, field: str,
                       stratum: str) -> dict[int, list[float]]:
    """One pool's values for one length stratum, laid out by stream.

    Only completions the structural term is defined on reach a stratum at all, so
    a stratified structural contrast and the structural reward share one domain.
    """

    out: dict[int, list[float]] = {}
    for row in rows:
        if rp.length_stratum(int(row["length"])) != stratum:
            continue
        value = row.get(field)
        if value is None:
            continue
        out.setdefault(int(row["stream"]), []).append(float(value))
    return out


def stratified_family(per_sequence: Mapping[str, Sequence[Mapping[str, Any]]], *,
                      family: str, field: str) -> dict[str, Any]:
    """One family of pool-minus-base contrasts, crossed with the length strata.

    A column is ``pool|length=<stratum>``. The strata are declared from the base
    policy's own length distribution and fixed before any trained pool was read,
    so a stratum that leaves a column unresolved does so because the pools do not
    overlap there and not because the edges were chosen to make it so.
    """

    base = per_sequence.get(rp.POOL_BASE)
    columns = []
    for pool in rp.CONTRAST_POOLS:
        side = per_sequence.get(pool)
        for stratum in rp.LENGTH_STRATUM_NAMES:
            key = f"{pool}|length={stratum}"
            if base is None or side is None:
                absent = rp.POOL_BASE if base is None else pool
                columns.append({"key": key,
                                "skip_reason": f"no per-sequence table for pool "
                                               f"{absent!r}"})
                continue
            columns.append({"key": key,
                            "left": stratified_streams(side, field=field,
                                                       stratum=stratum),
                            "right": stratified_streams(base, field=field,
                                                        stratum=stratum)})
    record = ra.contrast_bands(columns, family=family)
    record["stratification"] = {
        "edges": list(rp.LENGTH_STRATUM_EDGES),
        "names": list(rp.LENGTH_STRATUM_NAMES),
        "field": field,
        "rule": rp.STRATIFICATION_RULE,
        "role": ("derived diagnostic; the gate reads the reward gain alone"
                 if family in rp.STRATIFIED_DIAGNOSTIC_FAMILIES else "verdict input"),
    }
    if family == rp.FAMILY_PLDDT_CRITERION_RATE_BY_LENGTH:
        record["stratification"]["why"] = rp.CRITERION_RATE_AMPLIFICATION
    return record


def read_values(pools: Path) -> dict[str, dict[str, dict[str, dict[int, list[float]]]]]:
    """Each pool's value tables, with stream keys as integers.

    JSON has no integer keys, so the sampler's stream labels come back as
    strings; the resampling machinery keys its declared seed on the stream's
    position, so the keys are restored to integers here rather than sorted as
    text, where stream 10 would precede stream 2.
    """

    out: dict[str, Any] = {}
    for pool in rp.EVALUATED_POOLS:
        path = Path(pools) / f"{pool}_{'pool_values.json'}"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[pool] = {
            quantity: {realisation: {int(stream): [float(value) for value in items]
                                     for stream, items in streams.items()}
                       for realisation, streams in by_realisation.items()}
            for quantity, by_realisation in payload["values"].items()
        }
    if rp.POOL_BASE not in out:
        raise SystemExit(
            f"no base pool value table under {pools}; the base pool is the control for "
            "every column and no family can be formed without it"
        )
    return out


def _streams(values: Mapping[str, Any], pool: str, quantity: str,
             realisation: str) -> dict[int, list[float]] | None:
    return values.get(pool, {}).get(quantity, {}).get(realisation)


def pool_contrast_family(values: Mapping[str, Any], *, family: str, quantity: str,
                         realisation: str) -> dict[str, Any]:
    """One family of pool-minus-base contrasts on one quantity."""

    base = _streams(values, rp.POOL_BASE, quantity, realisation)
    columns = []
    for pool in rp.CONTRAST_POOLS:
        side = _streams(values, pool, quantity, realisation)
        if base is None or side is None:
            absent = rp.POOL_BASE if base is None else pool
            columns.append({"key": pool,
                            "skip_reason": f"no {quantity!r} values at realisation "
                                           f"{realisation!r} for pool {absent!r}"})
            continue
        columns.append({"key": pool, "left": side, "right": base})
    return ra.contrast_bands(columns, family=family)


def reward_gain_families(values: Mapping[str, Any]) -> dict[str, Any]:
    """One family per reward: every pool's gain over the base under that reward.

    Every pool is read under every reward, which is how the experiment asks
    whether an environment-supplied bias cost the memory-derived score. The
    realisation is A throughout; a pool mean is unbiased under either.
    """

    out = {}
    for reward in sr.REWARD_NAMES:
        family = rp.family_name(rp.FAMILY_REWARD_GAIN, reward)
        out[family] = pool_contrast_family(values, family=family, quantity=reward,
                                           realisation="A")
    return out


def best_of_k_families(values: Mapping[str, Any]) -> dict[str, Any]:
    """One family per reward: R@k over the base pool minus each pool's mean.

    The base pool's resampled ``R@k`` curve is drawn **once** per reward, at
    column zero of the family, and every one of that family's ten ``k`` columns
    reads its own slice of that one draw. That is not an economy: the curve's
    columns have to come from the same resampled base pool or the band would
    treat ten readings of one pool as ten independent pools, and ``k*`` would be
    read off a family whose correlation structure had been thrown away.

    Selection is on realisation A and evaluation on realisation B wherever the
    reward contains the pLDDT term. For the exactly computable reward the two
    realisations are the same numbers and the cross-fit is a no-op, which is
    stated in the record rather than special-cased away.
    """

    out = {}
    for reward in sr.REWARD_NAMES:
        family = rp.family_name(rp.FAMILY_BEST_OF_K, reward)
        selection = _streams(values, rp.POOL_BASE, reward, "A")
        needs_cross_fit = sr.REWARD_NEEDS_STRUCTURE[reward]
        evaluation = (_streams(values, rp.POOL_BASE, reward, "B") if needs_cross_fit
                      else selection)
        columns = []
        curve = None
        point = None
        if selection is not None and evaluation is not None:
            curve = ra.best_of_k_draws(selection, evaluation, family=family, column=0,
                                       ks=rp.K_GRID)
            order = sorted(selection)
            pooled_selection = np.concatenate([np.asarray(selection[stream], dtype=float)
                                               for stream in order])
            pooled_evaluation = np.concatenate([np.asarray(evaluation[stream], dtype=float)
                                                for stream in order])
            # Stream-equal, so that the k = 1 column is exactly minus the reward
            # gain and k = 1 is the no-selection baseline it is meant to be.
            probability = ra.stream_equal_member_probability(
                [len(selection[stream]) for stream in order])
            point = ra.best_of_k(pooled_selection, pooled_evaluation, rp.K_GRID,
                                 probability)
        for pool in rp.CONTRAST_POOLS:
            side = _streams(values, pool, reward, "A")
            for index, k in enumerate(rp.K_GRID):
                key = f"{pool}|k={k}"
                if curve is None or side is None:
                    missing = ("the base pool's two folded realisations"
                               if curve is None else f"pool {pool!r}")
                    columns.append({"key": key,
                                    "skip_reason": f"no {reward!r} values for {missing}"})
                    continue
                column = len(columns)
                columns.append({
                    "key": key,
                    "units": f"{reward} reward",
                    "left_point": float(point[index]),
                    "right_point": ra.pool_mean(side),
                    "left_draws": curve[:, index],
                    "right_draws": ra.pool_draws(side, family=family, column=column,
                                                 side=1),
                })
        record = ra.contrast_bands(columns, family=family)
        record["cross_fit"] = {
            "applies": bool(needs_cross_fit),
            "selection_realisation": "A",
            "evaluation_realisation": "B" if needs_cross_fit else "A",
            "note": rp.CROSS_FIT_NOTE if needs_cross_fit else
                    "this reward is exactly computable and carries no realisation "
                    "noise, so selecting and evaluating on one realisation is not a "
                    "winner's curse and the two realisations are the same numbers",
        }
        record["base_curve_drawn_once"] = (
            "one resample of the base pool serves all ten k columns of this family, so "
            "their deviations carry the correlation a curve over one pool has"
        )
        out[family] = record
    return out


def external_families(supplied: Mapping[str, Any]) -> dict[str, Any]:
    """The declared families whose instrument this producer does not run."""

    out = {}
    for family, instrument in EXTERNAL_FAMILIES.items():
        if family in supplied:
            out[family] = supplied[family]
            continue
        columns = [{"key": pool, "skip_reason": f"not measured: {instrument} has not been "
                                                "produced for this campaign"}
                   for pool in rp.CONTRAST_POOLS]
        if family == rp.FAMILY_CROSS_NLL:
            columns = [{"key": f"{pool}|{direction}",
                        "skip_reason": f"not measured: {instrument} has not been produced"}
                       for pool in rp.CONTRAST_POOLS
                       for direction in ("pool_under_base", "base_under_pool")]
        out[family] = ra.contrast_bands(columns, family=family)
    return out


def base_quantiles(values: Mapping[str, Any]) -> dict[str, Any]:
    """The base-pool reference points the declared degeneracy bounds are read against."""

    def pooled(quantity: str, realisation: str) -> np.ndarray | None:
        streams = _streams(values, rp.POOL_BASE, quantity, realisation)
        if streams is None:
            return None
        return np.concatenate([np.asarray(streams[stream], dtype=float)
                               for stream in sorted(streams)])

    plddt = pooled("mean_ca_plddt", "A")
    shingle = pooled("distinct_5mer_fraction", "closed_form")
    runs = pooled("longest_single_residue_run", "closed_form")
    entropy = pooled("composition_entropy_bits", "closed_form")
    out: dict[str, Any] = {}
    if plddt is not None:
        out["mean_ca_plddt"] = float(plddt.mean())
    if shingle is not None:
        out["distinct_5mer_fraction_mean"] = float(shingle.mean())
        out["distinct_5mer_fraction_quantile"] = float(
            np.quantile(shingle, rp.DISTINCT_5MER_BASE_QUANTILE))
    if runs is not None:
        out["longest_single_residue_run_mean"] = float(runs.mean())
        out["longest_single_residue_run_quantile"] = float(
            np.quantile(runs, rp.LONGEST_RUN_BASE_QUANTILE))
    if entropy is not None:
        out["composition_entropy_bits"] = float(entropy.mean())
    out["quantiles"] = {"distinct_5mer_fraction": rp.DISTINCT_5MER_BASE_QUANTILE,
                        "longest_single_residue_run": rp.LONGEST_RUN_BASE_QUANTILE}
    return out


def reachability(pools: Path) -> dict[str, Any]:
    """The base pool's declared reachability ladder, from its closed-form statistics.

    Declared before any evaluation pool existed. It is the half of the
    positive-control question the data in hand can answer: whether the declared
    charge target was reachable by selection from the base at all, and how much
    selection pressure each harder rung would need. The other half -- whether
    reinforcement learning installs a bias against a target selection cannot reach
    -- needs a policy trained against a harder rung and is not an analysis.
    """

    from src.capability.extensions import sequence_reward as sr

    path = Path(pools) / f"{rp.POOL_BASE}_closed_form_statistics.jsonl"
    if not path.is_file():
        return {"diagnostic": "base_pool_reachability_ladder", "status": "not_measured",
                "reason": f"no closed-form statistics for the base pool at {path}",
                "note": rp.LADDER_NOTE}
    streams: dict[int, list[Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["length"]) == 0:
            continue
        streams.setdefault(int(row["stream"]), []).append(
            sr.SequenceStatistics(
                sequence="", length=int(row["length"]),
                net_charge=float(row["net_charge_ph7"]),
                low_complexity_fraction=float(row["low_complexity_fraction"]),
                composition_entropy_bits=float(row["composition_entropy_bits"]),
                longest_single_residue_run=int(row["longest_single_residue_run"]),
                distinct_5mer_fraction=float(row["distinct_5mer_fraction"]),
                non_canonical_residues=int(row["non_canonical_residues"]),
            )
        )
    curve = ra.reachability_curve(streams)
    attempts = sum(len(values) for values in streams.values())
    # Both rules are kept. The first, posed on the conjunction indicator, is
    # recorded with what it selected and why it was voided; the second, posed on
    # the shaped reward, is the one the arm trained under.
    curve["rung_choice_conjunction_rule"] = ra.choose_hard_rung(curve["rungs"],
                                                                n=attempts)
    curve["rung_choice"] = ra.choose_shaped_rung(curve["rungs"])
    curve["rung_rule_voided"] = rp.RUNG_RULE_VOIDED
    curve["base_pool_attempts_scored"] = attempts
    return curve


def read_pool(pool: str, families: Mapping[str, Any], reference: Mapping[str, Any],
              support: Mapping[str, Any] | None) -> dict[str, Any]:
    """One pool's reading, by the decision rules declared before anything was trained."""

    own = rp.ARM_BY_NAME[pool].reward if pool in rp.ARM_BY_NAME else sr.REWARD_MEMORY
    gain_family = families[rp.family_name(rp.FAMILY_REWARD_GAIN, own)]
    gain = gain_family["columns"].get(pool, {})
    optimised = ra.optimisation_finding(gain.get("interval"))
    k = ra.k_star(families[rp.family_name(rp.FAMILY_BEST_OF_K, own)]["columns"], pool=pool)
    stratified = None
    if own in rp.structural_rewards():
        stratified = ra.stratified_gain(
            families[rp.family_name(rp.FAMILY_REWARD_GAIN_BY_LENGTH, own)]["columns"],
            pool=pool,
        )
    degeneracy = ra.degeneracy_breaches(
        {family: families[family]["columns"].get(pool, {}) for family in rp.DEGENERACY_BOUNDS
         if family in families},
        reference,
    ) if reference else {"breached": False, "breaches": {},
                         "unmeasured_bounds": list(rp.DEGENERACY_BOUNDS),
                         "bounds": dict(rp.DEGENERACY_BOUNDS)}
    if support is None:
        verdict = {"reading": rp.READING_UNRESOLVED,
                   "reason": "the support reading needs the base model's own per-residue "
                             "NLL of this pool, which has not been produced; without it "
                             "an installed bias cannot be distinguished from a "
                             "reweighting inside the base's support"} \
            if optimised.get("optimised") is True and k.get("redistribution") is False \
            else ra.reading(optimised=optimised, k=k,
                            support={"share_above": 0.0, "null_share":
                                     1 - rp.SUPPORT_TAIL_QUANTILE},
                            degeneracy=degeneracy, stratified=stratified)
    else:
        verdict = ra.reading(optimised=optimised, k=k, support=support,
                             degeneracy=degeneracy, stratified=stratified)
    attainment_family = rp.OBJECTIVE_ATTAINMENT_FAMILY.get(own)
    attainment = None
    if attainment_family in families:
        columns = families[attainment_family]["columns"]
        attainment = ra.objective_attainment(
            gain, columns.get(pool),
            base_rate=(columns.get(pool, {}) or {}).get("right_point"),
        )
    return {
        "pool": pool,
        "own_reward": own,
        "objective_attainment": attainment,
        "objective_attainment_family": attainment_family,
        "trained": pool in rp.TRAINED_ARMS,
        "reward_gain": {"point": gain.get("point"), "interval": gain.get("interval"),
                        "status": gain.get("status"), "reason": gain.get("reason")},
        "optimisation": optimised,
        "k_star": k,
        "support": support,
        "degeneracy": degeneracy,
        "length_stratified_gain": stratified,
        **verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pools", type=Path, required=True,
                        help="directory holding the pools' value tables")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supplied", type=Path, action="append", default=None,
                        help="JSON of already-banded external families, keyed by family "
                             "name; repeatable, one per instrument that produced some")
    parser.add_argument("--support", type=Path, default=None,
                        help="JSON of each pool's support-tail share, keyed by pool")
    args = parser.parse_args(argv)

    values = read_values(args.pools)
    supplied: dict[str, Any] = {}
    for path in args.supplied or ():
        for family, record in json.loads(Path(path).read_text(encoding="utf-8")).items():
            if family in supplied:
                raise SystemExit(
                    f"family {family!r} is supplied twice, by {path} and by an earlier "
                    "file; one family has one instrument and one banding"
                )
            if family not in EXTERNAL_FAMILIES:
                raise SystemExit(
                    f"{path} supplies {family!r}, which this producer forms itself from "
                    "the pool value tables; a family has one source"
                )
            supplied[family] = record
    support = (json.loads(args.support.read_text(encoding="utf-8"))
               if args.support else {})

    families: dict[str, Any] = {}
    families.update(reward_gain_families(values))
    families.update(best_of_k_families(values))
    for family, (quantity, realisation) in QUANTITY_FAMILIES.items():
        families[family] = pool_contrast_family(values, family=family, quantity=quantity,
                                                realisation=realisation)
    per_sequence = read_per_sequence(args.pools)
    for reward in rp.structural_rewards():
        families[rp.family_name(rp.FAMILY_REWARD_GAIN_BY_LENGTH, reward)] = (
            stratified_family(per_sequence,
                              family=rp.family_name(rp.FAMILY_REWARD_GAIN_BY_LENGTH,
                                                    reward),
                              field=f"reward_{reward}_A")
        )
    families[rp.FAMILY_PLDDT_BY_LENGTH] = stratified_family(
        per_sequence, family=rp.FAMILY_PLDDT_BY_LENGTH, field="mean_ca_plddt_A")
    families[rp.FAMILY_PLDDT_CRITERION_RATE_BY_LENGTH] = stratified_family(
        per_sequence, family=rp.FAMILY_PLDDT_CRITERION_RATE_BY_LENGTH,
        field="hit_plddt_criterion")
    families.update(external_families(supplied))
    undeclared = sorted(set(families) ^ set(rp.FAMILY_SIZES))
    if undeclared:
        raise SystemExit(
            f"the realised family set is not the declared one; the difference is "
            f"{undeclared}. Changing a declared family is an amendment to "
            "docs/rl-environment-provenance.md, not a run-time decision"
        )

    reference = base_quantiles(values)
    ladder = reachability(args.pools)
    readings = {pool: read_pool(pool, families, reference, support.get(pool))
                for pool in rp.CONTRAST_POOLS}
    record = {
        "schema": rp.SCHEMA,
        "campaign": rp.CAMPAIGN,
        "stage": "analyse",
        "contract": rp.contract(),
        "pools_present": sorted(values),
        "pools_absent": [pool for pool in rp.EVALUATED_POOLS if pool not in values],
        "base_reference": reference,
        "base_reachability_ladder": ladder,
        "families": families,
        "readings": readings,
        "decision_rules": rp.DECISION_RULES,
        "falsification": rp.FALSIFICATION,
        "bounds": rp.BOUNDS,
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / RESULT, record)
    chosen = (ladder.get("rung_choice") or {}).get("chosen_window")
    print(json.dumps({"chosen_hard_rung": chosen}), flush=True)
    print(json.dumps({pool: {"reading": reading["reading"],
                             "reward_gain": reading["reward_gain"]["point"],
                             "k_star": reading["k_star"]["k_star"]}
                      for pool, reading in readings.items()}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
