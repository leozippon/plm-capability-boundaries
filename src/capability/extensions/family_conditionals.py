"""An observed residue's conditional log-probability on a natural HSP fragment.

Hold every other position of one natural fragment fixed, enumerate the twenty
residues at one observed query position, and read the model's canonical score of
each complete string::

    q_theta(a | rest) = softmax_a( - native_full_sequence_NLL(x^{i->a}) )

``q_theta`` is the local conditional distribution *induced* by a canonical
sequence score. It is not a native next-token probability, and it does not
recover the generative distribution: the fixed background carries family
information on its own, and pretraining exposure of that family is unknown. What
is measured is the observed residue's log-probability under that induced
conditional, against baselines normalised over the very same twenty strings.

Only one endpoint is added: the conditional log-score. Per-position rank-20
order, sampled generation and pairwise drift are out of scope.

The cohort, its fixed wild types, its splits and its native fragments come from
the frozen first-stage queue; that queue is read, never modified, and no
first-stage score is read. Selection is a deterministic function of the cohort
and this protocol alone, so it is frozen before any model output exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np

from ..core.amino_acids import AA20
from ..core.statistics import bootstrap_unit_floor
from .family_distribution import (
    SCHEMA_QUEUE,
    NativeRow,
    _atomic_json,
    _family_inference,
    _nll_and_tokens,
    checkpoint_identity,
    evaluation_rows,
    load_causal,
    load_json,
    markov_model,
    markov_score,
    profile_frequencies,
    realised_dtype,
    scientific_inputs,
    verify_loaded,
)

SCHEMA_INPUTS = "family_conditionals_inputs"
SCHEMA_SCORES = "family_conditionals_scores"
SCHEMA_ANALYSIS = "family_conditionals_analysis"
#: One write-once manifest, shared by every score document. Nothing is copied.
INPUT_MANIFEST = "conditional-inputs.json"
PARTITIONS = ("pilot", "evaluation")
BASELINES = ("profile", "markov")
#: A native index aligned to no query position: a subject insertion.
INSERTION = -1
AA_INDEX = {residue: index for index, residue in enumerate(AA20)}

SCORE_DEFINITION = (
    "q_theta(a|rest) = softmax over the declared AA20 order of the negated summed native NLL "
    "of the twenty complete single-substitution strings, stabilised with a log-sum-exp; the "
    "same twenty strings are scored once per context and the result is not renormalised by "
    "scored-token counts")
#: A normalised twenty-way vector may miss one by floating-point summation only.
PROBABILITY_MASS_TOLERANCE = 1e-6

REQUIRED_CONFIG = ("declaration", "parents", "selection", "baselines", "statistics", "output")


def read_config(path: Path) -> dict[str, Any]:
    """Load this layer's protocol, refusing a partial or inconsistent one."""

    config = load_json(path)
    missing = [key for key in REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError(f"{path}: incomplete protocol, missing {missing}")
    selection, statistics = config["selection"], config["statistics"]
    if selection["alphabet"] != "AA20":
        raise ValueError("this layer enumerates the canonical AA20 alphabet only")
    if selection["max_sites_per_unit"] < 1 or selection["max_contexts_per_site"] < 1:
        raise ValueError("the declared site and context caps are positive")
    if selection["min_training_rows_with_site"] < 1 or selection["min_heldout_rows_with_site"] < 1:
        raise ValueError("the declared per-position support floors are positive")
    if statistics["bootstrap_draws"] < 2 or statistics["min_inference_units"] < 2:
        raise ValueError("inference needs at least two draws and two families")
    if statistics["bootstrap_seed"] != selection["seed"]:
        raise ValueError("the declared selection and bootstrap seeds differ")
    for name in ("profile_pseudocount", "markov_pseudocount"):
        if config["baselines"][name] <= 0:
            raise ValueError(f"the declared {name} is positive")
    if config["output"] == config["parents"]["cohort"]:
        raise ValueError("this layer must not write into the parent cohort")
    return config


# ------------------------------------------------------------- parent cohort


def _parent(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The frozen first-stage cohort, and the declarations inherited from it.

    The cohort's own protocol snapshot is the authority for the roster, the pilot
    models, the precision policy and the token budget, so no second roster is
    written by hand in this layer.
    """

    path = Path(root) / config["parents"]["cohort"]
    queue = load_json(path)
    if not isinstance(queue, dict) or queue.get("schema") != SCHEMA_QUEUE:
        raise ValueError(f"{path}: not a frozen family-distribution cohort")
    protocol = queue["protocol"]
    scientific_inputs(root, protocol, queue)  # Read-only: never create a parent snapshot.
    declaration = protocol["declaration"]
    if declaration["id"] != config["parents"]["declaration_id"]:
        raise ValueError(f"{path}: the cohort declares {declaration['id']!r}, not "
                         f"{config['parents']['declaration_id']!r}")
    declarations = {
        "cohort": str(config["parents"]["cohort"]),
        "declaration_id": str(declaration["id"]),
        "roster": [str(arm) for arm in protocol["scoring"]["arms"]],
        "pilot_models": [str(arm) for arm in protocol["queue"]["pilot_models"]],
        "dtype_policy": str(protocol["scoring"]["dtype_policy"]),
        "score_function": str(protocol["scoring"]["score_function"]),
        "max_len": int(protocol["scoring"]["max_len"]),
        "units": {partition: sorted(entry["unit_id"] for entry in queue["units"]
                                    if entry.get("partition") == partition)
                  for partition in PARTITIONS},
    }
    if not declarations["roster"] or not set(declarations["pilot_models"]) <= set(declarations["roster"]):
        raise ValueError(f"{path}: the inherited roster and pilot models are inconsistent")
    return queue, declarations


def _training_rows(wildtype: dict[str, Any]) -> list[NativeRow]:
    rows = [NativeRow.of(payload) for payload in wildtype["rows"]
            if payload["split"] == "train" and payload["representative"]]
    if not rows:
        raise ValueError(f"{wildtype['wildtype_id']}: no training representative")
    return rows


def _coverage(rows: Sequence[NativeRow], length: int) -> np.ndarray:
    """How many of these rows carry an observation at each query position."""

    mapped = np.concatenate([row.query_index for row in rows])
    return np.bincount(mapped[mapped != INSERTION].astype(np.int64), minlength=length)


# ---------------------------------------------------------------- selection


def _draw_without_replacement(rng, count: int, size: int) -> list[int]:
    """Ascending indices drawn uniformly without replacement.

    A partial Fisher-Yates pass consumes only ``rng.random()`` doubles, so the
    frozen selection cannot move when a distribution method's values change
    between library versions.
    """

    if not 0 <= size <= count:
        raise ValueError("a without-replacement draw needs 0 <= size <= count")
    pool = list(range(count))
    drawn: list[int] = []
    for _ in range(size):
        drawn.append(pool.pop(min(int(rng.random() * len(pool)), len(pool) - 1)))
    return sorted(drawn)


def _unit_targets(entry: dict[str, Any], config: dict[str, Any], rng) -> dict[str, Any]:
    """The declared sites and fixed contexts of one independent unit."""

    rules = config["selection"]
    unit_id = str(entry["unit_id"])
    if entry["partition"] not in PARTITIONS:
        raise ValueError(f"{unit_id}: the cohort declares an unknown partition")
    if len(entry["wildtypes"]) != 1 or len(entry["wildtype_ids"]) != 1:
        raise ValueError(f"{unit_id}: a unit must carry exactly one fixed wild type")
    wildtype = entry["wildtypes"][0]
    wildtype_id = str(wildtype["wildtype_id"])
    length = len(wildtype["wildtype"])
    training = _training_rows(wildtype)
    heldout = evaluation_rows(wildtype)
    clusters = {str(payload["row_id"]): str(payload["cluster"]) for payload in wildtype["rows"]}
    train_counts, held_counts = _coverage(training, length), _coverage(heldout, length)
    eligible = np.flatnonzero(
        (train_counts >= int(rules["min_training_rows_with_site"]))
        & (held_counts >= int(rules["min_heldout_rows_with_site"])))

    sites = []
    taken = min(int(rules["max_sites_per_unit"]), int(eligible.size))
    for index in _draw_without_replacement(rng, int(eligible.size), taken):
        position = int(eligible[index])
        carriers = sorted((row for row in heldout if position in row.query_index),
                          key=lambda row: row.row_id)
        contexts = []
        for pick in _draw_without_replacement(
                rng, len(carriers), min(int(rules["max_contexts_per_site"]), len(carriers))):
            row = carriers[pick]
            mapped = np.flatnonzero(row.query_index == position)
            if mapped.size != 1:
                raise ValueError(f"{wildtype_id}/{row.row_id}: query position {position} "
                                 "is not carried exactly once")
            native_index = int(mapped[0])
            contexts.append({"row_id": row.row_id, "cluster": clusters[row.row_id],
                             "native": row.native, "native_index": native_index,
                             "observed": row.native[native_index]})
        sites.append({"query_position": position,
                      "training_rows_with_site": int(train_counts[position]),
                      "heldout_rows_with_site": int(held_counts[position]),
                      "contexts_available": len(carriers),
                      "contexts": contexts})
    unit = {"unit_id": unit_id, "partition": str(entry["partition"]),
            "selection_rank": int(entry["selection_rank"]),
            "family_clusters": [int(value) for value in entry["family_clusters"]],
            "wildtype_id": wildtype_id, "wildtype_length": length,
            "eligible_positions": int(eligible.size), "sites": sites}
    unit["baselines"] = _fit_parameters(unit, training, config)
    return unit


def selection_from_parent(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """The frozen selection: a function of the cohort and this protocol only."""

    queue, declarations = _parent(root, config)
    rng = np.random.default_rng(int(config["selection"]["seed"]))
    units = [_unit_targets(entry, config, rng)
             for entry in sorted(queue["units"], key=lambda entry: entry["unit_id"])]
    return {"schema": SCHEMA_INPUTS, "protocol": config, "alphabet": AA20,
            "parent": declarations, "units": units}


# ------------------------------------------------------------- baselines


@dataclass(frozen=True)
class _Chain:
    """A residue string for the Markov control, which reads only ``native``."""

    native: str


def variant_strings(native: str, index: int) -> list[str]:
    """The twenty complete strings, one per AA20 residue at ``index``."""

    if not 0 <= index < len(native):
        raise ValueError(f"native index {index} leaves a sequence of length {len(native)}")
    return [native[:index] + residue + native[index + 1:] for residue in AA20]


def check_variants(native: str, index: int, variants: Sequence[str]) -> None:
    """Twenty distinct candidates, one changed position, the original residue kept."""

    if set(native) - set(AA20):
        raise ValueError("the frozen native carries a non-AA20 residue")
    if len(variants) != len(AA20) or len(set(variants)) != len(AA20):
        raise ValueError("the candidate set is not twenty distinct strings")
    rest = native[:index] + native[index + 1:]
    for variant, residue in zip(variants, AA20):
        if len(variant) != len(native) or variant[:index] + variant[index + 1:] != rest:
            raise ValueError("a candidate changes a position other than the queried one")
        if variant[index] != residue:
            raise ValueError("a candidate carries another residue than its AA20 slot")
    if variants[AA20.index(native[index])] != native:
        raise ValueError("the original residue is not restored verbatim among the candidates")


def log_softmax(raw: Sequence[float] | np.ndarray) -> np.ndarray:
    """Twenty-way normalisation of canonical scores, shifted by their maximum."""

    values = np.asarray(raw, dtype=float)
    if values.shape != (len(AA20),) or not np.isfinite(values).all():
        raise ValueError("a conditional needs twenty finite AA20 values")
    peak = float(values.max())
    return values - (peak + math.log(float(np.exp(values - peak).sum())))


def _fit_parameters(unit: dict[str, Any], training: Sequence[NativeRow],
                    config: dict[str, Any]) -> dict[str, Any]:
    """Fit the declared controls on training representatives and freeze them.

    These parameters are part of the frozen selection, so a cohort whose training
    residues changed is refused even when every unit and row identifier is
    unchanged.
    """

    rules = config["baselines"]
    profile = profile_frequencies(training, pseudocount=float(rules["profile_pseudocount"]))
    markov = markov_model(training, pseudocount=float(rules["markov_pseudocount"]))
    positions: dict[str, list[float]] = {}
    for site in unit["sites"]:
        position = int(site["query_position"])
        frequency = profile.get(position)
        if frequency is None:
            raise ValueError(f"{unit['unit_id']}: position {position} has no training profile")
        positions[str(position)] = [float(value) for value in frequency]
    return {"profile_frequencies": positions,
            "markov": {"start": [float(value) for value in markov["start"]],
                       "transitions": [[float(value) for value in row]
                                       for row in markov["transitions"]]},
            "pseudocounts": {"profile": float(rules["profile_pseudocount"]),
                             "markov": float(rules["markov_pseudocount"])}}


def _attach_baselines(unit: dict[str, Any]) -> None:
    """Score every frozen target with both controls, on the complete strings.

    Both baselines are normalised over the same twenty variants the model sees.
    The Profile carries one observed-per-position distribution, so the untouched
    positions cancel exactly; the Markov baseline is evaluated on each complete
    string, never as a lone forward transition.
    """

    stored = unit["baselines"]
    markov = {"start": np.asarray(stored["markov"]["start"]),
              "transitions": np.asarray(stored["markov"]["transitions"])}
    for site in unit["sites"]:
        frequency = np.asarray(stored["profile_frequencies"][str(site["query_position"])])
        controlled = []
        for context in site["contexts"]:
            variants = variant_strings(context["native"], context["native_index"])
            check_variants(context["native"], context["native_index"], variants)
            controlled.append([float(value) for value in log_softmax(
                [markov_score(cast(NativeRow, _Chain(variant)), markov)
                 for variant in variants])])
        site["baseline"] = {
            "profile": [float(value) for value in log_softmax(np.log(frequency))],
            "markov": controlled,
        }


def _counts(units: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Declared targets and the resulting forward-pass budget, before any score."""

    partitions: dict[str, Any] = {}
    for partition in PARTITIONS:
        selected = [unit for unit in units if unit["partition"] == partition]
        sites = sum(len(unit["sites"]) for unit in selected)
        contexts = sum(len(site["contexts"]) for unit in selected for site in unit["sites"])
        partitions[partition] = {
            "units": len(selected),
            "units_with_sites": sum(bool(unit["sites"]) for unit in selected),
            "units_insufficient_support": sum(not unit["sites"] for unit in selected),
            "eligible_positions": sum(unit["eligible_positions"] for unit in selected),
            "sites": sites,
            "contexts": contexts,
            "sequences_per_arm": len(AA20) * contexts,
        }
    return {"alphabet": AA20, "partitions": partitions}


def manifest_from_parent(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Freeze targets, fitted parameters and counts, not redundant baseline scores."""

    selection = selection_from_parent(root, config)
    selection["counts"] = _counts(selection["units"])
    return selection


def prepare(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Freeze the input manifest once, or accept an identical rebuild."""

    manifest = manifest_from_parent(root, config)
    path = Path(root) / config["output"] / INPUT_MANIFEST
    if path.exists():
        if load_json(path) != manifest:
            raise ValueError(f"{path} already holds a different selection; move the run "
                             "directory aside instead of re-preparing over scored targets")
    elif any((Path(root) / config["output"] / "scores").glob("*/*.json")):
        raise ValueError("scores exist but the input manifest is missing; it is not rebuilt")
    else:
        _atomic_json(path, manifest)
    return manifest


def load_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Verify the complete manifest, then derive baseline scores only in memory."""

    path = Path(root) / config["output"] / INPUT_MANIFEST
    if not path.exists():
        if any((Path(root) / config["output"] / "scores").glob("*/*.json")):
            raise ValueError(f"{path}: scores exist but their input manifest is missing; "
                             "they are not adopted retroactively")
        raise ValueError(f"{path}: the input manifest is missing; run prepare first")
    stored = load_json(path)
    if stored.get("schema") != SCHEMA_INPUTS or stored.get("alphabet") != AA20:
        raise ValueError(f"{path}: not a conditional input manifest for this alphabet")
    if stored.get("protocol") != config:
        raise ValueError(f"{path}: the manifest was written under another declaration")
    if stored != manifest_from_parent(root, config):
        raise ValueError(f"{path}: the frozen manifest differs from the parent-derived inputs; "
                         "changed targets, parameters, counts or extra fields are not accepted")
    for unit in stored["units"]:
        _attach_baselines(unit)
    return stored


# ------------------------------------------------------------------ scoring


def _declared_dtype(arm: str) -> str:
    from ..interactions.pairwise_epistasis import ARM_DTYPE

    return "torch." + ARM_DTYPE.get(arm, "float32")


def validate_arm_partition(arm: str, partition: str, manifest: dict[str, Any]) -> None:
    """Refuse an arm the inherited declaration does not place on this partition."""

    parent = manifest["parent"]
    roster = parent["roster"]
    if arm not in roster:
        raise ValueError(f"{arm} is outside the inherited {len(roster)}-arm roster")
    if partition == "pilot":
        if arm not in parent["pilot_models"]:
            raise ValueError(f"{arm} is not one of the inherited pilot models "
                             f"{parent['pilot_models']}")
    elif partition != "evaluation":
        raise ValueError(f"unknown partition {partition!r}")
    if not any(unit["partition"] == partition and unit["sites"] for unit in manifest["units"]):
        raise ValueError(f"the frozen manifest declares no scored {partition} unit")


def score_config(config: dict[str, Any], manifest: dict[str, Any], *,
                 arm: str, partition: str) -> dict[str, Any]:
    """Every declared field a stored score document must still agree with."""

    return {
        "arm": arm,
        "partition": partition,
        "max_len": int(manifest["parent"]["max_len"]),
        "batch_size": 1,
        "alphabet": AA20,
        "cohort": manifest["parent"]["declaration_id"],
        "score_definition": SCORE_DEFINITION,
        "baseline_definition": config["baselines"]["normalisation"],
    }


def validate_score_document(document: dict[str, Any], config: dict[str, Any],
                            manifest: dict[str, Any], *, partition: str, arm: str) -> None:
    """The same manifest, score definition and loaded identity in both readers."""

    expected = score_config(config, manifest, arm=arm, partition=partition)
    if (document.get("schema") != SCHEMA_SCORES or document.get("config") != expected
            or document.get("scientific_inputs") != INPUT_MANIFEST):
        raise ValueError(f"{arm}/{partition}: score document was written under another declaration")
    checkpoint = document.get("checkpoint") or {}
    if (checkpoint.get("arm") != arm or not checkpoint.get("config", {}).get("model_type")
            or not checkpoint.get("tensor_files")):
        raise ValueError(f"{arm}/{partition}: missing or inconsistent checkpoint identity")
    if document.get("dtype") != _declared_dtype(arm):
        raise ValueError(f"{arm}/{partition}: loaded dtype differs from the inherited policy")


def _check_conditional(probabilities: np.ndarray, observed: int) -> None:
    """Require finite normalized log probabilities and nonpositive observed logp."""

    if not np.isfinite(probabilities).all():
        raise RuntimeError("a twenty-way conditional is not finite")
    if float(probabilities[observed]) > PROBABILITY_MASS_TOLERANCE:
        raise RuntimeError("the observed residue carries a positive conditional log-probability")
    total = float(np.exp(probabilities).sum())
    if abs(total - 1.0) > PROBABILITY_MASS_TOLERANCE:
        raise RuntimeError(f"the twenty conditional probabilities sum to {total!r}, not one")


def _stored_unit(stored: dict[str, Any], unit: dict[str, Any], label: str) -> None:
    """A stored unit must still describe exactly the frozen targets and scores."""

    expected = {str(site["query_position"]): site for site in unit["sites"]}
    if set(stored.get("sites", {})) != set(expected):
        raise ValueError(f"{label}: the score record and the frozen sites differ")
    for key, site in expected.items():
        record = stored["sites"][key]
        if list(record["row_ids"]) != [context["row_id"] for context in site["contexts"]]:
            raise ValueError(f"{label}/{key}: the scored contexts differ from the frozen targets")
        logprob = np.asarray(record["logprob"], dtype=float)
        if logprob.shape != (len(site["contexts"]), len(AA20)):
            raise ValueError(f"{label}/{key}: the conditional record has the wrong shape")
        if not np.isfinite(logprob).all():
            raise ValueError(f"{label}/{key}: a conditional log-probability is not finite")
        observed = np.asarray([AA_INDEX[context["observed"]] for context in site["contexts"]])
        if float(logprob[np.arange(observed.size), observed].max()) > PROBABILITY_MASS_TOLERANCE:
            raise ValueError(f"{label}/{key}: an observed residue carries positive log-probability")
        if np.abs(np.exp(logprob).sum(axis=1) - 1.0).max() > PROBABILITY_MASS_TOLERANCE:
            raise ValueError(f"{label}/{key}: a conditional does not sum to one")
        tokens = np.asarray(record["scored_tokens"], dtype=float)
        if tokens.shape != logprob.shape or not np.isfinite(tokens).all() or (tokens < 1).any():
            raise ValueError(f"{label}/{key}: the scored-token diagnostic is missing or invalid")


def score_partition(root: Path, config: dict[str, Any], *, arm: str, partition: str,
                    device: str, log=lambda message: None) -> dict[str, Any]:
    """Score one arm on one declared partition, resuming completed units."""

    manifest = load_manifest(root, config)
    validate_arm_partition(arm, partition, manifest)
    max_len = int(manifest["parent"]["max_len"])
    declared = score_config(config, manifest, arm=arm, partition=partition)
    checkpoint = checkpoint_identity(load_causal(arm, device=None))
    path = Path(root) / config["output"] / "scores" / arm / f"{partition}.json"
    document: dict[str, Any] = {"schema": SCHEMA_SCORES, "config": declared,
                                "scientific_inputs": INPUT_MANIFEST,
                                "checkpoint": checkpoint, "dtype": None,
                                "units": {}, "complete": False}
    if path.exists():
        existing = load_json(path)
        validate_score_document(existing, config, manifest, partition=partition, arm=arm)
        if existing["checkpoint"] != checkpoint:
            raise ValueError(f"{path}: checkpoint identity differs; use a fresh run directory")
        document = existing
    units = [unit for unit in manifest["units"] if unit["partition"] == partition and unit["sites"]]
    expected = {unit["unit_id"]: unit for unit in units}
    for unit_id, stored in document["units"].items():
        if unit_id not in expected:
            raise RuntimeError(f"{path}: unit {unit_id} is not a frozen {partition} unit with sites")
        _stored_unit(stored, expected[unit_id], f"{arm}/{partition}/{unit_id}")
    pending = [unit for unit in units if unit["unit_id"] not in document["units"]]
    if pending:
        handle = load_causal(arm, device=device)
        verify_loaded(handle, checkpoint)
        handle.model.eval().requires_grad_(False)
        document["dtype"] = realised_dtype(handle)
        for unit in pending:
            sites: dict[str, Any] = {}
            for site in unit["sites"]:
                record: dict[str, Any] = {"row_ids": [context["row_id"] for context in site["contexts"]],
                                          "logprob": [], "scored_tokens": []}
                for context in site["contexts"]:
                    variants = variant_strings(context["native"], context["native_index"])
                    check_variants(context["native"], context["native_index"], variants)
                    nll, tokens = _nll_and_tokens(handle, variants, batch_size=1, max_len=max_len)
                    probabilities = log_softmax(-np.asarray(nll, dtype=float))
                    _check_conditional(probabilities, AA_INDEX[context["observed"]])
                    record["logprob"].append([float(value) for value in probabilities])
                    record["scored_tokens"].append([int(value) for value in tokens])
                sites[str(site["query_position"])] = record
            document["units"][unit["unit_id"]] = {"wildtype_id": unit["wildtype_id"],
                                                  "sites": sites}
            _atomic_json(path, document)
            log(f"{arm}/{partition}/{unit['unit_id']}: {len(sites)} sites, "
                f"{sum(len(site['contexts']) for site in unit['sites'])} contexts")
    if set(document["units"]) != set(expected):
        raise RuntimeError(f"{path}: not every frozen target carries a twenty-way score")
    document["complete"] = True
    _atomic_json(path, document)
    return document


# ----------------------------------------------------------------- analysis


def column_labels(arms: Sequence[str]) -> list[str]:
    """One label per contrast: the model log-score minus one baseline's."""

    return [f"{arm}|minus_{baseline}" for arm in arms for baseline in BASELINES]


def _unit_logscores(unit: dict[str, Any], stored: dict[str, Any] | None = None) -> dict[str, float]:
    """One value per family: each site's context mean, then the mean over sites.

    A site contributes one value whatever its context count, so a wider site does
    not weigh more. Baselines are derived in memory from frozen parameters;
    the model values need that unit's score record.
    """

    values: dict[str, list[float]] = {"model": [], "profile": [], "markov": []}
    for site in unit["sites"]:
        observed = [AA_INDEX[context["observed"]] for context in site["contexts"]]
        if stored is not None:
            logprob = np.asarray(stored["sites"][str(site["query_position"])]["logprob"],
                                 dtype=float)
            values["model"].append(float(np.mean(logprob[np.arange(len(observed)), observed])))
        values["profile"].append(float(np.mean([site["baseline"]["profile"][index]
                                                for index in observed])))
        values["markov"].append(float(np.mean([vector[index] for vector, index in
                                              zip(site["baseline"]["markov"], observed)])))
    return {key: float(np.mean(value)) for key, value in values.items() if value}


def analyse(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Family-equal conditional log-scores, the 66-column family and its controls."""

    manifest = load_manifest(root, config)
    output = Path(root) / config["output"]
    statistics = config["statistics"]
    roster = manifest["parent"]["roster"]
    summary: dict[str, Any] = {
        "schema": SCHEMA_ANALYSIS, "alphabet": AA20, "parent": manifest["parent"],
        "partitions": {}, "units": {}, "unscored_partitions": {}, "arms": [],
        "declared": manifest["counts"]["partitions"],
        "uniform_aa_logscore": -math.log(len(AA20)),
        "weighting": "one value per site (its context mean), then one value per family "
                     "(its site mean); families are equally weighted, so a site's weight does "
                     "not depend on how many contexts it carries",
        "notes": [
            "The endpoint is q_theta(a|rest): the local twenty-way conditional induced by the "
            "model's canonical sequence score with the rest of one natural HSP fragment fixed. "
            "It is not a native next-token probability and does not recover the generative "
            "distribution; the fixed background carries family information and pretraining "
            "exposure of that family is unknown.",
            "Both baselines are normalised over the same twenty complete variants as the model. "
            "The Markov description scores complete strings, never a lone forward transition, "
            "and the Profile description cancels the untouched positions exactly.",
            "The two contrasts per model are direct differences of observed log-scores "
            "(model minus Profile, model minus Markov), not a joint-model conditional increment. "
            + config["baselines"]["status"],
            "This 66-column family is new and separate from the first-stage R1/R2/R3 families "
            "and from the parent column-null family; they are never combined.",
            "The pilot partition is descriptive only. No formal completion or significance is "
            "reported until all 33 inherited arms cover every declared evaluation unit.",
            "r20 order, sampled generation and pairwise drift are not implemented here.",
        ],
    }
    for partition in PARTITIONS:
        units = [unit for unit in manifest["units"] if unit["partition"] == partition]
        documents = {}
        for path in sorted((output / "scores").glob(f"*/{partition}.json")):
            document = load_json(path)
            arm = str(document["config"]["arm"])
            validate_score_document(document, config, manifest, partition=partition, arm=arm)
            if arm in documents:
                raise ValueError(f"{partition}: repeated score document for {arm}")
            if not document.get("complete"):
                raise ValueError(f"{arm}/{partition}: the score document is incomplete; "
                                 "finish the partition before analysis")
            documents[arm] = document
        if not documents:
            summary["unscored_partitions"][partition] = (
                f"the manifest declares {len(units)} {partition} units and no score document exists")
            continue
        arms = sorted(documents)
        unlisted = sorted(set(arms) - set(roster))
        if unlisted:
            raise ValueError(f"{partition}: arms {unlisted} are outside the inherited roster")
        contributing = [unit for unit in units if unit["sites"]]
        unsupported = [unit["unit_id"] for unit in units if not unit["sites"]]
        declared_units = {unit["unit_id"] for unit in units}
        for arm in arms:
            extra = sorted(set(documents[arm]["units"]) - declared_units)
            if extra:
                raise RuntimeError(f"{arm}/{partition}: units {extra} are not in this partition")
        if not contributing:
            summary["partitions"][partition] = {
                "units": 0, "families_declared": len(units), "arms": arms,
                "hypothesis_family": len(column_labels(arms)),
                "unsupported_units": unsupported,
                "status": "descriptive_only",
                "reason": "no declared unit of this partition yields an eligible query position",
            }
            continue
        baselines = {unit["unit_id"]: _unit_logscores(unit) for unit in contributing}
        models = {arm: {} for arm in arms}
        for unit in contributing:
            for arm in arms:
                stored = documents[arm]["units"].get(unit["unit_id"])
                if stored is None:
                    raise ValueError(f"{arm}/{partition}: {unit['unit_id']} is unscored")
                _stored_unit(stored, unit, f"{arm}/{partition}/{unit['unit_id']}")
                models[arm][unit["unit_id"]] = _unit_logscores(unit, stored)["model"]
        labels = column_labels(arms)
        columns = {label: {} for label in labels}
        for unit in contributing:
            unit_id = unit["unit_id"]
            for arm in arms:
                model = models[arm][unit_id]
                columns[f"{arm}|minus_profile"][unit_id] = (
                    model - baselines[unit_id]["profile"])
                columns[f"{arm}|minus_markov"][unit_id] = (
                    model - baselines[unit_id]["markov"])
        groups = [unit["unit_id"] for unit in contributing]
        matrix = np.column_stack([[columns[label][group] for group in groups] for label in labels])
        missing = sorted(set(roster) - set(arms))
        record: dict[str, Any] = {
            "units": len(groups),
            "families_declared": len(units),
            "hypothesis_family": len(labels),
            "arms": arms,
            "roster": {"declared": len(roster), "scored": len(arms), "missing": missing},
            "unsupported_units": unsupported,
            "samples": {
                "families": len(groups),
                "sites": sum(len(unit["sites"]) for unit in contributing),
                "contexts": sum(len(site["contexts"]) for unit in contributing
                                for site in unit["sites"]),
                "sites_per_family": {unit["unit_id"]: len(unit["sites"]) for unit in contributing},
            },
            "columns": labels,
            "groups": groups,
            "absolute_logscore": {
                "model": {arm: float(np.mean([models[arm][group] for group in groups]))
                          for arm in arms},
                "profile": float(np.mean([baselines[group]["profile"] for group in groups])),
                "markov": float(np.mean([baselines[group]["markov"] for group in groups])),
                "uniform_aa": -math.log(len(AA20)),
                "note": "observed-residue log-scores in nats; the uniform value is a "
                        "description of the twenty-way enumeration, not a null hypothesis",
            },
        }
        floor = bootstrap_unit_floor(len(groups), minimum_units=statistics["min_inference_units"])
        record["unit_floor"] = floor
        reasons = []
        if missing:
            reasons.append(f"{len(missing)} of the declared {len(roster)} arms are unscored; a "
                           f"narrower family would be a different, undeclared one: {missing}")
        if unsupported:
            reasons.append(f"{len(unsupported)} declared units yield no eligible query position "
                           f"and are reported, not imputed: {unsupported}")
        if floor["degenerate"]:
            reasons.append(str(floor["degenerate_reason"]))
        if partition != "evaluation":
            reasons.append("the pilot partition is technical verification only")
        if reasons:
            record.update(status="descriptive_only", reason="; ".join(reasons),
                          descriptive_column_mean=dict(zip(labels, matrix.mean(axis=0).tolist())))
        else:
            record.update(status="complete",
                          inference=_family_inference(matrix, labels,
                                                      draws=int(statistics["bootstrap_draws"]),
                                                      seed=int(statistics["bootstrap_seed"])))
            record["degenerate_arms"] = record["inference"]["degenerate_arms"]
        summary["partitions"][partition] = record
        summary["units"][partition] = {
            unit["unit_id"]: {
                "wildtype_id": unit["wildtype_id"],
                "sites": len(unit["sites"]),
                "contexts": sum(len(site["contexts"]) for site in unit["sites"]),
                "profile": baselines[unit["unit_id"]]["profile"],
                "markov": baselines[unit["unit_id"]]["markov"],
                "observed": {arm: models[arm][unit["unit_id"]] for arm in arms},
                "contrasts": {label: columns[label][unit["unit_id"]] for label in labels},
            }
            for unit in contributing}
    summary["arms"] = sorted({arm for record in summary["partitions"].values()
                              for arm in record["arms"]})
    return summary