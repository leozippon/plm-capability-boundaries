#!/usr/bin/env python3
"""The downstream 2x2: does the *scorer* change how it ranks one generator's frozen pool?

The production E18 cohort freezes one 900-sequence pool per ProLLaMA stage and lets
either stage's checkpoint score either pool under the one unconditional rendering
family both share. That is two generating arms by two scoring arms. This module
reads the four likelihood cells and answers the only question the design can
answer without inventing a cross-model likelihood scale: **within one generator's
frozen pool, does swapping the scorer change the endpoint yield or the rank
association the likelihood buys?**

The pairing rule, stated once
=============================

A likelihood is a mean negative log-likelihood per scored token. It is comparable
inside one scoring checkpoint's own rendering, so the selector everywhere here is
the *negative* log-likelihood: **a higher likelihood is a smaller NLL, and the
selected set is the smallest-NLL rows**. Two scoring checkpoints produced nats per
token on different weights; their raw NLL means are therefore never subtracted,
and no cell in this artefact reports a cross-scorer nats-per-token difference.
What *is* paired is a quantity whose units are the endpoint's:

* the selected endpoint mean at a declared budget, under each scorer's ranking of
  the same candidates;
* the rank association between the scorer's likelihood and the endpoint, whole
  pool and inside each campaign stream.

The paired difference is always ``scorer_2 minus scorer_1`` on one of those
endpoint quantities. Adding a constant to every NLL, or multiplying every NLL by
a positive scale, leaves every rank and therefore every number here unchanged; the
analysis is rank-only by construction and the tests pin that.

Independence, and what this cannot bound
========================================

The production E18 census has 900 candidate identifiers per pool; counts here are
verified against each freeze's declaration, not assumed. The campaign streams are
the replication unit: the frozen seeds resample from fixed checkpoints and are not
900 independent lineages. This module reports one paired difference per campaign
stream, their mean and whether their sign replicated, and it **withholds a
population interval** while the number of streams is below the package's
eight-unit floor. Three streams cannot bound themselves, and the honest reading is
a descriptive comparison with limited precision, not a manufactured population
claim. Candidate identifiers are never treated as independent replicates.

The control ladder, and what "adjusted" means
=============================================

Length; length plus composition and complexity; then plus the nearest-corpus
identity and a family categorical. The designs are reused verbatim from
:mod:`src.capability.evaluation.likelihood_ranking`, built once from the same
rows for both scorers, so the two scorers are compared on identical matched
support. The residualized endpoint that the adjusted path selects on is a
*different quantity* from raw structural confidence: it is labelled a diagnostic
and its selected mean is never compared against a raw selected mean. A missing
structure, a missing novelty identity or a recognition row whose family key is
absent stops the run rather than silently shrinking the control ladder, because a
simpler control would answer a different question under the same name.

Structure, family and novelty are existing E18 artefacts:
``stage_pair_structure_index.jsonl`` (ESMFold2 mean CA pLDDT primary and pTM
secondary, per identifier and sequence digest), the Pfam oracle's
``family_recognition.jsonl`` and the homology search's ``generated_novelty.jsonl``.
Confidence is the predictor's uncertainty in its own coordinates; it is never
stability and never function, and the per-variant sequences are not carried into
this output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.capability.evaluation import prepare_stage_cross_scoring as pscs  # noqa: E402
from scripts.capability.evaluation import score_generated_likelihood as sjl  # noqa: E402
from scripts.capability.generation import summarise_stage_pair_structures as spst  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import bootstrap_unit_floor, mean_interval  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402
from src.capability.evaluation import likelihood_ranking as lr  # noqa: E402
from src.capability.generation import stage_pair as sp  # noqa: E402

COMPLETION = "stage_cross_scoring_analysis.json"
SCHEMA_VERSION = "d1_stage_cross_scoring_analysis_v1"

#: The structures, recognition and novelty files the E18 producers declare. The
#: shard tree behind ``stage_pair_structure_index.jsonl`` is never re-read: the
#: index already carries the per-identifier confidence this analysis joins on.
STRUCTURE_INDEX = "stage_pair_structure_index.jsonl"
STRUCTURE_COMPLETION = spst.COMPLETION
RECOGNITION_COMPLETION = "family_recognition.json"
RECOGNITION_ROWS = "family_recognition.jsonl"
RECOGNITION_CONTROL = "oracle_control.json"
NOVELTY_COMPLETION = "generated_novelty.json"
NOVELTY_ROWS = "generated_novelty.jsonl"

#: The scorer pair, in the declared stage order. The paired difference is always
#: ``stage_2 minus stage_1`` so its sign has one meaning.
SCORER_1 = sp.STAGES["stage_1"]
SCORER_2 = sp.STAGES["stage_2"]
SCORER_ARMS: tuple[str, ...] = (SCORER_1, SCORER_2)

#: The declared budget grid, minus the whole-pool point. At 1.00 every ranking
#: selects every candidate and the paired difference is zero by arithmetic rather
#: than by measurement, so it is not a comparison.
FRACTIONS: tuple[float, ...] = tuple(
    fraction for fraction in lr.SELECTION_FRACTIONS if fraction < 1.0
)

#: The one licensed generator set. The adapter refuses anything else, and this
#: analysis refuses a cohort whose pools are not exactly these.
LICENSED_POOLS: frozenset[str] = frozenset(sp.STAGES.values())

METHOD: dict[str, Any] = {
    "paired_quantity": (
        "every paired number is scorer_2 minus scorer_1 on an endpoint quantity -- "
        "the selected endpoint mean at a budget, or the rank association between the "
        "scorer's likelihood and the endpoint. Raw nats per token are never subtracted "
        "across scoring checkpoints"
    ),
    "selector_direction": (
        "a mean negative log-likelihood per scored token, so a smaller value is a more "
        "likely sequence. Ascending NLL is the selection direction everywhere, and the "
        "'high likelihood' association is the Spearman of -NLL against the endpoint, "
        "reported so that a positive value means a higher likelihood goes with a better "
        "endpoint"
    ),
    "rank_only": (
        "the selector is used only through its ordering. Adding a constant to every NLL "
        "or multiplying every NLL by a positive scale leaves every number here unchanged"
    ),
    "endpoint": (
        "ESMFold2 mean CA pLDDT (primary) and pTM (secondary), per identifier and "
        "sequence digest, taken from the E18 structure index. Both are the predictor's "
        "confidence in its own coordinates, not stability and not function"
    ),
    "control_ladder": [rung.statement for rung in lr.RUNGS],
    "adjusted_is_diagnostic": (
        "the adjusted path ranks the least-squares residual of the NLL on the declared "
        "design and evaluates the least-squares residual of the endpoint on the same "
        "design. That residual endpoint is a different quantity from raw structural "
        "confidence and the two selected means are never compared"
    ),
    "independence": (
        "the campaign stream is the replication unit. One paired difference is reported "
        "per stream with their mean and sign, and a population interval is withheld below "
        "the package's eight-unit floor. Candidate identifiers are not independent "
        "lineages"
    ),
}

LIMITATIONS: tuple[str, ...] = (
    "this is a descriptive comparison of two scorers on one frozen pool per generator, "
    "at the stream as the replication unit. Three streams cannot bound a population "
    "interval, so the paired differences are reported without one",
    "the endpoint is a predicted structural confidence. A scorer that selects higher "
    "pLDDT selects the predictor's easier sequences, not demonstrably better proteins",
    "the adjusted association is regression-adjusted, not exact matching and not causal. "
    "A rank-deficient or saturated design is reported unavailable rather than replaced "
    "by a simpler control",
    "nearest-corpus identity comes from one reference corpus at one gathering "
    "configuration; a low identity means no close relative there, not absence from any "
    "training set",
    "the identifiers are distinct attempts, but the frozen seeds resample from fixed "
    "checkpoints. Distinct identifiers are not distinct lineages",
)


# ------------------------------------------------------------------- readers


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: not a readable JSON artefact ({error})") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SystemExit(f"{path}: not a readable JSONL artefact ({error})") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{number}: {error}") from error
    if not rows:
        raise SystemExit(f"{path} carries no record; an empty input is refused")
    return rows


def _unique_by_id(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("id"))
        if not identifier or identifier == "None":
            raise SystemExit(f"{label}: a row carries no identifier")
        if identifier in values:
            raise SystemExit(f"{label}: identifier {identifier!r} appears twice")
        values[identifier] = dict(row)
    return values


# -------------------------------------------------------------------- inputs


def load_cohort(
    cohort_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """The prepared cohort manifest, its records by identifier and its pool blocks."""

    manifest = _read_json(Path(cohort_dir) / pscs.COMPLETION)
    if manifest.get("status") != "complete":
        raise SystemExit(
            f"{cohort_dir}: the cohort manifest status is {manifest.get('status')!r}, "
            "not 'complete'"
        )
    if manifest.get("schema_version") != pscs.SCHEMA_VERSION:
        raise SystemExit(
            f"{cohort_dir}: the cohort manifest schema is "
            f"{manifest.get('schema_version')!r}, not {pscs.SCHEMA_VERSION!r}"
        )
    cohort_path = Path(cohort_dir) / pscs.COHORT
    observed = sha256_file(cohort_path)
    if manifest.get("cohort_sha256") != observed:
        raise SystemExit(
            f"{cohort_path}: sha256 {observed} does not match the manifest's "
            f"{manifest.get('cohort_sha256')!r}; the cohort was edited after it was prepared"
        )
    rows = _read_jsonl(cohort_path)
    by_id = _unique_by_id(rows, "cohort")
    for identifier, row in by_id.items():
        if "pool" not in row.get("roles", []):
            raise SystemExit(f"cohort row {identifier!r} is not marked as a pool member")
        # A conditioning label would mean this pool was not drawn under the shared
        # unconditional interface; the stage-1 conditional counterpart does not exist.
        if row.get("class_key") not in (None, ""):
            raise SystemExit(
                f"cohort row {identifier!r} carries class_key {row['class_key']!r}; a "
                "conditioned pool has no unconditional stage-1 counterpart and is refused"
            )
        if row.get("condition") not in (None, "unconditioned"):
            raise SystemExit(
                f"cohort row {identifier!r} declares condition {row['condition']!r}; only "
                "the unconditional stage pair is analysed"
            )
        sequence = row.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            raise SystemExit(f"cohort row {identifier!r} carries no sequence")
        digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if row.get("sequence_sha256") != digest:
            raise SystemExit(
                f"cohort row {identifier!r} declares sequence_sha256 "
                f"{row.get('sequence_sha256')!r} and its sequence hashes to {digest!r}"
            )
        if int(row.get("length", -1)) != len(sequence):
            raise SystemExit(f"cohort row {identifier!r} declares a length that disagrees")
    if int(manifest.get("n_records", -1)) != len(rows):
        raise SystemExit(
            f"the manifest declares {manifest.get('n_records')} records and the cohort "
            f"carries {len(rows)}"
        )

    pools = manifest.get("pools")
    if not isinstance(pools, dict) or not pools:
        raise SystemExit("the cohort manifest declares no pool block")
    if set(pools) != set(LICENSED_POOLS):
        raise SystemExit(
            f"the cohort declares pools {sorted(pools)}, not the licensed stage pair "
            f"{sorted(LICENSED_POOLS)}"
        )
    for arm, block in sorted(pools.items()):
        members = sorted(
            (row for row in rows if str(row.get("arm")) == arm),
            key=lambda row: str(row["id"]),
        )
        if len(members) != int(block.get("n_members", -1)):
            raise SystemExit(
                f"pool {arm!r} declares {block.get('n_members')} members and the cohort "
                f"carries {len(members)}"
            )
        if gp.cohort_digest(members) != block.get("pool_digest"):
            raise SystemExit(
                f"pool {arm!r}: the recomputed pool digest does not match the manifest's "
                f"{block.get('pool_digest')!r}"
            )
        if block.get("candidate_key") != "id":
            raise SystemExit(
                f"pool {arm!r} is keyed on {block.get('candidate_key')!r}, not the "
                "identifier"
            )
        if block.get("independence_unit") != "stream":
            raise SystemExit(
                f"pool {arm!r} declares independence_unit {block.get('independence_unit')!r}; "
                "only the campaign stream is a replication unit, and reinterpreting "
                "candidate identifiers as independent lineages is refused"
            )
    # Bind the adaptation back to the exact producer freeze, not merely to a
    # self-consistent edited cohort and pool digest.
    provenance = manifest.get("provenance") or {}
    paths: dict[str, Path] = {}
    for role in ("freeze", "sequences"):
        source = provenance.get(role) or {}
        if not source.get("path") or not source.get("sha256"):
            raise SystemExit(f"the cohort lacks its {role} provenance receipt")
        paths[role] = Path(source["path"])
        if sha256_file(paths[role]) != source["sha256"]:
            raise SystemExit(f"the cohort {role} provenance sha256 disagrees with its source")
    freeze, _, arms = pscs.read_freeze(paths["freeze"], paths["sequences"])
    counts = pscs.declared_counts(freeze)
    frozen = pscs.read_pool_records(paths["sequences"], counts, arms)
    expected = _unique_by_id(
        [{**row, "roles": ["pool"], "condition": "unconditioned"} for row in frozen],
        "frozen adaptation",
    )
    if by_id != expected:
        raise SystemExit("the cohort does not preserve the exact declared frozen records")
    for stage, arm in arms.items():
        declared_streams = {stream: count for (s, stream), count in counts.items() if s == stage}
        block = pools[arm]
        if (
            block.get("stage") != stage
            or block.get("n_by_stream") != declared_streams
            or block.get("n_streams") != len(declared_streams)
        ):
            raise SystemExit(f"pool {arm!r} disagrees with the declared frozen stream counts")
    return manifest, by_id, pools


def _load_likelihood_cell(
    directory: Path,
    *,
    generator_arm: str,
    scorer_arm: str,
    pool_ids: Sequence[str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """One (generator, scorer) cell, checked against the cohort it claims to score."""

    receipt = _read_json(Path(directory) / sjl.COMPLETION)
    if receipt.get("schema_version") != sjl.SCHEMA_VERSION:
        raise SystemExit(
            f"{directory}: likelihood schema {receipt.get('schema_version')!r}, not "
            f"{sjl.SCHEMA_VERSION!r}"
        )
    if receipt.get("status") != "complete":
        raise SystemExit(f"{directory}: likelihood status is {receipt.get('status')!r}")
    bindings = {
        "arm": scorer_arm,
        "scorer_arm": scorer_arm,
        "pool_arm": generator_arm,
        "generator_arm": generator_arm,
    }
    for key, expected in bindings.items():
        if receipt.get(key) != expected:
            raise SystemExit(
                f"{directory}: the receipt declares {key}={receipt.get(key)!r} and this "
                f"cell is generator {generator_arm!r} scored by {scorer_arm!r}"
            )
    if bool(receipt.get("cross_arm")) != (generator_arm != scorer_arm):
        raise SystemExit(
            f"{directory}: cross_arm={receipt.get('cross_arm')!r} disagrees with the "
            "generator and scorer identities"
        )
    if receipt.get("cohort_sha256") != manifest.get("cohort_sha256"):
        raise SystemExit(
            f"{directory}: the receipt was produced against cohort sha256 "
            f"{receipt.get('cohort_sha256')!r}, not this cohort's "
            f"{manifest.get('cohort_sha256')!r}"
        )
    declared_digest = manifest["pools"][generator_arm]["pool_digest"]
    if receipt.get("pool_digest") != declared_digest:
        raise SystemExit(
            f"{directory}: the receipt carries pool digest {receipt.get('pool_digest')!r} "
            f"and the {generator_arm!r} pool declares {declared_digest!r}"
        )

    sidecar = Path(directory) / "generated_likelihood.jsonl"
    observed_sha = sha256_file(sidecar)
    if receipt.get("annotations_sha256") != observed_sha:
        raise SystemExit(
            f"{sidecar}: sha256 {observed_sha} does not match the receipt's "
            f"{receipt.get('annotations_sha256')!r}; the sidecar was edited after scoring"
        )
    rows = _read_jsonl(sidecar)
    by_id = _unique_by_id(rows, f"{sidecar}")
    wanted = set(str(identifier) for identifier in pool_ids)
    missing = sorted(wanted - set(by_id))
    extra = sorted(set(by_id) - wanted)
    if missing or extra:
        raise SystemExit(
            f"{sidecar}: does not cover the {generator_arm!r} frozen pool -- "
            f"{len(missing)} missing (first {missing[:3]}), {len(extra)} extra (first "
            f"{extra[:3]}). A cell that does not score the declared pool is refused"
        )
    for identifier, row in by_id.items():
        if row.get("arm") != generator_arm or row.get("pool_arm") != generator_arm:
            raise SystemExit(
                f"{sidecar}: row {identifier!r} names arm {row.get('arm')!r}/pool_arm "
                f"{row.get('pool_arm')!r}, not the generator {generator_arm!r}"
            )
        if row.get("scorer_arm") != scorer_arm:
            raise SystemExit(
                f"{sidecar}: row {identifier!r} names scorer {row.get('scorer_arm')!r}, "
                f"not {scorer_arm!r}"
            )
        try:
            value = float(row["mean_nll_per_token_nats"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(
                f"{sidecar}: row {identifier!r} carries no usable mean NLL ({error})"
            ) from error
        if not math.isfinite(value):
            raise SystemExit(f"{sidecar}: row {identifier!r} carries a non-finite NLL")
    return {
        "directory": str(directory),
        "receipt": receipt,
        "rows": by_id,
        "sidecar": str(sidecar),
        "sidecar_sha256": observed_sha,
    }


def load_likelihood_cells(
    directories: Sequence[Path],
    *,
    generator_arms: Sequence[str],
    scorer_arms: Sequence[str],
    manifest: dict[str, Any],
    cohort_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """The four (generator, scorer) cells, matched to the manifest's declared grid."""

    declared = {(str(cell["generator_arm"]), str(cell["scorer_arm"])) for cell in manifest["cells"]}
    grid = {(str(g), str(s)) for g in generator_arms for s in scorer_arms}
    if declared != grid:
        raise SystemExit(
            f"the cohort manifest declares cells {sorted(declared)}, not the full "
            f"generator-by-scorer grid {sorted(grid)}"
        )
    pool_ids = {
        arm: [
            identifier
            for identifier, row in cohort_by_id.items()
            if str(row.get("arm")) == arm
        ]
        for arm in generator_arms
    }
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for directory in directories:
        receipt = _read_json(Path(directory) / sjl.COMPLETION)
        key = (str(receipt.get("pool_arm")), str(receipt.get("scorer_arm")))
        if key not in declared:
            raise SystemExit(
                f"{directory}: the receipt names generator/scorer {key}, which the cohort "
                "manifest does not declare; an unbound likelihood cell is refused"
            )
        if key in cells:
            raise SystemExit(f"{directory}: a second likelihood cell for {key} is refused")
        cells[key] = _load_likelihood_cell(
            directory,
            generator_arm=key[0],
            scorer_arm=key[1],
            pool_ids=pool_ids[key[0]],
            manifest=manifest,
        )
    absent = sorted(declared - set(cells))
    if absent:
        raise SystemExit(
            f"the declared cells {absent} have no likelihood directory; a 2x2 with a "
            "missing cell is refused rather than reported one-sided"
        )
    return cells


def load_structures(
    directory: Path, cohort_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, float]], dict[str, Any], str]:
    """The E18 structure index joined to the cohort, at full coverage."""

    receipt = _read_json(Path(directory) / STRUCTURE_COMPLETION)
    if receipt.get("status") != "complete":
        raise SystemExit(f"{directory}: the structure summary status is {receipt.get('status')!r}")
    index = Path(directory) / STRUCTURE_INDEX
    observed_sha = sha256_file(index)
    if receipt.get("index_sha256") != observed_sha:
        raise SystemExit(
            f"{index}: sha256 {observed_sha} does not match the structure summary's "
            f"{receipt.get('index_sha256')!r}"
        )
    rows = _unique_by_id(_read_jsonl(index), "structure index")
    values: dict[str, dict[str, float]] = {}
    for identifier, row in rows.items():
        if identifier not in cohort_by_id:
            raise SystemExit(
                f"{index}: row {identifier!r} is not a member of the frozen cohort"
            )
        if row.get("sequence_sha256") != cohort_by_id[identifier].get("sequence_sha256"):
            raise SystemExit(
                f"{index}: row {identifier!r} describes a different sequence digest than "
                "the frozen cohort; the two artefacts are not the same set"
            )
        for field in (lr.PRIMARY_ENDPOINT, lr.SECONDARY_ENDPOINT):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise SystemExit(
                    f"{index}: row {identifier!r} carries no usable {field} ({error})"
                ) from error
            if not math.isfinite(value):
                raise SystemExit(f"{index}: row {identifier!r} carries a non-finite {field}")
        values[identifier] = {
            lr.PRIMARY_ENDPOINT: float(row[lr.PRIMARY_ENDPOINT]),
            lr.SECONDARY_ENDPOINT: float(row[lr.SECONDARY_ENDPOINT]),
        }
    missing = sorted(set(cohort_by_id) - set(values))
    if missing:
        raise SystemExit(
            f"{index}: {len(missing)} frozen members carry no structure (first "
            f"{missing[:3]}); the endpoint is incomplete and no adjusted result is "
            "reported on a silently smaller support"
        )
    return values, receipt, observed_sha


def load_recognition(
    directory: Path, cohort_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, list[str]], dict[str, Any], dict[str, Any], str]:
    """The Pfam oracle's family sets, at full coverage and with an explicit empty set."""

    receipt = _read_json(Path(directory) / RECOGNITION_COMPLETION)
    if receipt.get("status") != "complete":
        raise SystemExit(f"{directory}: the recognition status is {receipt.get('status')!r}")
    control = _read_json(Path(directory) / RECOGNITION_CONTROL)
    summary = receipt.get("oracle_control") or {}
    if control.get("passed") is not True or summary.get("passed") is not True:
        raise SystemExit(
            f"{directory}: the family oracle's control did not pass, so its recognition set "
            "is not evidence and no family control is computed"
        )
    # The producer emits this summary, not a digest of the full control artefact.
    # Agreement binds only these fields; it cannot authenticate all control details.
    for field in ("passed", "n_positive", "n_negative"):
        if field not in control or field not in summary or control[field] != summary[field]:
            raise SystemExit(f"{directory}: oracle_control summary disagrees on {field}")
    rows_path = Path(directory) / RECOGNITION_ROWS
    observed_sha = sha256_file(rows_path)
    if receipt.get("annotations_sha256") != observed_sha:
        raise SystemExit(
            f"{rows_path}: sha256 {observed_sha} does not match the recognition receipt's "
            f"{receipt.get('annotations_sha256')!r}"
        )
    rows = _unique_by_id(_read_jsonl(rows_path), "recognition")
    values: dict[str, list[str]] = {}
    for identifier, row in rows.items():
        if identifier not in cohort_by_id:
            raise SystemExit(f"{rows_path}: row {identifier!r} is not a frozen cohort member")
        for field in ("sequence_sha256", "arm"):
            if row.get(field) != cohort_by_id[identifier].get(field):
                raise SystemExit(f"{rows_path}: row {identifier!r} has a different {field} than the cohort")
        if row.get("profile_search_status") != "searched":
            raise SystemExit(f"{rows_path}: row {identifier!r} profile_search_status is not 'searched'")
        if "pfam_families" not in row:
            raise SystemExit(
                f"{rows_path}: row {identifier!r} carries no pfam_families key; a missing "
                "key is not the empty family set and is refused"
            )
        families = row["pfam_families"]
        if not isinstance(families, list):
            raise SystemExit(
                f"{rows_path}: row {identifier!r} carries pfam_families of type "
                f"{type(families).__name__}, not a list"
            )
        values[identifier] = [str(family) for family in families]
    missing = sorted(set(cohort_by_id) - set(values))
    if missing:
        raise SystemExit(
            f"{rows_path}: {len(missing)} frozen members carry no recognition row (first "
            f"{missing[:3]}); the family control is incomplete and is refused rather than "
            "run without its family term"
        )
    return values, receipt, control, observed_sha


def load_novelty(
    directory: Path, cohort_by_id: Mapping[str, Mapping[str, Any]], *, cohort_sha256: str
) -> tuple[dict[str, float], dict[str, Any], str]:
    """The homology search's nearest-corpus identity, at full coverage."""

    receipt = _read_json(Path(directory) / NOVELTY_COMPLETION)
    if receipt.get("status") != "complete":
        raise SystemExit(f"{directory}: the novelty status is {receipt.get('status')!r}")
    if receipt.get("cohort_sha256") != cohort_sha256:
        raise SystemExit(f"{directory}: novelty cohort sha256 does not match the prepared cohort")
    rows_path = Path(directory) / NOVELTY_ROWS
    observed_sha = sha256_file(rows_path)
    if receipt.get("annotations_sha256") != observed_sha:
        raise SystemExit(
            f"{rows_path}: sha256 {observed_sha} does not match the novelty receipt's "
            f"{receipt.get('annotations_sha256')!r}"
        )
    rows = _unique_by_id(_read_jsonl(rows_path), "novelty")
    values: dict[str, float] = {}
    for identifier, row in rows.items():
        if identifier not in cohort_by_id:
            raise SystemExit(f"{rows_path}: row {identifier!r} is not a frozen cohort member")
        if row.get("searched") is not True:
            raise SystemExit(f"{rows_path}: row {identifier!r} was not searched")
        if row.get("arm") != cohort_by_id[identifier].get("arm"):
            raise SystemExit(f"{rows_path}: row {identifier!r} has a different arm than the cohort")
        raw = row.get("nearest_corpus_identity")
        if raw is None:
            raise SystemExit(
                f"{rows_path}: row {identifier!r} carries no nearest_corpus_identity; a "
                "missing identity is not an identity of zero"
            )
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"{rows_path}: row {identifier!r} carries an unusable identity") from error
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise SystemExit(f"{rows_path}: row {identifier!r} identity must be finite and in 0..100")
        values[identifier] = value
    missing = sorted(set(cohort_by_id) - set(values))
    if missing:
        raise SystemExit(
            f"{rows_path}: {len(missing)} frozen members carry no novelty row (first "
            f"{missing[:3]}); the identity control is incomplete and is refused rather than "
            "dropped from the design"
        )
    return values, receipt, observed_sha


# ------------------------------------------------------------- pool diagnostics


def _duplicate_diagnostics(
    ids: Sequence[str], sequences: Sequence[str], streams: Sequence[str]
) -> dict[str, Any]:
    """Exact candidate duplicates, and how many span more than one stream."""

    grouped: dict[str, list[tuple[str, str]]] = {}
    for identifier, sequence, stream in zip(ids, sequences, streams):
        grouped.setdefault(sequence, []).append((identifier, stream))
    duplicate_groups = {sequence: entries for sequence, entries in grouped.items() if len(entries) > 1}
    spanning = {
        sequence: entries
        for sequence, entries in duplicate_groups.items()
        if len({stream for _, stream in entries}) > 1
    }
    examples = []
    for sequence in sorted(duplicate_groups, key=lambda value: (-len(duplicate_groups[value]), value))[:20]:
        entries = duplicate_groups[sequence]
        examples.append(
            {
                "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                "count": len(entries),
                "n_streams": len({stream for _, stream in entries}),
                "ids": [identifier for identifier, _ in entries][:5],
            }
        )
    return {
        "n_candidates": len(ids),
        "n_distinct_sequences": len(grouped),
        "n_exact_duplicate_rows": len(ids) - len(grouped),
        "n_duplicate_groups": len(duplicate_groups),
        "n_groups_spanning_streams": len(spanning),
        "n_rows_in_groups_spanning_streams": sum(len(entries) for entries in spanning.values()),
        "examples": examples,
        "statement": (
            "exact sequence duplicates are reported rather than collapsed: two identical "
            "generated products are not two independent candidates, and a duplicate that "
            "spans streams is a cross-stream dependency"
        ),
    }


def _family_dependency(families: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Family repertoire and the concentration a family control has to absorb."""

    repertoire = gp.family_repertoire([list(block) for block in families])
    counts: Counter[str] = Counter()
    for block in families:
        for family in set(block):
            counts[family] += 1
    return {
        **repertoire,
        "n_unrecognised": int(sum(1 for block in families if not block)),
        "n_families_in_multiple_sequences": int(sum(1 for value in counts.values() if value > 1)),
        "max_sequences_per_family": int(max(counts.values())) if counts else 0,
        "statement": (
            "family sets are recognised Pfam assignments. An empty set is a valid "
            "'nothing recognised' and is not a missing row; a family shared by many "
            "candidates is a dependency the family categorical must absorb"
        ),
    }


def _selected_set_diagnostics(
    ids: Sequence[str],
    sequences: Sequence[str],
    families: Sequence[Sequence[str]],
    identities: Sequence[float],
    nll_vectors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Diversity, nearest identity, length and composition of each selected set."""

    diagnostics: dict[str, Any] = {}
    for scorer, nll in nll_vectors.items():
        block: dict[str, Any] = {}
        for fraction in FRACTIONS:
            selected = int(gp.selection_count(len(ids), fraction))
            order = [
                int(position)
                for position in np.argsort(np.asarray(nll, dtype=np.float64), kind="stable")[
                    :selected
                ].tolist()
            ]
            chosen_sequences: list[str] = [sequences[position] for position in order]
            chosen_families: list[list[str]] = [
                [str(family) for family in families[position]] for position in order
            ]
            profile = gp.selected_set_profile(
                chosen_sequences,
                chosen_families,
                corpus_identity=[identities[position] for position in order],
            )
            profile["fraction"] = float(fraction)
            profile["n_selected"] = selected
            block[f"{fraction:.2f}"] = profile
        diagnostics[scorer] = block
    return diagnostics


# ---------------------------------------------------------------- the analysis


def _high_likelihood(block: Mapping[str, Any]) -> float | None:
    """The high-likelihood association, read from the helper without re-orienting it.

    ``likelihood_ranking`` reports every correlation in the settled high-likelihood
    orientation (``-NLL`` against the endpoint), so a positive value means a higher
    likelihood goes with a better endpoint and a negative value means it goes with a
    worse one. The value is passed through unchanged: applying another sign here would
    double-negate the declared convention.
    """

    value = block.get("spearman")
    return None if value is None else float(value)


def _association(nll: np.ndarray, endpoint: np.ndarray) -> dict[str, Any]:
    block = lr.whole_pool_spearman(nll, endpoint)
    block["high_likelihood_association"] = _high_likelihood(block)
    return block


def _stream_summary(values: Mapping[str, float | None]) -> dict[str, Any]:
    """Per-stream values, their mean and sign, and an interval only above the floor."""

    numbers = [float(value) for value in values.values() if value is not None]
    floor = bootstrap_unit_floor(len(numbers))
    summary: dict[str, Any] = {
        "per_stream": {key: (None if value is None else float(value)) for key, value in sorted(values.items())},
        "n_streams": len(numbers),
        "unit": "campaign stream",
        "mean_difference": float(np.mean(numbers)) if numbers else None,
        "direction_replicated": (
            bool(all(value > 0 for value in numbers) or all(value < 0 for value in numbers))
            if len(numbers) >= 2
            else None
        ),
        "unit_floor": floor,
        "interval": None,
        "interval_reason": floor["degenerate_reason"],
        "note": (
            "the campaign stream is the replication unit. A population interval is "
            "withheld below the eight-unit floor, and the candidate identifiers are "
            "never resampled as if they were independent lineages"
        ),
    }
    if not floor["degenerate"] and len(numbers) >= 2:
        summary["interval"] = mean_interval(numbers)["interval"]
        summary["interval_reason"] = None
    return summary


def _ladder_description(designs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        rung.name: {
            "level": rung.level,
            "statement": rung.statement,
            "status": designs[rung.name]["status"],
            "reason": designs[rung.name].get("reason"),
            "columns": designs[rung.name].get("column_names"),
            "n_support": designs[rung.name].get("diagnostics", {}).get("n_rows", 0),
            "design": designs[rung.name].get("diagnostics"),
        }
        for rung in lr.RUNGS
    }


def _build_designs(
    features: Sequence[lr.RowFeatures],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], bool]:
    vocabulary = lr.family_vocabulary(list(features))
    family_varies = lr.has_family_variation(vocabulary)
    designs: dict[str, dict[str, Any]] = {}
    for rung in lr.RUNGS:
        if rung.uses_family and not family_varies:
            designs[rung.name] = {
                "status": "unavailable",
                "reason": "family_categorical_has_no_variation",
            }
            continue
        matrix, names = lr.build_design(list(features), rung, vocabulary)
        diagnostics = lr.design_diagnostics(matrix)
        reason = None
        if diagnostics["rank"] < diagnostics["n_columns"]:
            reason = "rank_deficient"
        elif diagnostics["residual_df"] < lr.MINIMUM_RESIDUAL_DF:
            reason = "saturated"
        designs[rung.name] = {
            "status": "measured" if reason is None else "unavailable",
            "reason": reason,
            "matrix": matrix,
            "column_names": list(names),
            "diagnostics": diagnostics,
        }
    return designs, vocabulary, family_varies


def _scorer_endpoint_block(
    nll: np.ndarray,
    endpoint: np.ndarray,
    streams: Sequence[str],
    designs: Mapping[str, Mapping[str, Any]],
    scorer: str,
) -> dict[str, Any]:
    selection: dict[str, Any] = {}
    for fraction in FRACTIONS:
        block = lr.raw_selection_block(nll, endpoint, fraction=fraction)
        block["fraction"] = float(fraction)
        selection[f"{fraction:.2f}"] = block

    per_stream: dict[str, Any] = {}
    labels = np.asarray([str(stream) for stream in streams])
    for stream in sorted(set(str(stream) for stream in streams)):
        mask = labels == stream
        block = _association(nll[mask], endpoint[mask])
        block["n_rows"] = int(mask.sum())
        per_stream[stream] = block

    ladder: dict[str, Any] = {}
    for rung in lr.RUNGS:
        design = designs[rung.name]
        entry: dict[str, Any] = {
            "statement": rung.statement,
            "status": design["status"],
            "reason": design.get("reason"),
            "column_names": design.get("column_names"),
            "n_support": design.get("diagnostics", {}).get("n_rows", 0),
            "design": design.get("diagnostics"),
            "partial_spearman": None,
            "high_likelihood_association": None,
            "adjusted_selection": {},
        }
        if design["status"] == "measured":
            partial = lr.partial_spearman(nll, endpoint, design["matrix"], units=streams)
            entry["partial_spearman"] = partial
            entry["high_likelihood_association"] = _high_likelihood(partial)
            for fraction in FRACTIONS:
                diagnostic = lr.selection_diagnostics(
                    nll,
                    endpoint,
                    design["matrix"],
                    fraction=fraction,
                    rung_name=rung.name,
                )
                entry["adjusted_selection"][f"{fraction:.2f}"] = {
                    "status": diagnostic["status"],
                    "reason": diagnostic.get("reason"),
                    "residual_endpoint_selection": diagnostic.get("adjusted"),
                    "label": (
                        "regression_adjusted_residual_endpoint_diagnostic_not_raw_confidence"
                    ),
                }
        ladder[rung.name] = entry

    return {
        "scorer_arm": scorer,
        "nll_descriptive": {
            "min": float(np.min(nll)),
            "max": float(np.max(nll)),
            "mean": float(np.mean(nll)),
        },
        "nll_cross_scorer_comparison": (
            "none; the two scoring checkpoints assign nats per token on different weights "
            "and their raw NLL levels are never subtracted"
        ),
        "whole_pool": _association(nll, endpoint),
        "per_stream": per_stream,
        "selection": selection,
        "control_ladder": ladder,
    }


def _paired_block(
    nll_vectors: Mapping[str, np.ndarray],
    endpoint: np.ndarray,
    streams: Sequence[str],
    score_blocks: Mapping[str, Mapping[str, Any]],
    designs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    def difference(left: Any, right: Any) -> float | None:
        if left is None or right is None:
            return None
        return float(right - left)

    whole = difference(
        score_blocks[SCORER_1]["whole_pool"]["high_likelihood_association"],
        score_blocks[SCORER_2]["whole_pool"]["high_likelihood_association"],
    )

    labels = np.asarray([str(stream) for stream in streams])
    per_stream: dict[str, float | None] = {}
    for stream in sorted(set(str(stream) for stream in streams)):
        per_stream[stream] = difference(
            score_blocks[SCORER_1]["per_stream"][stream]["high_likelihood_association"],
            score_blocks[SCORER_2]["per_stream"][stream]["high_likelihood_association"],
        )

    selection: dict[str, Any] = {}
    for fraction in FRACTIONS:
        key = f"{fraction:.2f}"
        first = score_blocks[SCORER_1]["selection"][key]
        second = score_blocks[SCORER_2]["selection"][key]
        stream_differences: dict[str, float] = {}
        for stream in sorted(set(str(stream) for stream in streams)):
            mask = labels == stream
            left = lr.raw_selection_block(nll_vectors[SCORER_1][mask], endpoint[mask], fraction=fraction)
            right = lr.raw_selection_block(nll_vectors[SCORER_2][mask], endpoint[mask], fraction=fraction)
            stream_differences[stream] = float(right["selected_mean"] - left["selected_mean"])
        selection[key] = {
            "fraction": float(fraction),
            "n_selected": int(first["n_selected"]),
            "selected_mean": {
                SCORER_1: float(first["selected_mean"]),
                SCORER_2: float(second["selected_mean"]),
            },
            "selected_mean_difference": float(second["selected_mean"] - first["selected_mean"]),
            "gain_over_random_difference": float(
                second["gain_over_random"] - first["gain_over_random"]
            ),
            "oracle_gap_difference": float(second["oracle_gap"] - first["oracle_gap"]),
            "per_stream_selected_mean_difference": stream_differences,
            "stream_summary": _stream_summary(stream_differences),
        }

    ladder: dict[str, Any] = {}
    for rung in lr.RUNGS:
        left = score_blocks[SCORER_1]["control_ladder"][rung.name]["high_likelihood_association"]
        right = score_blocks[SCORER_2]["control_ladder"][rung.name]["high_likelihood_association"]
        ladder[rung.name] = {
            "scorer_1": left,
            "scorer_2": right,
            "high_likelihood_association_difference": difference(left, right),
        }

    return {
        "convention": f"{SCORER_2} minus {SCORER_1}",
        "scorer_1": SCORER_1,
        "scorer_2": SCORER_2,
        "whole_pool_high_likelihood_association_difference": whole,
        "per_stream_high_likelihood_association_difference": per_stream,
        "stream_summary": _stream_summary(per_stream),
        "selection": selection,
        "control_ladder_high_likelihood_association_difference": ladder,
        "note": (
            "a positive difference means the second scorer's ranking yields a better "
            "endpoint or a stronger high-likelihood association. No raw NLL level is "
            "compared between the two scorers"
        ),
    }


def analyse_pool(
    generator_arm: str,
    scorer_arms: Sequence[str],
    ids: Sequence[str],
    streams: Sequence[str],
    nll: Mapping[str, np.ndarray | Sequence[float]],
    features: Sequence[lr.RowFeatures],
    sequences: Sequence[str],
    families: Sequence[Sequence[str]],
    identities: Sequence[float],
    endpoints: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """The 2x2 comparison inside one generator's frozen pool, endpoint by endpoint."""

    ids = [str(value) for value in ids]
    stream_labels = [str(value) for value in streams]
    scorer_arms = [str(value) for value in scorer_arms]
    if set(scorer_arms) != set(SCORER_ARMS):
        raise SystemExit(
            f"the analysis compares {sorted(SCORER_ARMS)} and was given {sorted(scorer_arms)}"
        )
    n_rows = len(ids)
    if n_rows < 2:
        raise SystemExit("a pool of fewer than two candidates has no rank association")
    if not (len(stream_labels) == len(features) == len(sequences) == len(families) == len(identities) == n_rows):
        raise SystemExit("the pool inputs do not align on the candidate identifiers")

    nll_vectors = {scorer: np.asarray(nll[scorer], dtype=np.float64) for scorer in scorer_arms}
    for scorer, vector in nll_vectors.items():
        if vector.shape != (n_rows,) or not np.isfinite(vector).all():
            raise SystemExit(f"{scorer}: the NLL vector does not align or is non-finite")
    endpoint_vectors = {
        endpoint: np.asarray(endpoints[endpoint], dtype=np.float64) for endpoint in lr.ENDPOINTS
    }
    for endpoint, vector in endpoint_vectors.items():
        if vector.shape != (n_rows,) or not np.isfinite(vector).all():
            raise SystemExit(f"{endpoint}: the endpoint vector does not align or is non-finite")

    designs, vocabulary, family_varies = _build_designs(features)
    selected_diagnostics = _selected_set_diagnostics(ids, sequences, families, identities, nll_vectors)

    endpoint_blocks: dict[str, Any] = {}
    for endpoint in lr.ENDPOINTS:
        vector = endpoint_vectors[endpoint]
        score_blocks = {
            scorer: _scorer_endpoint_block(nll_vectors[scorer], vector, stream_labels, designs, scorer)
            for scorer in scorer_arms
        }
        endpoint_blocks[endpoint] = {
            "quantity": lr.ENDPOINT_QUANTITY[endpoint],
            "orientation": "larger is better",
            "n_rows": n_rows,
            "scorers": score_blocks,
            "paired": _paired_block(nll_vectors, vector, stream_labels, score_blocks, designs),
        }

    n_streams = len(set(stream_labels))
    independence = {
        "unit": "stream",
        "n_streams": n_streams,
        "n_rows": n_rows,
        "unit_floor": bootstrap_unit_floor(n_streams),
        "statement": (
            "the campaign stream is the replication unit. The candidate identifiers are "
            "distinct attempts, not distinct lineages, and are never resampled as if they "
            "were independent"
        ),
    }

    return {
        "generator_arm": generator_arm,
        "scorer_arms": sorted(scorer_arms),
        "n_candidates": n_rows,
        "n_streams": n_streams,
        "candidate_key": "id",
        "independence": independence,
        "family_vocabulary": list(vocabulary),
        "family_vocabulary_size": len(vocabulary),
        "family_categorical_has_variation": bool(family_varies),
        "duplicates": _duplicate_diagnostics(ids, sequences, stream_labels),
        "family_dependency": _family_dependency(families),
        "selected_set_diagnostics": selected_diagnostics,
        "control_ladder": _ladder_description(designs),
        "endpoints": endpoint_blocks,
    }


# ------------------------------------------------------------------------ run


def _source_hashes(
    cohort_dir: Path,
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
    structure_dir: Path,
    structure_sha: str,
    recognition_dir: Path,
    recognition_sha: str,
    novelty_dir: Path,
    novelty_sha: str,
) -> list[dict[str, str]]:
    """Every input file, with the hash of its actual bytes."""

    structure_dir = Path(structure_dir)
    recognition_dir = Path(recognition_dir)
    novelty_dir = Path(novelty_dir)
    entries: list[dict[str, str]] = [
        {"role": "cohort_manifest", "path": str(Path(cohort_dir) / pscs.COMPLETION), "sha256": sha256_file(Path(cohort_dir) / pscs.COMPLETION)},
        {"role": "cohort_rows", "path": str(Path(cohort_dir) / pscs.COHORT), "sha256": sha256_file(Path(cohort_dir) / pscs.COHORT)},
        {"role": "structure_index", "path": str(structure_dir / STRUCTURE_INDEX), "sha256": structure_sha},
        {"role": "structure_receipt", "path": str(structure_dir / STRUCTURE_COMPLETION), "sha256": sha256_file(structure_dir / STRUCTURE_COMPLETION)},
        {"role": "recognition_rows", "path": str(recognition_dir / RECOGNITION_ROWS), "sha256": recognition_sha},
        {"role": "recognition_receipt", "path": str(recognition_dir / RECOGNITION_COMPLETION), "sha256": sha256_file(recognition_dir / RECOGNITION_COMPLETION)},
        {"role": "recognition_control", "path": str(recognition_dir / RECOGNITION_CONTROL), "sha256": sha256_file(recognition_dir / RECOGNITION_CONTROL)},
        {"role": "novelty_rows", "path": str(novelty_dir / NOVELTY_ROWS), "sha256": novelty_sha},
        {"role": "novelty_receipt", "path": str(novelty_dir / NOVELTY_COMPLETION), "sha256": sha256_file(novelty_dir / NOVELTY_COMPLETION)},
    ]
    for (generator, scorer), cell in sorted(cells.items()):
        entries.append(
            {
                "role": f"likelihood_{generator}_{scorer}",
                "path": str(Path(cell["directory"]) / sjl.COMPLETION),
                "sha256": sha256_file(Path(cell["directory"]) / sjl.COMPLETION),
            }
        )
        entries.append(
            {
                "role": f"likelihood_rows_{generator}_{scorer}",
                "path": str(cell["sidecar"]),
                "sha256": cell["sidecar_sha256"],
            }
        )
    return entries


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    manifest, cohort_by_id, pools = load_cohort(args.cohort_dir)
    generator_arms = sorted(pools)
    cells = load_likelihood_cells(
        args.likelihood,
        generator_arms=generator_arms,
        scorer_arms=SCORER_ARMS,
        manifest=manifest,
        cohort_by_id=cohort_by_id,
    )
    structures, structure_receipt, structure_sha = load_structures(args.structures, cohort_by_id)
    recognition, recognition_receipt, control, recognition_sha = load_recognition(
        args.recognition, cohort_by_id
    )
    novelty, novelty_receipt, novelty_sha = load_novelty(
        args.novelty, cohort_by_id, cohort_sha256=manifest["cohort_sha256"]
    )

    pool_blocks: dict[str, Any] = {}
    for generator in generator_arms:
        members = sorted(
            (row for row in cohort_by_id.values() if str(row.get("arm")) == generator),
            key=lambda row: str(row["id"]),
        )
        identifiers = [str(row["id"]) for row in members]
        streams = [str(row["stream"]) for row in members]
        nll = {
            scorer: np.asarray(
                [
                    float(cells[(generator, scorer)]["rows"][identifier]["mean_nll_per_token_nats"])
                    for identifier in identifiers
                ],
                dtype=np.float64,
            )
            for scorer in SCORER_ARMS
        }
        features = [
            lr.make_row_features(
                str(row["sequence"]),
                int(row["length"]),
                nearest_corpus_identity=novelty[str(row["id"])],
                families=recognition[str(row["id"])],
            )
            for row in members
        ]
        analysis = analyse_pool(
            generator,
            SCORER_ARMS,
            identifiers,
            streams,
            nll,
            features,
            [str(row["sequence"]) for row in members],
            [recognition[str(row["id"])] for row in members],
            [novelty[str(row["id"])] for row in members],
            {
                lr.PRIMARY_ENDPOINT: [structures[identifier][lr.PRIMARY_ENDPOINT] for identifier in identifiers],
                lr.SECONDARY_ENDPOINT: [structures[identifier][lr.SECONDARY_ENDPOINT] for identifier in identifiers],
            },
        )
        analysis["n_declared_candidates"] = pools[generator]["n_members"]
        analysis["pool_digest"] = pools[generator]["pool_digest"]
        analysis["manifest_pool"] = pools[generator]
        pool_blocks[generator] = analysis

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": (
            "within one generator's frozen pool, does swapping the scoring checkpoint "
            "change the endpoint yield or the rank association the likelihood buys?"
        ),
        "convention": {
            "generator_arms": generator_arms,
            "scorer_arms": list(SCORER_ARMS),
            "paired_difference": f"{SCORER_2} minus {SCORER_1}",
            "selector": "ascending mean_nll_per_token_nats, i.e. descending likelihood",
            "high_likelihood_association": "Spearman of -NLL against the endpoint",
            "fractions": list(FRACTIONS),
            "primary_endpoint": lr.PRIMARY_ENDPOINT,
            "secondary_endpoint": lr.SECONDARY_ENDPOINT,
        },
        "sources": {
            "cohort_dir": str(args.cohort_dir),
            "likelihood": [str(directory) for directory in args.likelihood],
            "structures": str(args.structures),
            "recognition": str(args.recognition),
            "novelty": str(args.novelty),
        },
        "source_hashes": _source_hashes(
            args.cohort_dir,
            cells,
            args.structures,
            structure_sha,
            args.recognition,
            recognition_sha,
            args.novelty,
            novelty_sha,
        ),
        "cohort": {
            "path": str(Path(args.cohort_dir) / pscs.COHORT),
            "manifest": str(Path(args.cohort_dir) / pscs.COMPLETION),
            "cohort_sha256": manifest.get("cohort_sha256"),
            "n_records": len(cohort_by_id),
            "n_declared_records": manifest["n_records"],
            "n_by_stage": manifest.get("n_by_stage"),
            "generator_arms": manifest.get("generator_arms"),
            "cells": manifest.get("cells"),
        },
        "inputs": {
            "structure_receipt": {
                "path": str(Path(args.structures) / STRUCTURE_COMPLETION),
                "status": structure_receipt.get("status"),
                "index_sha256": structure_receipt.get("index_sha256"),
                "per_stage_counts": structure_receipt.get("per_stage_counts"),
            },
            "recognition_receipt": {
                "path": str(Path(args.recognition) / RECOGNITION_COMPLETION),
                "status": recognition_receipt.get("status"),
                "annotations_sha256": recognition_receipt.get("annotations_sha256"),
                "oracle_control_passed": control["passed"],
                "oracle_control_binding": "producer summary only: passed, n_positive, n_negative; no full control digest is emitted",
            },
            "novelty_receipt": {
                "path": str(Path(args.novelty) / NOVELTY_COMPLETION),
                "status": novelty_receipt.get("status"),
                "annotations_sha256": novelty_receipt.get("annotations_sha256"),
                "cohort_sha256": novelty_receipt["cohort_sha256"],
                "identity_strata": novelty_receipt.get("identity_strata"),
            },
        },
        "method": METHOD,
        "limitations": list(LIMITATIONS),
        "endpoints": list(lr.ENDPOINTS),
        "fractions": list(FRACTIONS),
        "control_ladder": [rung.statement for rung in lr.RUNGS],
        "pools": pool_blocks,
        "reading_guide": [
            "the selector is a smaller mean NLL, so the selected set is the smallest-NLL "
            "rows; a higher-likelihood association is reported with a positive sign",
            "the paired difference is scorer_2 minus scorer_1 on an endpoint quantity. A "
            "positive value means the second scorer's ranking buys a better endpoint",
            "raw nats per token are never subtracted across the two scoring checkpoints",
            "the campaign stream is the replication unit and three streams cannot bound a "
            "population interval, so the paired differences are descriptive",
            "the adjusted path selects on a residualized endpoint, a different quantity "
            "from raw structural confidence; the two selected means are never compared",
            "pLDDT and pTM are the structure predictor's confidence in its own "
            "coordinates, not stability and not function",
        ],
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        required=True,
        help="the prepared stage-cross-scoring cohort directory",
    )
    parser.add_argument(
        "--likelihood",
        type=Path,
        nargs="+",
        required=True,
        help="the four generator-by-scorer likelihood output directories",
    )
    parser.add_argument(
        "--structures",
        type=Path,
        required=True,
        help="the E18 stage_pair_structures output directory holding the index and summary",
    )
    parser.add_argument(
        "--recognition",
        type=Path,
        required=True,
        help="the Pfam oracle output directory for the frozen set",
    )
    parser.add_argument(
        "--novelty",
        type=Path,
        required=True,
        help="the homology-search output directory for the frozen set",
    )
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(
        json.dumps(
            {
                "status": record["status"],
                "pools": sorted(record["pools"]),
                "endpoints": record["endpoints"],
                "fractions": record["fractions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()