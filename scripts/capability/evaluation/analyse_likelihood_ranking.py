#!/usr/bin/env python3
"""Bounded E17 diagnostics: does the generating model's likelihood rank candidate quality?

This stage reads only frozen artefacts -- the E17 cohort and its declaration, the
ESMFold2 shard indexes, the Pfam oracle's annotations and passed control, the
recomputed likelihoods, and the novelty search -- and computes the ranking
diagnostics in :mod:`src.capability.evaluation.likelihood_ranking`. It loads no
model, folds no structure, recognises no family and searches no corpus. It accepts
``--device`` because the campaign queue injects it.

The endpoints are mean CA pLDDT (primary) and pTM (secondary). Both are ESMFold2's
confidence in its own coordinates, not stability and not function. The likelihood
is a mean negative log-likelihood per scored token, so lower is more likely.

Three things are reported for each endpoint:

* the descriptive whole-pool Spearman between likelihood and endpoint;
* the descriptive Spearman inside the upper-likelihood tail at E17's own
  fractions, with the selected-set size, the independence units it spans and the
  reason it is undefined when it is;
* a regression-adjusted association from a nested control ladder -- length;
  length plus composition, entropy and the longest homopolymer; then plus the
  nearest-corpus identity and a family categorical -- with the design's rank and
  residual degrees of freedom. Controls are refit inside every cluster bootstrap
  replicate, the adjusted and unadjusted estimates are paired on the same
  resampled rows, and a pool below the unit floor or a design that is
  rank-deficient or saturated is unavailable with its reason.

At every budget the raw selected endpoint mean is reported against random
selection's expectation (the pool mean) and against the empirical oracle bound,
and beside them the adjusted-score selection in which the pool is ranked by the
residualized likelihood and evaluated by the residualized endpoint. The adjusted
endpoint is a different quantity from the raw endpoint and the two means are
never differenced.

Every input is hashed and every exclusion counted; a duplicate identifier, a
likelihood that is missing for a pool member, a novelty row that is missing, or a
pool digest that does not match the declaration stops the run before any artefact
is written. The whole record is assembled in memory and written once, atomically
and with non-finite values rejected, so a partial success never reaches disk.

A rank-deficient control design is refused by default: that is the frozen E17
full-column-rank contract, and every refusal it produced stays in the frozen
artefact. ``--column-span-reanalysis`` runs a separately labelled post-hoc
reanalysis in which a rank-deficient design is admitted, because the least-squares
projection onto a design's column span is unique whether or not the columns that
span it are independent. Saturation, a constant residual, a constant score and the
unit floors still refuse, and each rung reports the columns its support leaves
identically zero and how many resampled draws were rank-deficient. The reanalysis
writes its own label, rank policy and code hashes into a record in its own output
directory, never into the frozen one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import bootstrap_unit_floor  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402
from src.capability.evaluation import likelihood_ranking as lr  # noqa: E402

COMPLETION = "likelihood_ranking.json"

#: The two sources that produce a record, hashed into it so a label binds to code.
CODE_HASH_SOURCES: tuple[tuple[str, str], ...] = (
    ("ranking_module", "src/capability/evaluation/likelihood_ranking.py"),
    ("analysis_stage", "scripts/capability/evaluation/analyse_likelihood_ranking.py"),
)

#: The fewest draws at which a nominal 2.5% tail endpoint is an order statistic of
#: at least ten draws rather than an extreme value (MINIMUM_DRAWS_IN_TAIL).
MINIMUM_DRAWS = 400


# ----------------------------------------------------------------------- readers


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"{path} carries no record")
    return rows


def read_cohort(path: Path) -> list[dict[str, Any]]:
    """The frozen cohort, refusing a duplicate identifier or a digest mismatch."""

    rows = read_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        identifier = str(row["id"])
        if identifier in seen:
            raise SystemExit(f"identifier {identifier!r} appears twice in {path}")
        seen.add(identifier)
        sequence = row.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            raise SystemExit(f"cohort row {identifier!r} carries no sequence")
        observed = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        declared = row.get("sequence_sha256")
        if declared is not None and str(declared) != observed:
            raise SystemExit(
                f"cohort row {identifier!r} declares sequence_sha256 {declared!r} but its "
                f"sequence hashes to {observed!r}"
            )
    return rows


def require_complete(receipt: dict[str, Any], source: Path) -> None:
    if receipt.get("status") != "complete":
        raise SystemExit(f"{source}: producer receipt is not complete")


def require_annotations_hash(receipt: dict[str, Any], sidecar: Path) -> None:
    declared = receipt.get("annotations_sha256")
    if declared is None:
        raise SystemExit(f"blocked input: {sidecar} receipt lacks annotations_sha256")
    if declared != sha256_file(sidecar):
        raise SystemExit(f"{sidecar}: annotations_sha256 mismatch")


def require_receipt_binding(
    receipt: dict[str, Any], cohort_path: Path, *, pool_rows: list[dict[str, Any]] | None = None
) -> None:
    """Validate every declared binding and require at least one producer-emitted binding."""

    bound = False
    if receipt.get("cohort_sha256") is not None:
        bound = True
        if receipt["cohort_sha256"] != sha256_file(cohort_path):
            raise SystemExit("producer cohort_sha256 mismatch: different cohort")
    if pool_rows is not None and receipt.get("pool_digest") is not None:
        bound = True
        if receipt["pool_digest"] != gp.cohort_digest(pool_rows):
            raise SystemExit("producer pool_digest mismatch: different pool")
    if not bound:
        raise SystemExit("blocked input: producer receipt lacks cohort/pool binding")


def require_sequence_binding(row: dict[str, Any], member: dict[str, Any], source: str) -> None:
    """ID is not sequence identity; validate all emitted raw/digest bindings."""

    expected = hashlib.sha256(member["sequence"].encode("utf-8")).hexdigest()
    bindings = [row.get("sequence_sha256"), (row.get("structure") or {}).get("sequence_sha256")]
    if row.get("sequence") is not None:
        bindings.append(hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest())
    present = [value for value in bindings if value is not None]
    if not present:
        raise SystemExit(f"blocked input: {source} row {member['id']!r} lacks sequence binding")
    if any(value != expected for value in present):
        raise SystemExit(f"{source} row {member['id']!r}: sequence binding mismatch")
    if row.get("arm") != member["arm"]:
        raise SystemExit(f"{source} row {member['id']!r}: arm mismatch")


def structure_values(
    directories: list[Path], *, cohort_rows: list[dict[str, Any]] | None = None
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Per identifier, the selected diffusion sample's structural confidences.

    Shards are sequence-hash disjoint by construction, so a repeated identifier
    across shards means two shard trees describe the same row and is refused.
    """

    values: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    members = {str(row["id"]): row for row in cohort_rows or []}
    for directory in directories:
        indexes = sorted(directory.glob("index-*-of-*.jsonl"))
        if not indexes:
            raise SystemExit(f"{directory} holds no ESMFold2 index; nothing was folded")
        summary = directory / "structure_evidence.json"
        if not summary.is_file():
            raise SystemExit(
                f"{summary} is absent: the folding cell did not reach its completion "
                "record, so its index is a partial tree and is not read"
            )
        summary_record = json.loads(summary.read_text(encoding="utf-8"))
        # Structure producer has status_counts, not a top-level completion status.
        if not isinstance(summary_record.get("status_counts"), dict):
            raise SystemExit(f"blocked input: {summary} lacks producer status_counts")
        if summary_record["status_counts"].get("pending", 0):
            raise SystemExit(f"{summary}: pending structure rows remain")
        if summary_record.get("status") not in (None, "complete"):
            raise SystemExit(f"{summary}: failed structure receipt")
        receipts.append(
            {
                "directory": str(directory),
                "summary": summary_record,
                "indexes": [sha256_file(index) for index in indexes],
            }
        )
        for index in indexes:
            for row in read_jsonl(index):
                identifier = str(row["id"])
                if identifier in values:
                    raise SystemExit(f"identifier {identifier!r} appears in two ESMFold2 shards")
                structure = row.get("structure") or {}
                if identifier in members:
                    require_sequence_binding(row, members[identifier], "structure")
                values[identifier] = {
                    "sequence": row.get("sequence"),
                    "sequence_sha256": row.get("sequence_sha256"),
                    "structure_sequence_sha256": structure.get("sequence_sha256"),
                    "status": structure.get("status"),
                    "reason": structure.get("reason"),
                    "mean_ca_plddt": structure.get("mean_ca_plddt"),
                    "ptm": structure.get("ptm"),
                }
    if not values:
        raise SystemExit("the ESMFold2 shards carried no row")
    return values, receipts


def recognition_values(
    directory: Path, *, cohort_rows: list[dict[str, Any]] | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The Pfam oracle's annotations, refused unless its control passed."""

    control = json.loads((directory / "oracle_control.json").read_text(encoding="utf-8"))
    if not control.get("passed"):
        raise SystemExit(
            "the family-recognition oracle's control did not pass, so its recognition set "
            "is not evidence and no family control or family-conditioned number is computed"
        )
    receipt_path = directory / "family_recognition.json"
    if not receipt_path.is_file():
        raise SystemExit(f"blocked input: {receipt_path} completion receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require_complete(receipt, receipt_path)
    require_annotations_hash(receipt, directory / "family_recognition.jsonl")
    members = {str(row["id"]): row for row in cohort_rows or []}
    values: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(directory / "family_recognition.jsonl"):
        identifier = str(row["id"])
        if identifier in values:
            raise SystemExit(f"identifier {identifier!r} appears twice in the family recognition")
        if identifier in members:
            require_sequence_binding(row, members[identifier], "recognition")
            if row.get("profile_search_status") != "searched":
                raise SystemExit(f"recognition row {identifier!r}: not searched")
        families = row.get("pfam_families")
        if families is None:
            raise SystemExit(
                f"family recognition row {identifier!r} carries no pfam_families key; a "
                "missing key is not the empty family set"
            )
        values[identifier] = {
            "sequence_sha256": row.get("sequence_sha256"),
            "arm": row.get("arm"),
            "pfam_families": [str(family) for family in families],
            "complete_domain": bool(row.get("complete_domain")),
            "any_profile_hit": bool(row.get("any_profile_hit")),
        }
    if not values:
        raise SystemExit(f"{directory} carries no family recognition row")
    return values, control


def novelty_values(
    directory: Path, pool_ids: list[str], *, cohort_path: Path | None = None,
    cohort_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per identifier, the nearest-corpus identity. Every pool member must carry one."""

    receipt = json.loads((directory / "generated_novelty.json").read_text(encoding="utf-8"))
    require_complete(receipt, directory / "generated_novelty.json")
    require_annotations_hash(receipt, directory / "generated_novelty.jsonl")
    if cohort_path is not None:
        require_receipt_binding(receipt, cohort_path)
    members = {str(row["id"]): row for row in cohort_rows or []}
    values: dict[str, float] = {}
    for row in read_jsonl(directory / "generated_novelty.jsonl"):
        identifier = str(row["id"])
        if identifier in values:
            raise SystemExit(f"identifier {identifier!r} appears twice in the novelty sidecar")
        if identifier in members:
            if row.get("arm") != members[identifier]["arm"]:
                raise SystemExit(f"novelty row {identifier!r}: arm mismatch")
            if row.get("searched") is not True:
                raise SystemExit(f"novelty row {identifier!r}: not searched")
            if row.get("sequence") is not None or row.get("sequence_sha256") is not None:
                require_sequence_binding(row, members[identifier], "novelty")
        raw = row.get("nearest_corpus_identity")
        if raw is None:
            raise SystemExit(
                f"novelty row {identifier!r} carries no nearest_corpus_identity; a missing "
                "identity is not the same as an identity of zero"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise SystemExit(f"novelty row {identifier!r} carries a non-finite identity")
        values[identifier] = value
    missing = sorted(set(pool_ids) - set(values))
    if missing:
        raise SystemExit(
            f"{len(missing)} pool members carry no novelty row (first {missing[:3]}); the "
            "identity control is incomplete and no rung that needs it is reported"
        )
    return values, receipt


def likelihood_values(
    directories: list[Path], *, cohort_path: Path | None = None,
    cohort_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Per identifier, the recomputed likelihood row. A second scoring is refused."""

    values: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for directory in directories:
        receipt = json.loads((directory / "generated_likelihood.json").read_text(encoding="utf-8"))
        require_complete(receipt, directory / "generated_likelihood.json")
        require_annotations_hash(receipt, directory / "generated_likelihood.jsonl")
        scorer = receipt.get("scorer_arm", receipt.get("arm"))
        pool_arm = receipt.get("pool_arm", receipt.get("arm"))
        if not scorer or not pool_arm or scorer != pool_arm or receipt.get("arm") != scorer:
            raise SystemExit("likelihood receipt: own-arm scoring required; arm mismatch")
        members = {str(row["id"]): row for row in cohort_rows or []}
        if cohort_path is not None:
            arm_rows = [row for row in cohort_rows or [] if row["arm"] == pool_arm]
            if not arm_rows:
                raise SystemExit("likelihood receipt: pool arm absent from cohort")
            require_receipt_binding(receipt, cohort_path, pool_rows=arm_rows)
        receipts.append(receipt)
        for row in read_jsonl(directory / "generated_likelihood.jsonl"):
            identifier = str(row["id"])
            if identifier in values:
                raise SystemExit(f"identifier {identifier!r} was scored twice for likelihood")
            if row.get("arm") != pool_arm or row.get("pool_arm", pool_arm) != pool_arm:
                raise SystemExit(f"likelihood row {identifier!r}: pool arm mismatch")
            if row.get("scorer_arm", scorer) != scorer:
                raise SystemExit(f"likelihood row {identifier!r}: scorer arm mismatch")
            if cohort_rows is not None:
                if identifier not in members or members[identifier]["arm"] != pool_arm:
                    raise SystemExit(f"likelihood row {identifier!r}: not in declared pool")
                if row.get("sequence") is not None or row.get("sequence_sha256") is not None:
                    require_sequence_binding(row, members[identifier], "likelihood")
            values[identifier] = row
    if not values:
        raise SystemExit("the likelihood directories carried no row")
    return values, receipts


# ------------------------------------------------------------------- assembling


def endpoint_values(
    rows: list[dict[str, Any]], structure: dict[str, dict[str, Any]], endpoint: str
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """One endpoint over the rows the structure instrument evaluated, with drops named."""

    values: list[float] = []
    kept: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for row in rows:
        block = structure.get(str(row["id"]))
        if block is None:
            dropped["no_structure_row"] += 1
            continue
        if block.get("status") != "ok":
            dropped[f"structure_status_{block.get('status')}"] += 1
            continue
        raw = block.get(endpoint)
        if raw is None:
            dropped[f"{endpoint}_absent"] += 1
            continue
        value = float(raw)
        if not math.isfinite(value):
            dropped[f"{endpoint}_non_finite"] += 1
            continue
        values.append(value)
        kept.append(row)
    return np.asarray(values, dtype=np.float64), kept, dict(sorted(dropped.items()))


def unit_labels(rows: list[dict[str, Any]], unit: str) -> list[str]:
    labels: list[str] = []
    for row in rows:
        if unit == "id":
            labels.append(str(row["id"]))
        else:
            value = row.get(unit)
            if value is None:
                raise SystemExit(
                    f"pool member {row['id']!r} carries no {unit!r}, which its declaration "
                    "names as the independence unit"
                )
            labels.append(str(value))
    return labels


def source_hashes(args: argparse.Namespace, structure_receipts: list[dict[str, Any]]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    def add(role: str, path: Path) -> None:
        entries.append({"role": role, "path": str(path), "sha256": sha256_file(path)})

    add("cohort_jsonl", args.cohort_dir / "evaluation_cohort.jsonl")
    add("e17_declaration", args.cohort_dir / "e17_declaration.json")
    add("residue_background", args.cohort_dir / "residue_background.json")
    for index, receipt in enumerate(structure_receipts):
        directory = Path(receipt["directory"])
        add(f"structure_{index}_evidence", directory / "structure_evidence.json")
        for shard, index_file in enumerate(sorted(directory.glob("index-*-of-*.jsonl"))):
            add(f"structure_{index}_index_{shard}", index_file)
    add("family_recognition_receipt", args.recognition / "family_recognition.json")
    add("oracle_control", args.recognition / "oracle_control.json")
    add("family_recognition", args.recognition / "family_recognition.jsonl")
    for index, directory in enumerate(args.likelihood):
        add(f"likelihood_{index}_receipt", directory / "generated_likelihood.json")
        add(f"likelihood_{index}_rows", directory / "generated_likelihood.jsonl")
    if args.novelty is not None:
        add("novelty_receipt", args.novelty / "generated_novelty.json")
        add("novelty_rows", args.novelty / "generated_novelty.jsonl")
    return entries


def code_hashes() -> list[dict[str, str]]:
    """SHA-256 of the two sources that produced a record."""

    return [
        {"role": role, "path": path, "sha256": sha256_file(REPO_ROOT / path)}
        for role, path in CODE_HASH_SOURCES
    ]


def rung_support(
    support_rows: list[dict[str, Any]],
    novelty: dict[str, float] | None,
    recognition: dict[str, dict[str, Any]],
) -> tuple[list[int], Counter[str]]:
    """The endpoint support's indices that carry every covariate the full rung needs."""

    indices: list[int] = []
    excluded: Counter[str] = Counter()
    for position, row in enumerate(support_rows):
        identifier = str(row["id"])
        if identifier not in recognition:
            excluded["recognition_missing"] += 1
            continue
        if novelty is None:
            excluded["novelty_output_not_supplied"] += 1
            continue
        if identifier not in novelty:
            excluded["novelty_missing"] += 1
            continue
        indices.append(position)
    return indices, excluded


def endpoint_report(
    support_rows: list[dict[str, Any]],
    endpoint: str,
    endpoint_vector: np.ndarray,
    dropped: dict[str, int],
    likelihood: dict[str, dict[str, Any]],
    features: dict[str, lr.RowFeatures],
    recognition: dict[str, dict[str, Any]],
    novelty: dict[str, float] | None,
    unit: str,
    *,
    seed: int,
    draws: int,
    rank_policy: str,
) -> dict[str, Any]:
    """Every diagnostic for one arm and one endpoint."""

    labels = unit_labels(support_rows, unit)
    likelihood_vector = np.asarray(
        [likelihood[str(row["id"])]["mean_nll_per_token_nats"] for row in support_rows],
        dtype=np.float64,
    )
    if not np.isfinite(likelihood_vector).all():
        raise SystemExit("a pool member carries a non-finite likelihood")
    full_indices, rung3_excluded = rung_support(support_rows, novelty, recognition)
    vocabulary = (
        lr.family_vocabulary([features[str(support_rows[i]["id"])] for i in full_indices])
        if full_indices
        else ()
    )

    record: dict[str, Any] = {
        "quantity": lr.ENDPOINT_QUANTITY[endpoint],
        "orientation": "larger is better",
        "n_evaluable": int(len(support_rows)),
        "n_dropped": int(sum(dropped.values())),
        "dropped_reasons": dropped,
        "likelihood": {
            "declared_selector": "mean_nll_per_token_nats",
            "orientation": lr.METHOD["likelihood_orientation"],
            "n_missing": 0,
            "statement": (
                "every pool member carries a recomputed likelihood; a missing one stops "
                "the run rather than shrinking the pool"
            ),
        },
        "whole_pool_spearman": (
            lr.whole_pool_spearman(likelihood_vector, endpoint_vector) if support_rows else
            {"status": "unavailable", "reason": "no_support", "spearman": None, "n_rows": 0}
        ),
        "tail_spearman": {},
        "control_ladder": {},
        "selection": {},
    }
    if not support_rows:
        record.update(status="unavailable", reason="no_support", family_vocabulary=[],
                      family_vocabulary_size=0, family_reference=None, recognition_missing_in_support=0)
        return record
    record.update(status="measured", reason=None)
    for fraction in lr.SELECTION_FRACTIONS:
        tail = lr.tail_spearman(likelihood_vector, endpoint_vector, labels, fraction=fraction)
        tail["bootstrap"] = (
            lr.cluster_bootstrap_tail_spearman(
                likelihood_vector,
                endpoint_vector,
                labels,
                fraction=fraction,
                seed=seed,
                n_bootstrap=draws,
            )
            if tail["spearman"] is not None
            else None
        )
        record["tail_spearman"][f"{fraction:.2f}"] = tail

    ladder: dict[str, dict[str, Any]] = {}
    for rung in lr.RUNGS:
        if rung.level < 3:
            subset: list[int] = list(range(len(support_rows)))
            excluded: Counter[str] = Counter()
            rung_vocabulary: tuple[str, ...] = ()
        else:
            subset = list(full_indices)
            excluded = Counter(rung3_excluded)
            rung_vocabulary = vocabulary
        ladder[rung.name] = {
            "subset": subset,
            "excluded": excluded,
            "vocabulary": rung_vocabulary,
            "status": None,
            "reason": None,
            "design": None,
        }

    for rung in lr.RUNGS:
        state = ladder[rung.name]
        subset = state["subset"]
        excluded = state["excluded"]
        rung_vocabulary = state["vocabulary"]
        previous = ladder[lr.RUNGS[rung.level - 2].name] if rung.level > 1 else None
        block: dict[str, Any] = {
            "statement": rung.statement,
            "family_reference": lr.family_reference(rung_vocabulary) if rung.uses_family else None,
            "column_names": list(lr.rung_column_names(rung, rung_vocabulary)),
            "n_columns": len(lr.rung_column_names(rung, rung_vocabulary)),
            "columns_added_over_previous_rung": list(
                lr.rung_added_columns(rung, rung_vocabulary)
            ),
            "n_support": int(len(subset)),
            "n_excluded": int(sum(excluded.values())),
            "excluded_reasons": dict(sorted(excluded.items())),
            "support_overlap_with_previous_rung": (
                None if previous is None else int(len(set(subset) & set(previous["subset"])))
            ),
        }
        unavailable_reason: str | None = None
        if rung.level == 3 and novelty is None:
            unavailable_reason = "novelty_output_not_supplied"
        elif rung.level == 3 and not lr.has_family_variation(rung_vocabulary):
            unavailable_reason = "family_categorical_has_no_variation"
        elif not subset:
            unavailable_reason = "no_support"
        if unavailable_reason is not None:
            block.update(
                status="unavailable",
                reason=unavailable_reason,
                partial_spearman=None,
                bootstrap=None,
            )
            state["status"] = "unavailable"
            state["reason"] = unavailable_reason
            record["control_ladder"][rung.name] = block
            continue
        subset_rows = [support_rows[index] for index in subset]
        design, column_names = lr.build_design(
            [features[str(row["id"])] for row in subset_rows], rung, rung_vocabulary
        )
        block["absent_columns"] = list(lr.absent_column_names(design, column_names))
        subset_likelihood = likelihood_vector[subset]
        subset_endpoint = endpoint_vector[subset]
        subset_units = [labels[index] for index in subset]
        point = lr.partial_spearman(
            subset_likelihood, subset_endpoint, design, units=subset_units, rank_policy=rank_policy
        )
        unadjusted = lr.whole_pool_spearman(subset_likelihood, subset_endpoint)
        block["unadjusted_spearman_on_rung_support"] = unadjusted
        block["partial_spearman"] = point
        block["paired_difference_point"] = (
            None
            if point["spearman"] is None or unadjusted["spearman"] is None
            else float(point["spearman"] - unadjusted["spearman"])
        )
        if point["status"] == "measured":
            block["bootstrap"] = lr.cluster_bootstrap_association(
                subset_likelihood,
                subset_endpoint,
                design,
                subset_units,
                seed=seed,
                n_bootstrap=draws,
                rank_policy=rank_policy,
            )
        else:
            block["bootstrap"] = None
        block["status"] = point["status"]
        block["reason"] = point["reason"]
        state["status"] = point["status"]
        state["reason"] = point["reason"]
        state["design"] = design
        state["subset_likelihood"] = subset_likelihood
        state["subset_endpoint"] = subset_endpoint
        record["control_ladder"][rung.name] = block

    raw_by_fraction = {
        f"{fraction:.2f}": lr.raw_selection_block(
            likelihood_vector, endpoint_vector, fraction=fraction
        )
        for fraction in lr.SELECTION_FRACTIONS
    }
    for fraction in lr.SELECTION_FRACTIONS:
        key = f"{fraction:.2f}"
        point: dict[str, Any] = {
            "fraction": float(fraction),
            "n_pool": int(len(support_rows)),
            "raw": raw_by_fraction[key],
            "adjusted": {},
        }
        for rung in lr.RUNGS:
            state = ladder[rung.name]
            if state["status"] != "measured":
                point["adjusted"][rung.name] = {
                    "status": "unavailable",
                    "reason": state["reason"],
                }
                continue
            block = lr.selection_diagnostics(
                state["subset_likelihood"],
                state["subset_endpoint"],
                state["design"],
                fraction=fraction,
                rung_name=rung.name,
                rank_policy=rank_policy,
            )
            point["adjusted"][rung.name] = {
                "status": block["status"],
                "reason": block.get("reason"),
                **({} if block.get("adjusted") is None else block["adjusted"]),
            }
        record["selection"][key] = point

    record["family_reference"] = lr.family_reference(vocabulary)
    record["family_vocabulary"] = list(vocabulary)
    record["family_vocabulary_size"] = len(vocabulary)
    record["recognition_missing_in_support"] = int(rung3_excluded.get("recognition_missing", 0))
    return record


# --------------------------------------------------------------------------- run


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    rank_policy = (
        lr.RANK_POLICY_COLUMN_SPAN
        if args.column_span_reanalysis
        else lr.RANK_POLICY_FULL_COLUMN_RANK
    )
    cohort_path = args.cohort_dir / "evaluation_cohort.jsonl"
    e17 = json.loads((args.cohort_dir / "e17_declaration.json").read_text(encoding="utf-8"))
    background = json.loads((args.cohort_dir / "residue_background.json").read_text(encoding="utf-8"))
    cohort = read_cohort(cohort_path)
    pool_rows = [row for row in cohort if "pool" in row.get("roles", [])]
    if not pool_rows:
        raise SystemExit(f"{cohort_path} carries no E17 pool member")
    digest = gp.cohort_digest(pool_rows)
    if e17.get("pool_digest") != digest:
        raise SystemExit(
            f"the recomputed pool digest {digest} does not match the declaration's "
            f"{e17.get('pool_digest')}; the cohort and its declaration are not the same artefact"
        )

    structure, structure_receipts = structure_values(args.structure, cohort_rows=pool_rows)
    recognition, control = recognition_values(args.recognition, cohort_rows=pool_rows)
    likelihood, likelihood_receipts = likelihood_values(
        args.likelihood, cohort_path=cohort_path, cohort_rows=pool_rows
    )
    pool_ids = [str(row["id"]) for row in pool_rows]
    absent = [identifier for identifier in pool_ids if identifier not in likelihood]
    if absent:
        raise SystemExit(
            f"{len(absent)} pool members carry no recomputed likelihood (first {absent[:3]}); "
            "the selector is incomplete and no analysis is reported"
        )
    novelty: dict[str, float] | None = None
    novelty_receipt: dict[str, Any] | None = None
    if args.novelty is not None:
        novelty, novelty_receipt = novelty_values(
            args.novelty, pool_ids, cohort_path=cohort_path, cohort_rows=pool_rows
        )

    features: dict[str, lr.RowFeatures] = {}
    for row in pool_rows:
        identifier = str(row["id"])
        families = recognition[identifier]["pfam_families"] if identifier in recognition else []
        features[identifier] = lr.make_row_features(
            str(row["sequence"]),
            int(row["length"]),
            nearest_corpus_identity=None if novelty is None else novelty.get(identifier),
            families=families,
        )

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        by_arm[str(row["arm"])].append(row)

    arms: dict[str, Any] = {}
    for arm in sorted(by_arm):
        members = by_arm[arm]
        declaration = e17.get("pools", {}).get(arm)
        if declaration is None:
            raise SystemExit(f"the cohort carries pool arm {arm!r}, which the declaration does not")
        unit = str(declaration["independence_unit"])
        labels = unit_labels(members, unit)
        n_units = len({*labels})
        if int(declaration.get("n_pool", -1)) != len(members):
            raise SystemExit(
                f"arm {arm!r} declares {declaration.get('n_pool')} pool members but the cohort "
                f"carries {len(members)}"
            )
        if int(declaration.get("n_units", -1)) != n_units:
            raise SystemExit(
                f"arm {arm!r} declares {declaration.get('n_units')} independence units but the "
                f"cohort resolves {n_units}"
            )
        arm_record: dict[str, Any] = {
            "declaration": declaration,
            "n_pool": len(members),
            "independence": {
                "unit": unit,
                "n_units": n_units,
                "unit_floor": bootstrap_unit_floor(n_units),
                "statement": (
                    "replicates resample these units with replacement, retaining every row of "
                    "a drawn unit and preserving a repeated unit in full; no identity dedup"
                ),
            },
            "recognition_present": int(sum(str(row["id"]) in recognition for row in members)),
            "novelty_present": (
                None
                if novelty is None
                else int(sum(str(row["id"]) in novelty for row in members))
            ),
            "length": {
                "min": int(min(row["length"] for row in members)),
                "max": int(max(row["length"] for row in members)),
                "mean": float(np.mean([row["length"] for row in members])),
            },
            "endpoints": {},
        }
        for endpoint in lr.ENDPOINTS:
            vector, support_rows, dropped = endpoint_values(members, structure, endpoint)
            arm_record["endpoints"][endpoint] = endpoint_report(
                support_rows,
                endpoint,
                vector,
                dropped,
                likelihood,
                features,
                recognition,
                novelty,
                unit,
                seed=args.seed,
                draws=args.draws,
                rank_policy=rank_policy,
            )
        arms[arm] = arm_record

    record = {
        "schema_version": lr.SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "cohort_dir": str(args.cohort_dir),
            "structure": [str(path) for path in args.structure],
            "recognition": str(args.recognition),
            "likelihood": [str(path) for path in args.likelihood],
            "novelty": None if args.novelty is None else str(args.novelty),
        },
        "source_hashes": source_hashes(args, structure_receipts),
        "cohort": {
            "path": str(cohort_path),
            "pool_digest": digest,
            "declaration_pool_digest": e17.get("pool_digest"),
            "n_pool": len(pool_rows),
            "question": e17.get("question"),
            "independence": e17.get("independence"),
            "validity_conditions": e17.get("validity_conditions"),
            "residue_background": {
                "schema_version": background.get("schema_version"),
                "source": background.get("source"),
                "n_records": background.get("n_records"),
                "smoothing": background.get("smoothing"),
                "role": background.get("role"),
            },
        },
        "structure_receipts": structure_receipts,
        "likelihood_receipts": likelihood_receipts,
        "oracle_control": {
            key: control[key] for key in ("passed", "n_positive", "n_negative") if key in control
        },
        "novelty_receipt": (
            None
            if novelty_receipt is None
            else {
                key: novelty_receipt[key]
                for key in (
                    "n_searched",
                    "n_with_any_hit",
                    "identity_over_query",
                    "identity_strata",
                    "masking_rationale",
                    "database",
                    "ceiling",
                )
                if key in novelty_receipt
            }
        ),
        "method": lr.METHOD,
        "rank_policy": rank_policy,
        "rank_policy_statement": lr.RANK_POLICY_STATEMENTS[rank_policy],
        "limitations": list(lr.LIMITATIONS)
        + (list(lr.COLUMN_SPAN_LIMITATIONS) if args.column_span_reanalysis else []),
        "control_ladder": [
            {
                "name": rung.name,
                "level": rung.level,
                "statement": rung.statement,
                "columns": list(lr.rung_column_names(rung, ())),
            }
            for rung in lr.RUNGS
        ],
        "selection_fractions": list(lr.SELECTION_FRACTIONS),
        "primary_endpoint": lr.PRIMARY_ENDPOINT,
        "secondary_endpoint": lr.SECONDARY_ENDPOINT,
        "endpoints": list(lr.ENDPOINTS),
        "arms": arms,
        "seed": int(args.seed),
        "n_bootstrap": int(args.draws),
        "reading_guide": [
            "confidence is not stability and not function: both endpoints are ESMFold2's own "
            "prediction confidence",
            "a lower mean negative log-likelihood is a higher likelihood, so every selection "
            "direction ranks ascending NLL. All correlations use -mean NLL (high likelihood), "
            "so negative means higher likelihood accompanies worse quality",
            "the tail association is a selection-conditioned diagnostic on the tail alone, not "
            "a pool-wide or causal statement. Below-floor tail point estimates are flagged "
            "and intervals withheld; every draw counts distinct original tail units",
            "read the control ladder top-down: a likelihood association that length alone "
            "reproduces is a length association, and one that composition also reproduces is a "
            "composition association",
            "the adjusted selection evaluates a residualized endpoint, a different quantity from "
            "the raw endpoint; the two selected means are never differenced",
            "every outcome regression here is a diagnostic of what the likelihood association "
            "survives and is not a trained selector",
        ],
    }
    if args.column_span_reanalysis:
        record["analysis_label"] = lr.POSTHOC_COLUMN_SPAN_LABEL
        record["code_hashes"] = code_hashes()
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True, help="the frozen cohort directory")
    parser.add_argument("--structure", type=Path, nargs="+", required=True, help="ESMFold2 shard output dirs")
    parser.add_argument("--recognition", type=Path, required=True, help="the Pfam oracle output dir")
    parser.add_argument("--likelihood", type=Path, nargs="+", required=True, help="likelihood output dirs")
    parser.add_argument(
        "--novelty",
        type=Path,
        default=None,
        help="the homology-search output dir supplying each pool member's nearest corpus identity",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument(
        "--column-span-reanalysis",
        action="store_true",
        help=(
            "run the labelled post-hoc column-span reanalysis: admit a rank-deficient "
            "control design because the least-squares projection onto its column span is "
            "unique, while saturation, constant residuals and the unit floors still refuse"
        ),
    )
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < MINIMUM_DRAWS:
        parser.error(
            f"--draws below {MINIMUM_DRAWS} gives percentile tail endpoints that are extreme "
            "draws rather than order statistics"
        )
    record = run(args)
    print(
        json.dumps(
            {
                "status": record["status"],
                "arms": sorted(record["arms"]),
                "endpoints": record["endpoints"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
