#!/usr/bin/env python3
"""Downstream analysis of the objective-tuned generation campaign.

Reads completed artefacts only: the free-length lane's attempt ledgers, the
ESMFold2 prediction records, the UniRef50 novelty searches and -- where they
exist -- the Pfam recognition table and a per-sequence selector score. It runs
no model, folds nothing, searches nothing and draws nothing beyond the declared
bootstrap. Everything the registration fixes lives in
``src.capability.generation.objective_tuned_analysis``; this script finds the
cells, joins them and refuses an input that cannot carry an endpoint.

Why the join is on the sequence digest and not on the attempt id
----------------------------------------------------------------
``unconditional_generation.attempt_row`` derives an attempt id from the role,
the arm, the lane's campaign string and the attempt's index *within its cell*.
This campaign runs twelve base cells and three tuned cells per arm, so index 0
of every cell of one arm carries the same id -- measured, and true even when the
two cells drew different sequences. The id therefore cannot key a pooled table.
Both quantities the join needs are functions of the sequence: the structure
stage already folds exact duplicates once ("duplicate sequences share a fold,
not a sampling unit") and the nearest natural relative of a sequence is a
property of that sequence. So pLDDT, identity, recognition and selector score
are all keyed by ``sequence_sha256``, while every attempt stays its own
resampling unit under a key of this analysis's own. The novelty sidecar carries
no sequence, so a ``--novelty`` source is the stage's output *directory*: its
record pins the cohort it searched and the cohort supplies the digest.

What comes out, and in what units
---------------------------------
* the primary stratified contrast: tuned minus base mean per-residue ESMFold2
  CA-pLDDT in points on the 0-100 scale, within each of the four frozen
  identity strata, per matched pair, with its scale-calibrated margin and the
  registration's decision rule;
* the explicit-statistics control: per stratum and per rule, the share of the
  tuned-minus-base pLDDT gain a top-quarter filter of the base pool reproduces,
  dimensionless, with a simultaneous band on its own declared family -- three
  families, one per rule -- and the disposition of that band against full
  reproduction, beside the pLDDT points it is built from and the identity axis;
* the identity-distribution shift: tuned minus base mean percent identity over
  the query, and tuned minus base fraction of retained attempts below 30%;
* family recognition: tuned minus base rate of any Pfam family and of a complete
  domain at the declared coverage, as fractions of retained attempts;
* selector competence: the pLDDT headroom an oracle reaches at a budget of ten
  within an arm's own pool, in pLDDT points, and the share a selector captures;
* the ZymCTRL pair's within-arm identity distribution under native EC
  conditioning, with its mismatched-label negative beside it and no structural
  contrast at all.

Invocation
----------
::

    scripts/capability/generation/analyse_objective_tuned.py \\
      --campaign-root results/external_baseline/<run-id> \\
      --structure <dir-or-predictions.jsonl> [...] \\
      --novelty <novelty-stage-output-dir> [...] \\
      --kmer-background data/kmer_background/uniref50_high_order \\
      --out results/R6/objective_tuned_analysis_<date> \\
      [--recognition recognition.jsonl] [--selector-score selector.jsonl] \\
      [--within-arm-novelty <dir> --conditioned-root results/R6/conditioned_generation_20260826]

``--dry-run`` resolves and validates every input, writes nothing and prints the
realised cell layout, which is the way to check a campaign before it finishes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_objective_tuned_campaign as builder  # noqa: E402

from src.capability.context.homology import assign_stratum  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import generative_control as gc  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402
from src.capability.generation.structure_evidence import sequence_digest  # noqa: E402

COMPLETION = "objective_tuned_analysis.json"

#: The queue's cell directory name, as ``build_objective_tuned_campaign.label``
#: writes it. The arm name is carried verbatim, dots included.
CELL_DIRECTORY = re.compile(rf"^{builder.KEY}_(?P<arm>.+)_s(?P<stream>\d+)_r(?P<replicate>\d+)$")

#: Corpus k-mer orders the two declared corpus rules read: order 0 needs k = 1,
#: order 2 needs k = 3.
KMER_ORDERS = (1, 3)

JOIN_RULE = (
    "structure_evidence.sequence_digest of the sequence's own bytes, recomputed "
    "here rather than read from a column. The lane's attempt id is a digest of "
    "role, arm, campaign and the attempt's index within its cell, so the "
    "replicate cells of one arm share ids and the id cannot key a pooled table; "
    "and a cohort may key its rows by the sequence digest or carry it in a "
    "sequence_sha256 column, so the bytes are the one thing every shape agrees "
    "on. A recorded digest that disagrees with the bytes is refused. pLDDT, "
    "identity, recognition and selector score are all functions of the "
    "sequence; every attempt remains its own resampling unit under a key of "
    "this analysis's own"
)


def digest_of(row: dict[str, Any], source: Path | str, what: str) -> str:
    """The sequence digest of one record, from its bytes, checked against its column."""

    sequence = row.get("sequence")
    recorded = row.get("sequence_sha256")
    if not isinstance(sequence, str) or not sequence:
        if isinstance(recorded, str) and recorded:
            return recorded
        fail(f"{source}: a {what} carries neither a sequence nor a sequence_sha256, so it "
             "cannot be joined")
    digest = sequence_digest(sequence)
    if isinstance(recorded, str) and recorded and recorded != digest:
        fail(f"{source}: a {what} records sequence_sha256 {recorded} for bytes that digest to "
             f"{digest}; the join key and the sequence disagree")
    return digest


def fail(message: str) -> None:
    raise SystemExit(f"analyse_objective_tuned: {message}")


def bands(columns: list[dict[str, Any]], family: str) -> dict[str, Any]:
    """One family's simultaneous bands, refusing a family the declaration did not size."""

    try:
        return ota.contrast_bands(columns, family=family)
    except (ValueError, KeyError) as error:
        fail(str(error))
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def expand(paths: list[Path], name: str) -> list[Path]:
    """Accept a file, or a directory holding that file anywhere beneath it."""

    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(path.rglob(name))
            if not found:
                fail(f"{path} holds no {name}")
            resolved.extend(found)
        elif path.is_file():
            resolved.append(path)
        else:
            fail(f"{path} is neither a file nor a directory")
    return resolved


# ------------------------------------------------------------- the cell layout

def pair_of(arm: str) -> ot.ObjectiveTunedPair:
    try:
        return ot.pair_for(arm)
    except KeyError as error:
        fail(f"{arm} is not an arm of any declared objective-tuned pair ({error})")
        raise


def read_campaign(root: Path) -> dict[str, Any]:
    """Every cell of the free-length campaign, grouped into (arm, stream) pools.

    Refuses a cell that is not a declared arm of a free-length pair, a cell whose
    ledger is short of the policy's attempt count, a stream whose cell count
    differs from the declared base-to-tuned budget ratio, a repeated draw seed,
    and any arm of the pair the amendment reads within-arm only.
    """

    if not root.is_dir():
        fail(f"--campaign-root {root} is not a directory")
    within_arm = {arm for pair in ota.WITHIN_ARM_PAIRS for arm in pair.arms}
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        match = CELL_DIRECTORY.match(directory.name)
        if match is None:
            continue
        arm = match.group("arm")
        if arm in within_arm:
            fail(f"{directory.name}: {arm} belongs to the pair the 2026-11-10 amendment reads "
                 f"within-arm only. {ota.ZYMCTRL_NON_IDENTIFIABLE}")
        pair = pair_of(arm)
        if pair.lane != ot.LANE_FREE_LENGTH:
            fail(f"{directory.name}: {arm} is declared into the {pair.lane} lane")
        summary_path = directory / ug.EXPECT_GENERATE
        ledger_path = directory / ug.LEDGER_NAME
        for path in (summary_path, ledger_path):
            if not path.is_file():
                fail(f"{directory.name} is incomplete: {path.name} is missing")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("arm") != arm:
            fail(f"{directory.name}: the ledger summary names arm {summary.get('arm')!r}")
        if summary.get("is_scientific_measurement") is not True:
            fail(f"{directory.name}: is_scientific_measurement is "
                 f"{summary.get('is_scientific_measurement')!r}; a cell short of the policy's "
                 "attempt count is an interface check, not a measurement")
        rows = read_jsonl(ledger_path)
        if len(rows) != int(summary.get("attempts", -1)):
            fail(f"{directory.name}: the ledger holds {len(rows)} attempts and the summary "
                 f"declares {summary.get('attempts')}")
        cells[(arm, int(match.group("stream")))].append(
            {"label": directory.name, "replicate": int(match.group("replicate")),
             "seed": summary.get("seed"), "checkpoint": summary.get("checkpoint"),
             "rows": rows}
        )
    if not cells:
        fail(f"--campaign-root {root} holds no {builder.KEY}_<arm>_s<stream>_r<replicate> cell")

    pools: dict[tuple[str, int], dict[str, Any]] = {}
    seeds: dict[int, str] = {}
    for (arm, stream), group in sorted(cells.items()):
        pair = pair_of(arm)
        role = "base" if arm == pair.base else "tuned"
        declared = builder.CELLS_PER_STREAM[role]
        if len(group) != declared:
            fail(f"{arm} stream {stream}: {len(group)} cells against the declared {declared} for "
                 f"a {role} stream. The base arm's four-to-one budget is the selection headroom "
                 "the explicit-statistics control needs")
        group.sort(key=lambda cell: cell["replicate"])
        for cell in group:
            if not isinstance(cell["seed"], int):
                fail(f"{cell['label']}: the ledger summary records no integer draw seed; a "
                     "stream cannot be told from its neighbour without it")
            if cell["seed"] in seeds:
                fail(f"{cell['label']} and {seeds[cell['seed']]} share the draw seed "
                     f"{cell['seed']}")
            seeds[cell["seed"]] = cell["label"]
        attempts = [(cell["label"], row) for cell in group for row in cell["rows"]]
        pools[(arm, stream)] = {
            "arm": arm, "stream": stream, "role": role, "pair": pair.tuned,
            "cells": [cell["label"] for cell in group],
            "seeds": [cell["seed"] for cell in group],
            "checkpoints": sorted({str(cell["checkpoint"]) for cell in group}),
            "attempts": attempts,
        }
    streams_of: dict[str, list[int]] = defaultdict(list)
    for arm, stream in pools:
        streams_of[arm].append(stream)
    for arm, streams in sorted(streams_of.items()):
        if len(streams) != builder.STREAMS:
            fail(f"{arm}: {len(streams)} streams against the declared {builder.STREAMS}")
    return {"root": str(root), "pools": pools}


# ------------------------------------------------------------- the joined table

def read_structure(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Mean CA-pLDDT per sequence digest, refusing an absent or impossible column.

    A record whose ``structure.status`` is not ``ok`` is a declared measurement
    failure and carries no value. A record that claims success and holds no
    ``mean_ca_plddt`` is an absent column -- a broken instrument rather than a
    missing datum -- and is refused by name.
    """

    values: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            digest = digest_of(row, path, "prediction record")
            structure = row.get("structure")
            if not isinstance(structure, dict):
                fail(f"{path}: {digest} carries no structure record")
            if structure.get("status") != "ok":
                entry = {"plddt": None, "status": str(structure.get("status")),
                         "role": row.get("role"), "paired_id": row.get("paired_id")}
            else:
                if structure.get("mean_ca_plddt") is None:
                    fail(f"{path}: {digest} reports status ok and no mean_ca_plddt; the pLDDT "
                         "column is absent and the primary endpoint cannot be read")
                plddt = float(structure["mean_ca_plddt"])
                if not np.isfinite(plddt) or not 0.0 <= plddt <= 100.0:
                    fail(f"{path}: {digest} reports mean_ca_plddt {plddt}, outside the 0-100 "
                         "scale the structure instrument writes")
                entry = {"plddt": plddt, "status": "ok",
                         "role": row.get("role"), "paired_id": row.get("paired_id")}
            previous = values.get(digest)
            if previous is not None and previous["plddt"] != entry["plddt"]:
                fail(f"{digest} carries two different pLDDT values across the supplied "
                     "prediction records; one sequence has one fold")
            values[digest] = entry
    if not values:
        fail("the supplied prediction records hold no rows")
    return values


def novelty_sources(paths: list[Path]) -> list[tuple[Path, Path]]:
    """Each novelty source as its pinned record and its sidecar."""

    sources = []
    for record_path in expand(paths, "generated_novelty.json"):
        rows_path = record_path.parent / "generated_novelty.jsonl"
        if not rows_path.is_file():
            fail(f"{record_path.parent} holds no generated_novelty.jsonl beside its record")
        sources.append((record_path, rows_path))
    if not sources:
        fail("no novelty source was resolved")
    return sources


def read_novelty(sources: list[tuple[Path, Path]]) -> dict[str, dict[str, Any]]:
    """Identity over the query per sequence digest, through each search's own cohort.

    ``identity_stratum`` is the identity band; ``stratum`` in the same records is
    the *length* stratum, and feeding the wrong column in would silently
    restratify the whole analysis, so the band is recomputed from the identity
    and a disagreement is refused. The cohort each search pins supplies the
    digest and is checked against the digest the record pins.
    """

    values: dict[str, dict[str, Any]] = {}
    for record_path, rows_path in sources:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        cohort_path = Path(record["cohort"])
        if not cohort_path.is_absolute():
            cohort_path = REPO_ROOT / cohort_path
        if not cohort_path.is_file():
            fail(f"{record_path} names cohort {record['cohort']}, which is not on disk")
        if sha256_file(cohort_path) != record.get("cohort_sha256"):
            fail(f"{cohort_path} no longer hashes to the digest {record_path} pins")
        cohort = {str(row["id"]): digest_of(row, cohort_path, "cohort row")
                  for row in read_jsonl(cohort_path)}
        for row in read_jsonl(rows_path):
            identifier = str(row.get("id"))
            if identifier not in cohort:
                fail(f"{rows_path}: {identifier} is not in the cohort {cohort_path} that the "
                     "record pins, so its sequence is unknown")
            if "nearest_corpus_identity" not in row or "identity_stratum" not in row:
                fail(f"{rows_path}: {identifier} carries no nearest_corpus_identity and "
                     "identity_stratum; this is not a generated_novelty record")
            if row.get("searched") is not True:
                fail(f"{rows_path}: {identifier} is not marked searched")
            identity = float(row["nearest_corpus_identity"])
            band = str(row["identity_stratum"])
            if assign_stratum(identity) != band:
                fail(f"{rows_path}: {identifier} reports identity {identity} in stratum {band}, "
                     f"which the frozen edges assign to {assign_stratum(identity)}")
            digest = cohort[identifier]
            previous = values.get(digest)
            if previous is not None and previous["identity"] != identity:
                fail(f"{digest} carries two different identities across the supplied novelty "
                     "searches; identity over the query is a property of the sequence")
            values[digest] = {"identity": identity, "stratum": band,
                              "had_any_hit": bool(row.get("had_any_hit"))}
    if not values:
        fail("the supplied novelty searches hold no rows")
    return values


def assemble(campaign: dict[str, Any], structure: dict[str, dict[str, Any]],
             novelty: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Join the ledgers, the folds and the searches into one per-attempt table.

    The denominator is the lane's: every retained attempt counts, empty and
    budget-censored continuations included. An empty attempt carries no identity
    and no fold, which is why the rate endpoints and the mean endpoints state
    different denominators and both are recorded.
    """

    attempt_digests = {str(row["sequence_sha256"])
                       for pool in campaign["pools"].values() for _, row in pool["attempts"]}
    for digest, entry in sorted(structure.items()):
        if digest in attempt_digests or entry["role"] == "composition_shuffle" or entry["paired_id"]:
            continue
        fail(f"the prediction records carry {digest}, which is neither a campaign attempt's "
             "sequence nor a composition shuffle of one")
    outside = sorted(set(novelty) - attempt_digests)

    pools: dict[tuple[str, int], dict[str, Any]] = {}
    for key, pool in campaign["pools"].items():
        rows = []
        for position, (label, row) in enumerate(pool["attempts"]):
            digest = digest_of(row, label, "ledger attempt")
            empty = row["stop_status"] == "empty_or_noncanonical"
            fold = structure.get(digest)
            search = novelty.get(digest)
            if empty and fold is not None and fold["plddt"] is not None:
                fail(f"{label} attempt {position} is empty or non-canonical and carries a pLDDT")
            if fold is not None and fold["plddt"] is not None and search is None:
                fail(f"{label} attempt {position} carries a pLDDT and no identity annotation; "
                     "the primary endpoint is stratified by identity and cannot place it")
            rows.append({
                "key": f"{pool['arm']}|s{pool['stream']}|{position}",
                "cell": label, "digest": digest,
                "sequence": str(row["sequence"]), "length": int(row["length"]),
                "stop_status": row["stop_status"], "empty": empty,
                "valid_aa20": bool(row.get("valid_aa20")),
                "plddt": None if fold is None else fold["plddt"],
                "identity": None if search is None else search["identity"],
                "stratum": None if search is None else search["stratum"],
            })
        non_empty = [row for row in rows if not row["empty"]]
        searched = [row for row in non_empty if row["identity"] is not None]
        folded = [row for row in non_empty if row["plddt"] is not None]
        pools[key] = {
            **{name: pool[name] for name in ("arm", "stream", "role", "pair", "cells", "seeds",
                                             "checkpoints")},
            "rows": rows,
            "counts": {
                "attempts": len(rows), "non_empty": len(non_empty),
                "budget_censored": sum(1 for row in rows if row["stop_status"] == "budget_censored"),
                "empty_or_noncanonical": len(rows) - len(non_empty),
                "searched": len(searched), "folded": len(folded),
                "distinct_sequences": len({row["digest"] for row in non_empty}),
                "identity_search_coverage": len(searched) / len(non_empty) if non_empty else 0.0,
                "fold_coverage": len(folded) / len(non_empty) if non_empty else 0.0,
            },
        }
    return {"pools": pools, "join_rule": JOIN_RULE,
            "novelty_digests_outside_the_campaign": len(outside),
            "denominator_rule": (
                "retained attempts, with empty and budget-censored continuations kept, as the "
                "free-length lane requires. The rate endpoints use that denominator; mean "
                "identity uses the searched attempts and mean pLDDT the folded ones, because "
                "neither quantity exists for an empty attempt"
            )}


def pool_of(table: dict[str, Any], arm: str) -> dict[int, dict[str, Any]]:
    return {stream: pool for (name, stream), pool in table["pools"].items() if name == arm}


def plddt_by_stratum(pools: dict[int, dict[str, Any]], stratum: str,
                     keep: dict[int, set[str]] | None = None) -> dict[int, list[float]]:
    return {stream: [row["plddt"] for row in pool["rows"]
                     if row["plddt"] is not None and row["stratum"] == stratum
                     and (keep is None or row["key"] in keep[stream])]
            for stream, pool in sorted(pools.items())}


def require_pair(table: dict[str, Any], pair: ot.ObjectiveTunedPair) -> tuple[dict, dict]:
    base, tuned = pool_of(table, pair.base), pool_of(table, pair.tuned)
    if not base:
        fail(f"{pair.tuned}: the matched base pool {pair.base} is missing from the campaign "
             "root; a pair without its own control carries no contrast")
    if not tuned:
        fail(f"{pair.tuned}: the tuned pool is missing from the campaign root")
    return base, tuned


# ------------------------------------------------------------------- endpoints

def stratified_contrast(table: dict[str, Any], pairs: tuple[ot.ObjectiveTunedPair, ...],
                        family: str, *, shift: dict[str, Any] | None = None) -> dict[str, Any]:
    """The registration's primary contrast, with its margin and decision rule."""

    columns = []
    for pair in pairs:
        base, tuned = require_pair(table, pair)
        for stratum in ota.STRATA:
            columns.append({"key": f"{pair.tuned}|{stratum}",
                            "base": plddt_by_stratum(base, stratum),
                            "tuned": plddt_by_stratum(tuned, stratum)})
    band = bands(columns, family)
    decisions = {}
    for pair in pairs:
        base, tuned = pool_of(table, pair.base), pool_of(table, pair.tuned)
        per_stratum = {}
        for stratum in ota.STRATA:
            entry = band["columns"][f"{pair.tuned}|{stratum}"]
            if entry["status"] != "resolved":
                per_stratum[stratum] = {"finding": ota.UNRESOLVED, "reason": entry["reason"],
                                        "margin": None, "streams": None}
                continue
            margin = ota.margin_for(entry["base_mean"])
            left, right = plddt_by_stratum(base, stratum), plddt_by_stratum(tuned, stratum)
            streams = {}
            for stream in sorted(set(left) & set(right)):
                difference = float(np.mean(right[stream]) - np.mean(left[stream]))
                streams[str(stream)] = {
                    "difference": difference,
                    "exceeds_margin": bool(difference > margin["margin"]),
                    "units": {"base": len(left[stream]), "tuned": len(right[stream])},
                }
            per_stratum[stratum] = {**ota.margin_finding(entry["interval"], margin["margin"]),
                                    "margin": margin, "streams": streams}
        split = None if shift is None else shift["shift_share"].get(pair.tuned)
        moves = None if shift is None else {
            ota.IDENTITY_MEAN_FAMILY: shift["mean_identity"]["columns"][pair.tuned],
            ota.IDENTITY_FRACTION_FAMILY: shift["fraction_below_30"]["columns"][pair.tuned]}
        decisions[pair.tuned] = {"per_stratum": per_stratum,
                                 "pair_decision": pair_decision(per_stratum, split=split,
                                                                moves=moves)}
    return {"bands": band, "per_pair": decisions,
            "endpoint": "mean per-residue ESMFold2 CA-pLDDT, tuned minus base, within stratum",
            "units": "pLDDT points on the 0-100 scale",
            "decision_rule": (
                "a design-relevant bias needs the difference to exceed its margin in the stratum "
                "below 30% identity, in all three streams; a narrowing is the finding where the "
                "difference is confined to the close-homology and near-duplicate strata. "
                "'Exceeds its margin' is read of the whole simultaneous interval, and the stream "
                "requirement is a conjunction on the per-stream point differences, because the "
                "declared family is pairs crossed with strata and carries no per-stream column"
            )}


def pair_decision(per_stratum: dict[str, dict[str, Any]], *,
                  split: dict[str, Any] | None = None,
                  moves: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The pair-level reading the registration declares, state by state.

    Narrowing has two declared routes and both are reported: the difference
    being confined to the strata nearest the record, and the identity
    distribution itself shifting toward the record by enough to account for the
    pooled gain. ``split`` and ``moves`` carry that second route's inputs, and a
    pair with no declared identity-shift family carries no verdict on it.
    """

    novel = per_stratum[ota.NOVEL_STRATUM]
    remote = per_stratum[ota.STRATA[1]]
    toward_record = [per_stratum[name] for name in ota.RECORD_STRATA]
    streams = novel.get("streams") or {}
    conjunction = bool(streams) and all(entry["exceeds_margin"] for entry in streams.values())
    if novel["finding"] == ota.SUPPORTED and conjunction:
        design = {"finding": ota.SUPPORTED,
                  "reason": "the interval exceeds the margin in the novel stratum and every "
                            "stream's point difference does too"}
    elif novel["finding"] == ota.BOUNDED_NEGATIVE:
        design = {"finding": ota.BOUNDED_NEGATIVE,
                  "reason": f"the novel-stratum interval lies inside its margin: {novel['reason']}"}
    elif novel["finding"] == ota.SUPPORTED:
        design = {"finding": ota.UNRESOLVED,
                  "reason": "the pooled interval exceeds the margin but the three streams are not "
                            "a conjunction, which the registration requires"}
    else:
        design = {"finding": ota.UNRESOLVED, "reason": novel["reason"]}
    confined = (novel["finding"] == ota.BOUNDED_NEGATIVE
                and remote["finding"] == ota.BOUNDED_NEGATIVE
                and any(entry["finding"] == ota.SUPPORTED for entry in toward_record))
    narrowing = ({"finding": ota.SUPPORTED,
                  "reason": "the difference is bounded inside the margin in both remote strata "
                            "and exceeds it toward the record"} if confined else
                 {"finding": ota.UNRESOLVED,
                  "reason": "the difference is not shown to be confined to the close-homology and "
                            "near-duplicate strata. The registration's second route -- the "
                            "identity distribution shifting toward the record 'by enough to "
                            "account for the pooled gain' -- is not quantified in the "
                            "declaration, so no verdict is issued on it and the shift-share size "
                            "is reported descriptively instead"})
    if split is None or moves is None:
        composition = {"finding": ota.UNRESOLVED, "rule": ota.NARROWING_COMPOSITION_RULE,
                       "reason": "this pair carries no declared identity-shift family, so the "
                                 "second narrowing route cannot be read for it"}
    else:
        composition = ota.narrowing_by_composition(split, moves)
    overall = (ota.SUPPORTED
               if ota.SUPPORTED in (narrowing["finding"], composition["finding"])
               else ota.UNRESOLVED)
    return {"design_relevant_bias": design,
            "narrowing_confined_to_record_strata": narrowing,
            "narrowing_by_composition_shift": composition,
            "narrowing": {"finding": overall,
                          "reason": "either declared route suffices; both are reported"}}


def identity_levels(base: dict[int, dict[str, Any]], tuned: dict[int, dict[str, Any]],
                    retained: dict[int, set[str]], stratum: str) -> dict[str, Any]:
    """Mean identity over the query for the three pools of one control cell.

    The registration compares the filtered base pool with the tuned pool "on both
    axes", so the identity axis travels with the objective axis rather than being
    read from a different table.
    """

    def mean(pools: dict[int, dict[str, Any]], keep: dict[int, set[str]] | None) -> float | None:
        values = [row["identity"] for stream, pool in pools.items() for row in pool["rows"]
                  if row["identity"] is not None and row["stratum"] == stratum
                  and (keep is None or row["key"] in keep[stream])]
        return float(np.mean(values)) if values else None

    return {"base_mean_identity": mean(base, None), "tuned_mean_identity": mean(tuned, None),
            "filtered_mean_identity": mean(base, retained),
            "units": "percent identity over the query against UniRef50, masking off"}


def scored_by_stratum(scored: dict[int, list[dict[str, Any]]], stratum: str,
                      keep: dict[int, set[str]] | None = None) -> dict[int, list[float]]:
    """One stratum's pLDDT values per stream, over the rule's own scorable pool."""

    return {stream: [row["plddt"] for row in members
                     if row["stratum"] == stratum
                     and (keep is None or row["key"] in keep[stream])]
            for stream, members in sorted(scored.items())}


def explicit_statistics_control(table: dict[str, Any], pairs: tuple[ot.ObjectiveTunedPair, ...],
                                counts: dict[int, np.ndarray],
                                primary: dict[str, Any]) -> dict[str, Any]:
    """The control that carries the claim: what an explicit rule reproduces.

    Each base pool is ranked within stream, because the declared four-to-one
    budget is per stream and a top quarter is only size-matched to the tuned pool
    there. The pool a rule reads is its scorable folded attempts: the readout
    compares mean pLDDT, so an attempt with no fold would be a different
    denominator rather than a filtered pool, and an attempt the rule cannot score
    could never be retained.

    Three families, one per rule, each the matched pairs crossed with the four
    strata, each banded on its own nested resample and never merged with another
    rule or with the primary family.

    ``primary`` is the primary contrast, and it decides whether each cell's share
    is a quantity at all: a share of a gain the primary endpoint finds not
    bounded away from zero is undefined rather than imprecise, and is recorded
    non-identifiable with the primary endpoint named as the reason.
    """

    rules = ota.filter_rules(counts)
    out: dict[str, Any] = {
        "readout": "the share of the tuned-minus-base pLDDT gain that a top-quarter filter of "
                   "the base pool reproduces, per stratum and per rule",
        "units": "dimensionless; built from pLDDT points",
        "at_or_above_one": "an explicit statistical filter over the model's own unaligned samples "
                           "does everything the alignment did on this endpoint",
        "ranking_scope": "within (arm, stream), over the scorable folded attempts of the base pool",
        "top_fraction": ota.TOP_FRACTION,
        "interval_note": ota.CONTROL_INTERVAL_NOTE,
        "nested_draws": ota.NESTED_DRAW_METHOD,
        "moot_note": ota.CONTROL_MOOT_NOTE,
        "rules": {},
    }
    for rule in rules:
        family = ota.CONTROL_FAMILIES[rule.name]
        columns: list[dict[str, Any]] = []
        selection: dict[str, Any] = {}
        identity: dict[str, Any] = {}
        for index, pair in enumerate(pairs):
            base, tuned = require_pair(table, pair)
            scored: dict[int, list[dict[str, Any]]] = {}
            retained: dict[int, set[str]] = {}
            draws: dict[int, Any] = {}
            per_stream: dict[str, Any] = {}
            for position, stream in enumerate(sorted(base)):
                folded = [row for row in base[stream]["rows"] if row["plddt"] is not None]
                scoring = ota.score_pool(folded, rule)
                scored[stream] = scoring["scored"]
                retention = ota.rank_and_retain(scored[stream], rule)
                retained[stream] = set(retention.pop("retained_keys"))
                try:
                    draws[stream] = ota.nested_base_draws(
                        scored[stream], retain=rule.retain, family=family, column=index,
                        stream_index=position)
                except ValueError as error:
                    fail(f"{pair.base} stream {stream} under {rule.name}: {error}")
                    raise
                per_stream[str(stream)] = {
                    **retention, "n_folded": len(folded),
                    "n_unscorable": scoring["n_unscorable"],
                    "unscorable_keys": scoring["unscorable_keys"],
                    "tuned_folded_pool": (
                        sum(1 for row in tuned[stream]["rows"] if row["plddt"] is not None)
                        if stream in tuned else None),
                }
            selection[pair.tuned] = per_stream
            identity[pair.tuned] = {}
            for stratum in ota.STRATA:
                base_pool = scored_by_stratum(scored, stratum)
                filtered_pool = scored_by_stratum(scored, stratum, retained)
                tuned_pool = plddt_by_stratum(tuned, stratum)
                short = {"base": ota.pool_shortfall(base_pool),
                         "filtered": ota.pool_shortfall(filtered_pool),
                         "tuned": ota.pool_shortfall(tuned_pool)}
                key = f"{pair.tuned}|{stratum}"
                column: dict[str, Any] = {
                    "key": key,
                    "primary": primary["bands"]["columns"].get(key),
                    "units": {"base": ota.pool_units(base_pool),
                              "filtered": ota.pool_units(filtered_pool),
                              "tuned": ota.pool_units(tuned_pool)},
                    "base": {stream: draws[stream][stratum]["base"] for stream in draws},
                    "filtered": {stream: draws[stream][stratum]["filtered"] for stream in draws},
                    "valid": {stream: draws[stream][stratum]["valid"] for stream in draws},
                    "tuned": tuned_pool,
                }
                if any(short.values()):
                    column["skip_reason"] = (f"below the resampling floor on stream(s) {short}; "
                                             + ota.ADMISSIBILITY_RULE)
                else:
                    column.update(base_realised=ota.pool_mean(base_pool),
                                  filtered_realised=ota.pool_mean(filtered_pool),
                                  tuned_realised=ota.pool_mean(tuned_pool))
                    identity[pair.tuned][stratum] = identity_levels(base, tuned, retained, stratum)
                columns.append(column)
        try:
            band = ota.reproduced_fraction_bands(columns, family=family)
        except (ValueError, KeyError) as error:
            fail(str(error))
            raise
        out["rules"][rule.name] = {
            "family": family, "retain": rule.retain, "statistic_units": rule.units,
            "provenance": rule.provenance, "bands": band,
            "selection": selection, "identity_axis": identity,
        }
    return out


def identity_shift(table: dict[str, Any], pairs: tuple[ot.ObjectiveTunedPair, ...]) -> dict[str, Any]:
    """The direct narrowing readout: where the identity distribution moved.

    Two families, never merged: mean percent identity over the query, and the
    fraction of retained attempts below the declared novel edge. They are in
    different units, and a maximum-statistic band in native units may not span
    two units.
    """

    def columns(endpoint: str) -> list[dict[str, Any]]:
        built = []
        for pair in pairs:
            base, tuned = require_pair(table, pair)
            entry: dict[str, Any] = {"key": pair.tuned}
            for side, pools in (("base", base), ("tuned", tuned)):
                if endpoint == "mean":
                    entry[side] = {stream: [row["identity"] for row in pool["rows"]
                                            if row["identity"] is not None]
                                   for stream, pool in sorted(pools.items())}
                else:
                    entry[side] = {stream: [float(row["identity"] is not None
                                                  and row["identity"] < ota.NOVEL_IDENTITY_EDGE)
                                            for row in pool["rows"]]
                                   for stream, pool in sorted(pools.items())}
            incomplete = sorted(f"{pool['arm']}/s{stream}"
                                for pools in (base, tuned) for stream, pool in pools.items()
                                if pool["counts"]["identity_search_coverage"] < 1.0)
            if incomplete:
                entry["skip_reason"] = (
                    "the novelty search does not cover every non-empty attempt of "
                    f"{incomplete}; an identity distribution read off a partial search is "
                    "biased by whatever selected the searched subset")
            built.append(entry)
        return built

    shares: dict[str, Any] = {}
    for pair in pairs:
        for arm in pair.arms:
            pools = pool_of(table, arm)
            searched = sum(pool["counts"]["searched"] for pool in pools.values())
            per = {}
            for stratum in ota.STRATA:
                rows = [row for pool in pools.values() for row in pool["rows"]
                        if row["stratum"] == stratum]
                folded = [row["plddt"] for row in rows if row["plddt"] is not None]
                per[stratum] = {"n": len(rows),
                                "share": (len(rows) / searched) if searched else None,
                                "mean": float(np.mean(folded)) if folded else None}
            shares[arm] = {"searched": searched, "per_stratum": per}
    decomposition = {}
    for pair in pairs:
        left, right = shares[pair.base]["per_stratum"], shares[pair.tuned]["per_stratum"]
        common = sorted(name for name in ota.STRATA
                        if left[name]["share"] is not None and left[name]["mean"] is not None
                        and right[name]["share"] is not None and right[name]["mean"] is not None)
        decomposition[pair.tuned] = (
            ota.shift_share({name: left[name] for name in common},
                            {name: right[name] for name in common}) if common else
            {"reason": "no stratum carries a share and a mean on both arms"})
    return {
        "mean_identity": bands(columns("mean"), ota.IDENTITY_MEAN_FAMILY),
        "fraction_below_30": bands(columns("fraction"), ota.IDENTITY_FRACTION_FAMILY),
        "units": {"mean_identity": "percent identity over the query",
                  "fraction_below_30": "fraction of retained attempts"},
        "stratum_shares": shares,
        "shift_share": decomposition,
        "reading": ("a gain in objective attainment together with a move away from the record is "
                    "the strongest positive for design bias; a move toward it is narrowing"),
    }


def recognition(table: dict[str, Any], pairs: tuple[ot.ObjectiveTunedPair, ...],
                path: Path | None) -> dict[str, Any]:
    """Pfam recognition rates, tuned minus base, on the two declared endpoints."""

    endpoints = ("any_family", "complete_domain")
    if path is None:
        return {"status": "not_measured",
                "reason": "no --recognition table was supplied; family recognition is an hmmscan "
                          "endpoint and is never inferred from a sequence here",
                "endpoints": endpoints,
                "complete_domain_coverage": ota.COMPLETE_DOMAIN_COVERAGE}
    flags: dict[str, dict[str, bool]] = {}
    for row in read_jsonl(path):
        digest = digest_of(row, path, "recognition row")
        for key in endpoints:
            if key not in row:
                fail(f"{path}: {digest} carries no {key}")
        flags[digest] = {key: bool(row[key]) for key in endpoints}

    def columns(endpoint: str) -> list[dict[str, Any]]:
        built = []
        for pair in pairs:
            base, tuned = require_pair(table, pair)
            entry: dict[str, Any] = {"key": pair.tuned}
            missing = []
            for side, pools in (("base", base), ("tuned", tuned)):
                entry[side] = {stream: [float(flags.get(row["digest"], {}).get(endpoint, False))
                                        for row in pool["rows"]]
                               for stream, pool in sorted(pools.items())}
                missing.extend(f"{pool['arm']}/s{stream}" for stream, pool in pools.items()
                               if any(row["digest"] not in flags for row in pool["rows"]
                                      if not row["empty"]))
            if missing:
                entry["skip_reason"] = ("the recognition table does not cover every non-empty "
                                        f"attempt of {sorted(set(missing))}")
            built.append(entry)
        return built

    return {"status": "measured", "table": str(path), "table_sha256": sha256_file(path),
            "oracle": "hmmscan at the release's gathering thresholds (--cut_ga)",
            "complete_domain_coverage": ota.COMPLETE_DOMAIN_COVERAGE,
            "units": "fraction of retained attempts",
            "any_family": bands(columns("any_family"), ota.RECOGNITION_ANY_FAMILY),
            "complete_domain": bands(columns("complete_domain"), ota.RECOGNITION_COMPLETE_FAMILY),
            "prediction": "narrowing predicts recognition rises toward the natural reference",
            "natural_reference": "not supplied; the length-matched natural fragment reference is "
                                 "its own recognition run and is not inferred here"}


def selector_competence(table: dict[str, Any], path: Path | None) -> dict[str, Any]:
    """Oracle headroom per arm, and the share of it a supplied selector captures."""

    scores: dict[tuple[str, str], float] = {}
    if path is not None:
        for row in read_jsonl(path):
            for key in ("arm", "score"):
                if key not in row:
                    fail(f"{path}: a selector row carries no {key}")
            scores[(str(row["arm"]), digest_of(row, path, "selector row"))] = float(row["score"])
    per_arm = {}
    for arm in sorted({pool["arm"] for pool in table["pools"].values()}):
        folded = [row for pool in pool_of(table, arm).values() for row in pool["rows"]
                  if row["plddt"] is not None]
        values = [row["plddt"] for row in folded]
        entry: dict[str, Any] = {
            "n_pool": len(folded),
            "oracle_gain": ota.oracle_headroom(values),
            "oracle_statement": ota.ORACLE_STATEMENT,
            "pool_mean_plddt": float(np.mean(values)) if values else None,
        }
        if path is None:
            entry["captured"] = {
                "status": "not_measured",
                "reason": "the free-length lane records no per-sequence likelihood and no "
                          "--selector-score was supplied; the arm's own likelihood over its own "
                          "pool is a separate scoring step and is not fabricated here"}
        elif any((arm, row["digest"]) not in scores for row in folded):
            entry["captured"] = {"status": ota.UNRESOLVED,
                                 "reason": "the selector score does not cover every folded "
                                           f"attempt of {arm}"}
        else:
            entry["captured"] = ota.captured_fraction(
                values, [scores[(arm, row["digest"])] for row in folded])
        per_arm[arm] = entry
    return {"budget": ota.SELECTION_BUDGET, "units": "pLDDT points; the fraction is dimensionless",
            "selector_score": None if path is None else str(path),
            "endpoint": "the fraction of oracle pLDDT headroom a selector captures within its own "
                        "pool, on analyse_e20_confirmatory's definitions",
            "per_arm": per_arm}


# -------------------------------------------------------- the within-arm lane

def within_arm_reading(novelty_dir: Path | None, conditioned_root: Path | None) -> dict[str, Any]:
    """The amendment's reading of the ZymCTRL pair: within-arm identity only.

    The conditioned-generation artefacts are the population -- arm by condition,
    with the mismatched-label negative beside the native request -- and the
    frozen novelty search is the identity instrument. They are joined on the
    sequence digest, because the conditioned artefacts carry samples, not ids.

    A distribution is reported as an estimate only where the search covers the
    whole condition and clears the resampling floor. A within-arm *distribution*
    read off part of a pool is biased by whatever selected that part, so a
    partial search is recorded with its coverage and named unresolved rather than
    averaged. No matched structural contrast is formed for this pair at all.
    """

    arms = sorted({arm for pair in ota.WITHIN_ARM_PAIRS for arm in pair.arms})
    base: dict[str, Any] = {
        "pairs": [pair.tuned for pair in ota.WITHIN_ARM_PAIRS],
        "arms": arms,
        "structural_contrast": {"finding": ota.NON_IDENTIFIABLE,
                                "reason": ota.ZYMCTRL_NON_IDENTIFIABLE},
        "endpoint": "the within-arm distribution of identity to the nearest natural sequence, "
                    "under native EC conditioning, with the mismatched-label negative beside it",
        "units": "percent identity over the query against UniRef50, masking off",
        "coverage_rule": "an estimate needs the search to cover the whole condition and to clear "
                         f"the {ota.MIN_UNITS_PER_POOL}-unit resampling floor",
        "population_note": ota.WITHIN_ARM_POPULATION_NOTE,
    }
    if novelty_dir is None or conditioned_root is None:
        return {**base, "status": "not_measured",
                "reason": "--within-arm-novelty and --conditioned-root are both needed: the "
                          "identity distribution comes from the frozen novelty search and its "
                          "population from the conditioned-generation artefacts"}
    sources = novelty_sources([novelty_dir])
    searched = read_novelty(sources)

    per_arm: dict[str, Any] = {}
    matched_total = 0
    for arm in arms:
        path = conditioned_root / f"generations_{arm}.json"
        if not path.is_file():
            fail(f"--conditioned-root {conditioned_root} holds no {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        population: dict[str, dict[str, Any]] = {}
        for cell in (payload.get("cells") or {}).values():
            block = population.setdefault(str(cell["condition"]), {"samples": 0, "digests": set()})
            for sample in cell["samples"]:
                block["samples"] += 1
                block["digests"].add(sequence_digest(str(sample)))
        conditions: dict[str, Any] = {}
        for condition, block in sorted(population.items()):
            hits = [searched[digest] for digest in sorted(block["digests"]) if digest in searched]
            matched_total += len(hits)
            coverage = len(hits) / len(block["digests"]) if block["digests"] else 0.0
            entry: dict[str, Any] = {
                "generated_samples": block["samples"],
                "distinct_sequences": len(block["digests"]),
                "searched": len(hits),
                "search_coverage": coverage,
            }
            if hits:
                values = np.asarray([hit["identity"] for hit in hits], dtype=float)
                entry.update(
                    mean_identity=float(values.mean()),
                    median_identity=float(np.median(values)),
                    fraction_below_30=float(np.mean(values < ota.NOVEL_IDENTITY_EDGE)),
                    stratum_shares={name: float(np.mean([hit["stratum"] == name for hit in hits]))
                                    for name in ota.STRATA})
            if coverage >= 1.0 and len(hits) >= ota.MIN_UNITS_PER_POOL:
                entry.update(status="measured", reason=None)
            else:
                entry.update(status=ota.UNRESOLVED,
                             reason=f"the frozen novelty search covers {len(hits)} of "
                                    f"{len(block['digests'])} distinct sequences of this "
                                    "condition. The amendment's within-arm reading needs identity "
                                    "over the query across the condition; that search is CPU-only "
                                    "and adds no generation, and the numbers here are descriptive "
                                    "counts rather than an estimate")
            conditions[condition] = entry
        per_arm[arm] = {"provenance": {"path": str(path), "sha256": sha256_file(path)},
                        "conditions": conditions}
    if matched_total == 0:
        fail(f"no sequence of {arms} in {conditioned_root} appears in the novelty search under "
             f"{novelty_dir}; the supplied search is not of this pair's generations")
    return {**base, "status": "measured",
            "novelty_sources": [str(record) for record, _ in sources],
            "join": "sequence digest, because the conditioned artefacts carry samples not ids",
            "per_arm": per_arm}


# ------------------------------------------------------------------------ main

def run(args: argparse.Namespace) -> dict[str, Any]:
    campaign = read_campaign(args.campaign_root)
    structure = read_structure(expand(args.structure, "predictions.jsonl"))
    novelty = read_novelty(novelty_sources(args.novelty))
    table = assemble(campaign, structure, novelty)
    counts = gc.load_kmer_counts(args.kmer_background, KMER_ORDERS)

    layout = {f"{pool['arm']}|s{pool['stream']}":
              {**pool["counts"], "role": pool["role"], "pair": pool["pair"],
               "cells": pool["cells"], "seeds": pool["seeds"],
               "checkpoints": pool["checkpoints"]}
              for pool in table["pools"].values()}
    layout = {key: layout[key] for key in sorted(layout)}
    if args.dry_run:
        print(json.dumps({"cells": layout, "join_rule": table["join_rule"],
                          "denominator_rule": table["denominator_rule"]},
                         indent=2, sort_keys=True))
        return {"status": "dry_run", "cells": layout}

    present = {pool["arm"] for pool in table["pools"].values()}
    secondary = tuple(pair for pair in ota.SECONDARY_PAIRS if set(pair.arms) <= present)
    # The identity shift is read before the primary contrast, because the
    # registration's second narrowing route is a statement about both; and the
    # primary contrast before the control, because it decides whether each of
    # the control's cells has a gain to be a share of.
    shift = identity_shift(table, ota.PRIMARY_PAIRS)
    primary = stratified_contrast(table, ota.PRIMARY_PAIRS, ota.PRIMARY_FAMILY, shift=shift)
    record = {
        "schema": ota.SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "docs/objective-alignment.md, declared 2026-11-09, amended 2026-11-10",
        "campaign": ot.CAMPAIGN,
        "campaign_root": campaign["root"],
        "cells": layout,
        "join_rule": table["join_rule"],
        "denominator_rule": table["denominator_rule"],
        "novelty_digests_outside_the_campaign": table["novelty_digests_outside_the_campaign"],
        "kmer_background": {"path": str(args.kmer_background), "orders_read": list(KMER_ORDERS)},
        "primary": primary,
        "explicit_statistics_control": explicit_statistics_control(
            table, ota.PRIMARY_PAIRS, counts, primary),
        "identity_shift": shift,
        "recognition": recognition(table, ota.PRIMARY_PAIRS, args.recognition),
        "selector_competence": selector_competence(table, args.selector_score),
        "prediction_side_effect": {"status": "not_measured",
                                   "reason": ota.LADDER_NOT_MEASURABLE},
        "within_arm": within_arm_reading(args.within_arm_novelty, args.conditioned_root),
        "secondary_pair": (stratified_contrast(table, secondary, ota.SECONDARY_FAMILY)
                           if secondary else
                           {"status": "not_measured",
                            "reason": "the declared narrowing control is not in this campaign "
                                      "root; it is reported beside the primary family and never "
                                      "merged into it"}),
        "finding_states": ota.FINDING_STATES,
        "bounds": ota.BOUNDS,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / COMPLETION, record)
    return record


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-root", type=Path, required=True,
                        help="the run directory holding the ot_<arm>_s<stream>_r<replicate> cells")
    parser.add_argument("--structure", type=Path, nargs="+", required=True,
                        help="predictions.jsonl files, or directories holding them")
    parser.add_argument("--novelty", type=Path, nargs="+", required=True,
                        help="novelty-stage output directories (generated_novelty.json beside "
                             "its sidecar), whose records pin the cohort that carries the digest")
    parser.add_argument("--kmer-background", type=Path, required=True,
                        help="corpus k-mer counts for the composition and Markov rules")
    parser.add_argument("--recognition", type=Path, default=None,
                        help="JSONL of sequence (or sequence_sha256), any_family, complete_domain")
    parser.add_argument("--selector-score", type=Path, default=None,
                        help="JSONL of arm, sequence (or sequence_sha256), score")
    parser.add_argument("--within-arm-novelty", type=Path, default=None,
                        help="the within-arm pair's frozen novelty-stage output directory")
    parser.add_argument("--conditioned-root", type=Path, default=None,
                        help="the EXP-R2-227 conditioned-generation results directory")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print the realised cell layout, writing nothing")
    args = parser.parse_args(argv)
    if args.out is None and not args.dry_run:
        parser.error("--out is required unless --dry-run")
    if (args.within_arm_novelty is None) != (args.conditioned_root is None):
        parser.error("--within-arm-novelty and --conditioned-root are supplied together")
    record = run(args)
    if not args.dry_run:
        print(json.dumps({"status": record["status"], "out": str(args.out / COMPLETION),
                          "primary_declared": record["primary"]["bands"]["declared_family_size"],
                          "primary_resolved": record["primary"]["bands"]["resolved_family_size"]},
                         sort_keys=True))
    return record


if __name__ == "__main__":
    main()
