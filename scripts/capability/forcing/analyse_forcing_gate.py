#!/usr/bin/env python3
"""E19 gate: the pair-resolved double difference, its calibration and its verdict.

A CPU stage. For every arm it pairs the two conditions of each unit, forms the
Jensen-Shannon divergence between the realised residue distributions at the
prescribed partner and at the matched non-contacting reference partners, and
reports the difference -- the endpoint whose null is exactly zero -- as a
percentile interval over backbones.

Everything the design named as a confound reaches the output as a declared
variant or a stratum rather than as a sentence: completed-only against
all-draws, with and without repeat-flagged draws, with and without the
near-duplicate grouping, within hydrophobic-fraction bins, and stratified by
length band, anchor burial band, separation stratum and -- when the coverage
sidecar is supplied -- reference-database identity band. The permutation
calibration measures the estimator's own residual bias by permuting which draws
of a cell are called native, which realises the null rather than arguing it.

The verdict is read against the effect size pre-registered before the run, and
the structure channel opens only if the gate resolves positive.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.forcing import forcing_design as D  # noqa: E402
from src.capability.forcing import gate_divergence as G  # noqa: E402
from src.capability.forcing.backbone_cohort import load_cohort  # noqa: E402

COMPLETION = "forcing_gate.json"
STRATA = ("length_band", "anchor_rsa_band", "separation_stratum", "coverage_band")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_arm(directory: Path, *, cohort_sha256: str) -> dict:
    """One arm's generation products, with its per-draw distributions attached.

    The cohort digest is checked rather than trusted. A unit id names an anchor
    and a partner, so analysing an arm against a different cohort build than it
    was generated from would silently read the right draws at the wrong
    positions, and nothing downstream could notice.
    """

    summary = json.loads((directory / "forcing_generation.json").read_text())
    if summary.get("status") != "complete":
        raise SystemExit(f"{directory}: generation is not complete")
    if summary["cohort"]["sha256"] != cohort_sha256:
        raise SystemExit(
            f"{directory} was generated from a cohort digesting to "
            f"{summary['cohort']['sha256'][:12]} and is being analysed against "
            f"{cohort_sha256[:12]}; a unit id names positions, so this would read the "
            "right draws at the wrong positions"
        )
    records: dict[tuple[str, str, str], dict] = {}
    for line in (directory / summary["journal"]).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        array = directory / summary["objects"] / f"{record['cell']}.npy"
        if not array.is_file():
            raise SystemExit(f"{directory}: cell {record['cell']} has no distribution array")
        record["distributions"] = np.load(array)
        records[(record["unit_id"], record["condition"], record["mode"])] = record
    return {"summary": summary, "records": records, "directory": str(directory)}


#: Fields two shards of one arm must agree on. A shard that differs on any of
#: them is not a shard of the same measurement, and merging it would pool two
#: quantities under one name.
SHARD_IDENTITY = ("arm", "token_grid", "dtype", "draws_per_cell", "sampling_seed", "decoding")


def read_arms(directories: list[Path], *, cohort_sha256: str) -> list[dict]:
    """Group the supplied directories by arm, merging shards of one arm.

    An arm large enough to set the campaign's wall clock is sharded over several
    cards, so several directories can carry one arm. They are merged here rather
    than entering the panel as separate columns -- two shards of one checkpoint
    are one measurement, and a panel that counted them twice would resample the
    same backbone under two names. A directory that disagrees with its siblings
    on the measurement's identity, or that repeats a cell, is refused.
    """

    groups: dict[str, dict] = {}
    for directory in directories:
        block = read_arm(directory, cohort_sha256=cohort_sha256)
        summary = block["summary"]
        name = str(summary["arm"])
        existing = groups.get(name)
        if existing is None:
            groups[name] = {
                "summary": dict(summary),
                "records": dict(block["records"]),
                "directories": [block["directory"]],
                "shards": [summary.get("shard")],
            }
            continue
        differing = [
            key for key in SHARD_IDENTITY
            if existing["summary"].get(key) != summary.get(key)
        ]
        if differing:
            raise SystemExit(
                f"{directory} claims arm {name!r} but disagrees with "
                f"{existing['directories'][0]} on {differing}; these are not shards of one "
                "measurement and are not merged"
            )
        overlap = sorted(set(existing["records"]) & set(block["records"]))
        if overlap:
            raise SystemExit(
                f"{directory} repeats {len(overlap)} cell(s) already carried by "
                f"{existing['directories'][0]}, e.g. {overlap[:3]}; overlapping shards would "
                "double-count draws"
            )
        existing["records"].update(block["records"])
        existing["directories"].append(block["directory"])
        existing["shards"].append(summary.get("shard"))
        censoring = existing["summary"].get("censoring") or {}
        incoming = summary.get("censoring") or {}
        for key in ("censored_draws", "total_draws", "cells_with_any_censoring"):
            censoring[key] = int(censoring.get(key, 0)) + int(incoming.get(key, 0))
        existing["summary"]["censoring"] = censoring
        existing["summary"]["cells"] = len(existing["records"])
    return [groups[name] for name in sorted(groups)]


def paired_cells(arm: dict, cohort: dict, *, mode: str, coverage: dict) -> list[dict]:
    """Join the two conditions of every unit this arm completed in one mode."""

    by_unit = {unit["unit_id"]: unit for unit in cohort["unit_rows"]}
    cells = []
    for unit_id, unit in sorted(by_unit.items()):
        native = arm["records"].get((unit_id, D.CONDITION_NATIVE, mode))
        transplant = arm["records"].get((unit_id, D.CONDITION_TRANSPLANT, mode))
        if native is None or transplant is None:
            continue
        cells.append(
            {
                "unit_id": unit_id,
                "accession": unit["accession"],
                "length_band": unit["length_band"],
                "anchor_rsa_band": unit["anchor_rsa_band"],
                "anchor_ss_class": unit["anchor_ss_class"],
                "separation_stratum": unit["separation_stratum"],
                "coverage_band": coverage.get(unit["accession"]),
                "partner": int(unit["partner"]),
                "reference_positions": [
                    int(row["position"]) for row in unit["reference_partners"]
                ],
                "separation_imbalance": float(
                    int(unit["separation"])
                    - float(np.mean([row["separation"] for row in unit["reference_partners"]]))
                ),
                "native": native,
                "transplant": transplant,
            }
        )
    return cells


def _carry(records: list[dict], cells: list[dict]) -> list[dict]:
    """Attach each cell's stratum labels to its endpoint record, for stratification."""

    index = {cell["unit_id"]: cell for cell in cells}
    carried = []
    for record in records:
        cell = index[record["unit_id"]]
        carried.append(
            dict(record) | {key: cell.get(key) for key in STRATA} | {
                "anchor_ss_class": cell["anchor_ss_class"],
                "separation_imbalance": cell["separation_imbalance"],
            }
        )
    return carried


def analyse_arm(arm: dict, cohort: dict, *, coverage: dict, args: argparse.Namespace) -> dict:
    sampled = paired_cells(arm, cohort, mode=D.MODE_SAMPLED, coverage=coverage)
    if not sampled:
        return {
            "arm": arm["summary"]["arm"],
            "status": "no unit carries both conditions in the sampled mode",
        }
    block: dict = {
        "arm": arm["summary"]["arm"],
        "in_declared_panel": arm["summary"].get("in_declared_panel"),
        "token_grid": arm["summary"].get("token_grid"),
        "units": len(sampled),
        "backbones": len({cell["accession"] for cell in sampled}),
        "censoring": arm["summary"].get("censoring"),
        "separation_imbalance_residues": {
            "mean": float(np.mean([cell["separation_imbalance"] for cell in sampled])),
            "sd": float(np.std([cell["separation_imbalance"] for cell in sampled])),
            "max_abs": float(np.max(np.abs([cell["separation_imbalance"] for cell in sampled]))),
            "reads_as": (
                "the prescribed partner's separation minus its controls' mean separation. "
                "A large imbalance would make the endpoint a statement about sequence "
                "distance wearing a structural label"
            ),
        },
        "variants": {},
        "strata": {},
    }
    primary_values: dict[str, float] | None = None
    for estimator in G.ESTIMATORS:
        for name, settings in G.VARIANTS.items():
            records = [
                G.cell_endpoint(cell, estimator=estimator, **settings) for cell in sampled
            ]
            values = G.backbone_values(records)
            estimate = G.bootstrap_interval(values, draws=args.draws, seed=args.seed)
            block["variants"].setdefault(estimator, {})[name] = estimate | {
                "settings": dict(settings),
                "resolved_units": sum(1 for row in records if row["endpoint_nats"] is not None),
                "partner_divergence_nats": _mean_of(records, "partner_divergence_nats"),
                "reference_divergence_nats": _mean_of(records, "reference_divergence_nats"),
            }
            if estimator == "empirical" and name == "primary":
                primary_values = values
                block["strata"] = {
                    stratum: G.by_stratum(
                        _carry(records, sampled), stratum=stratum,
                        draws=args.draws, seed=args.seed,
                    )
                    for stratum in STRATA
                }
                block["hydrophobic_bins"] = G.bootstrap_interval(
                    G.backbone_values(
                        [G.hydrophobic_bin_endpoint(cell, estimator=estimator)
                         for cell in sampled]
                    ),
                    draws=args.draws, seed=args.seed,
                )
                block["calibration"] = G.permutation_calibration(
                    sampled, estimate.get("point"), estimator=estimator,
                    draws=args.permutation_draws, seed=D.PERMUTATION_SEED,
                )
    teacher = paired_cells(arm, cohort, mode=D.MODE_TEACHER_FORCED, coverage=coverage)
    if teacher:
        records = [G.teacher_forced_endpoint(cell) for cell in teacher]
        block["teacher_forced"] = {
            "units": len(records),
            "exact": G.bootstrap_interval(
                G.backbone_values(records, key="exact_nats"), draws=args.draws, seed=args.seed,
            ),
            "resampled_to_matched_support": G.bootstrap_interval(
                G.backbone_values(records, key="resampled_nats"),
                draws=args.draws, seed=args.seed,
            ),
            "reads_as": (
                "the same intervention with the residues between the anchor and the partner "
                "fixed to wild type. The sampled-minus-teacher-forced difference is how much "
                "of the effect travels through the model's own intervening commitments rather "
                "than through direct conditioning on the anchor; it is read against "
                "resampled_to_matched_support, which shares the sampled mode's estimator and "
                "support, not against exact, which carries no sampling bias at all"
            ),
        }
        shared = sorted(
            set(G.backbone_values(records, key="resampled_nats")) & set(primary_values or {})
        )
        if shared:
            teacher_values = G.backbone_values(records, key="resampled_nats")
            block["teacher_forced"]["sampled_minus_teacher_forced"] = G.bootstrap_interval(
                {
                    accession: float((primary_values or {})[accession] - teacher_values[accession])
                    for accession in shared
                },
                draws=args.draws, seed=args.seed,
            )
    block["verdict"] = G.gate_verdict(
        block["variants"]["empirical"]["primary"], block.get("calibration", {}),
    )
    block["backbone_values"] = {
        key: float(value) for key, value in (primary_values or {}).items()
    }
    return block


def _mean_of(records: list[dict], key: str) -> float | None:
    values = [row[key] for row in records if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def run(args: argparse.Namespace) -> None:
    cohort = load_cohort(args.cohort)
    coverage: dict = {}
    coverage_source = None
    if args.coverage is not None:
        payload = json.loads(args.coverage.read_text())
        coverage = {
            str(key): str(value) for key, value in payload["identity_band"].items()
        }
        coverage_source = {"path": str(args.coverage), "sha256": sha256_file(args.coverage)}

    arms = read_arms(list(args.generation), cohort_sha256=sha256_file(args.cohort))
    blocks = [analyse_arm(arm, cohort, coverage=coverage, args=args) for arm in arms]
    panel: dict = {
        "columns": [block["arm"] for block in blocks],
        "policy": (
            "a panel-wide statement needs a simultaneous band; marginal per-arm intervals "
            "are not simultaneous statements. Absolute endpoints are comparable across arms "
            "only because each is already a within-arm double difference"
        ),
    }
    usable = [
        block for block in blocks
        if block.get("backbone_values") and block.get("in_declared_panel")
    ]
    if len(usable) >= 2:
        universe = set.intersection(*(set(block["backbone_values"]) for block in usable))
        if len(universe) >= 2:
            restricted = {
                block["arm"]: {
                    key: value for key, value in block["backbone_values"].items()
                    if key in universe
                }
                for block in usable
            }
            panel["simultaneous"] = G.simultaneous_band(
                [block["arm"] for block in usable], restricted,
                draws=args.draws, seed=args.seed,
            )
            panel["shared_backbones"] = len(universe)
        else:
            panel["simultaneous"] = {
                "undefined": "fewer than two backbones are shared by every declared arm"
            }
    else:
        panel["simultaneous"] = {
            "undefined": f"{len(usable)} declared arm(s) carry a backbone value; a band needs two"
        }

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / COMPLETION, {
        "schema": G.SCHEMA,
        "status": "complete",
        "created_utc": _now(),
        "experiment": D.EXPERIMENT,
        "question": (
            "does forcing an early residue change what a generative protein model emits at a "
            "prescribed contacting position more than at matched non-contacting positions"
        ),
        "pre_registration": D.pre_registration(
            units=cohort["backbones"], pairs=cohort["units"],
        ),
        "estimator_semantics": {
            "empirical": (
                "the histogram of the residues the draws realised at the position; the "
                "pre-registered primary"
            ),
            "mixture": (
                "the mean over draws of the sampling distribution the model used at the "
                "position: the same marginal, estimated with the per-draw conditional "
                "instead of one sample from it, so it has the same expectation and less "
                "variance. Reported beside the primary, never instead of it"
            ),
        },
        "minimum_draws_per_condition": G.MIN_DRAWS_PER_CONDITION,
        "variant_semantics": {
            name: dict(settings) for name, settings in G.VARIANTS.items()
        },
        "confounds": dict(D.CONFOUNDS),
        "non_identifiable": dict(D.NON_IDENTIFIABLE),
        "coverage_stratification": (
            coverage_source if coverage_source else {
                "status": "absent",
                "consequence": (
                    "no reference-database coverage stratum is reported. The coverage band "
                    "is a declared confound control and its absence is a gap in the reading, "
                    "not a result: the remote and close bands are not pooled here because "
                    "they are not known here"
                ),
            }
        ),
        "sources": {
            "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
            "generation": [
                {
                    "arm": arm["summary"]["arm"],
                    "shards": arm["shards"],
                    "directories": [
                        {
                            "path": directory,
                            "sha256": sha256_file(Path(directory) / "forcing_generation.json"),
                        }
                        for directory in arm["directories"]
                    ],
                }
                for arm in arms
            ],
        },
        "cohort_summary": {
            "backbones": cohort["backbones"],
            "backbones_per_band": cohort["backbones_per_band"],
            "units": cohort["units"],
            "units_per_band": cohort["units_per_band"],
            "pairs_per_backbone": cohort["pairs_per_backbone"],
            "forced_residue_multisets_match": cohort["forced_residue_multisets"]["matches"],
        },
        "arms": blocks,
        "panel": panel,
        "verdicts": dict(Counter(
            block.get("verdict", {}).get("verdict", "unresolved") for block in blocks
        )),
        "interpretation": D.INTERPRETATION,
        "structure_channel": D.STRUCTURE_CHANNEL_CONTRACT,
    })
    print(json.dumps({
        "arms": [
            {
                "arm": block["arm"],
                "units": block.get("units"),
                "backbones": block.get("backbones"),
                "primary": block.get("variants", {}).get("empirical", {}).get("primary", {}),
                "verdict": block.get("verdict", {}).get("verdict"),
            }
            for block in blocks
        ],
        "artefact": str(args.out / COMPLETION),
    }, indent=1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--generation", type=Path, nargs="+", required=True)
    parser.add_argument("--coverage", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=D.BOOTSTRAP_DRAWS)
    parser.add_argument("--permutation-draws", type=int, default=D.PERMUTATION_DRAWS)
    parser.add_argument("--seed", type=int, default=D.BOOTSTRAP_SEED)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
