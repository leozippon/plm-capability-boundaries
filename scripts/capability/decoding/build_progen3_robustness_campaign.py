#!/usr/bin/env python3
"""Freeze the ProGen3 cross-model robustness protocol, its cell plan and its manifests.

This is a robustness check on E20's confirmatory conclusion, not a second sealed
campaign: it re-runs E20's own evaluation design on a newer autoregressive protein
language model, through E20's own stages, and changes nothing it does not have to.
What it has to change is recorded in two places and nowhere else -- the declared
amendment in ``scripts/capability/decoding/collect_guided_decoding_pool.py``
(``PROGEN3_AMENDMENT``: the dtype, because ProGen3 has no float32 kernel, and the
calibrated doses, because nominal lambda is not comparable across models) and the
``batch_size`` of 1 declared here, because ProGen3's expert mixture reduces over
the flattened batch and a wider extent is a different draw.

The design
==========

Three rungs, eight domains, five guidance arms, 200 candidates: 120 cells.

``progen3-3b`` in bfloat16 is the model under test. ``progen3-112m`` in bfloat16
is the second size. ``progen3-112m`` in float16 is a declared precision control at
the *same* doses as its bfloat16 twin, so precision is the only thing that moves
between them: the 3B's bfloat16 noise floor is the one measurement that could
drive a null result by itself, and the 112M is numerically sound in float16 where
the 3B is not, so this rung tests that directly. Each rung records its own model
label (``recorded_model_label``), so the three are three strata of one design.

The arms are native (N), profile (Pr), Potts fields only (M), coupling only (C)
and permuted-coupling (Cx), the matched incorrect-constraint control, at E20's own
permutation seed so the control is the same wrong constraint E20 used. The domains
are E20's eight sealed evaluation domains; the domain is the independence unit and
eight meets the bootstrap floor.

Three protocol files, one per rung
==================================

The dtype and the admitted dose axis are both properties of the rung, and a
protocol carries one ``native_contract`` and one ``strength_grid``. Each rung
therefore gets its own file; all three declare the identical ``strengths``,
``seed_streams``, ``domains``, ``roles``, ``permutation_seed``,
``sequences_per_cell`` and ``sampling``, so any one of them is a complete
declaration for the merge and the measurement, and the three differ only in
``model_arms``, ``native_contract.dtype`` and ``strength_grid``.

Two terminal lines, because a precision is a native contract
============================================================

``merge_e20_cells.py`` refuses cells that do not share one native contract, which
is right: a pool sampled under a different kernel is a different cell. The
bfloat16 rungs therefore merge, fold and measure together and the float16 rung
does the same separately. Both lines read the same frozen wild-type references and
the same ESMFold2 instrument and settings, and both run ``--role evaluation``.

Manifests
=========

``campaign_<name>_generate.tsv`` holds the 120 generation cells, six per slot on
the six free GPUs; ``campaign_<name>_fold_<precision>.tsv`` holds that precision's
merge, cohort, six fold shards, structure merge and measurement.
``scripts/capability/decoding/run_local_gpu_manifest.py`` executes them and
injects ``--device`` and ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.generation import fold_recapitulation as fr  # noqa: E402

CAMPAIGN = "progen3_robustness_20261015"
SCHEMA = "coupling_guided_generation_v1"
KEY = "pg3rb"

#: The rungs, in report order: ``(model arm, dtype)``. The float16 112M is the declared
#: precision control and runs at the bfloat16 doses of the same checkpoint.
RUNGS: tuple[tuple[str, str], ...] = (
    ("progen3-3b", "bfloat16"),
    ("progen3-112m", "bfloat16"),
    ("progen3-112m", "float16"),
)
#: Which rung's calibrated doses each rung is scheduled at. A rung maps to itself except
#: the precision control, which maps to its own bfloat16 twin so that the dose is held
#: fixed and the kernel is the only difference.
DOSE_SOURCE: dict[tuple[str, str], str] = {
    ("progen3-3b", "bfloat16"): "progen3-3b",
    ("progen3-112m", "bfloat16"): "progen3-112m",
    ("progen3-112m", "float16"): "progen3-112m",
}

#: ``contract code -> guidance arm``, in report order. Five arms: the factorial's native,
#: fields-only and coupling-only corners, the conventional profile baseline, and the
#: matched incorrect-constraint control. ``F`` and ``Cd`` are outside this design.
ARMS: dict[str, str] = {
    "N": "native", "Pr": "profile", "M": "potts_fields", "C": "coupling",
    "Cx": "permuted_coupling",
}
DOMAINS: tuple[str, ...] = ("1E0L", "1F0M", "1IFY", "1V1C", "2K1B", "2LGW", "2LVN", "3DKM")
POOL_SIZE = 200

#: Seed streams. Base and step are chosen to be disjoint from, and nowhere near, every
#: E20 campaign: the development family occupies [20261013000, 20261268400] (256 cells,
#: depth 400) and the confirmatory campaign [20262013000, 20262236200] (224 cells, depth
#: 200). 120 cells at 1000 apart from this base occupy [20263015000, 20263134200], which
#: is 778800 clear of the nearest E20 seed. ``validate_protocol`` re-proves the
#: disjointness and non-adjacency of the streams within this protocol from the file.
SEED_BASE, SEED_STEP = 20263015000, 1000

#: Six free local cards. Indices 6 and 7 carry another user's allocation and are never
#: addressed by this campaign.
GPUS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
FOLD_SHARDS = 6

PRELUDE = "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8"


def tag(value: float) -> str:
    """The dose as it appears in a cell label: hundredths, zero-padded to three digits.

    E20's own labelling rule. The calibrated doses are not round, so the label is a
    rounded identifier and never the dose itself; the dose lives in the protocol and in
    every completion record.
    """

    return f"{round(value * 100):03d}"


def stage_module() -> Any:
    path = ROOT / "scripts/capability/decoding/collect_guided_decoding_pool.py"
    spec = importlib.util.spec_from_file_location("guided_pool_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_of(model_arm: str, dtype: str) -> str:
    return stage_module().recorded_model_label(model_arm, dtype)


def doses(stage: Any, rung: tuple[str, str]) -> dict[str, tuple[float, float]]:
    """``code -> (lambda_fields, lambda_coupling)`` of one rung, from the amendment alone."""

    declared = stage.PROGEN3_AMENDMENT["doses"][DOSE_SOURCE[rung]]
    out = {"N": (0.0, 0.0)}
    for code in ARMS:
        if code == "N":
            continue
        if code not in declared:
            raise ValueError(
                f"{rung}: the amendment declares no dose for {code}; a cell cannot be "
                "scheduled at a dose nothing calibrated for it")
        out[code] = tuple(float(value) for value in declared[code])
    return out


def cell_plan(stage: Any) -> list[dict[str, Any]]:
    """Every cell in manifest order, with its stream index and seed."""

    cells: list[dict[str, Any]] = []
    for model_arm, dtype in RUNGS:
        rung_doses = doses(stage, (model_arm, dtype))
        for domain in DOMAINS:
            for code in ARMS:
                fields, coupling = rung_doses[code]
                cells.append({
                    "model_arm": model_arm, "dtype": dtype,
                    "model_label": label_of(model_arm, dtype), "domain": domain,
                    "arm_code": code, "guidance_arm": ARMS[code],
                    "lambda_fields": fields, "lambda_coupling": coupling,
                    "pool_size": POOL_SIZE,
                })
    for index, cell in enumerate(cells):
        cell["stream_index"] = index
        cell["stream_seed"] = SEED_BASE + SEED_STEP * index
    return cells


def label(cell: dict[str, Any]) -> str:
    return (f"{KEY}_gen_{cell['model_label']}_{cell['domain']}_{cell['arm_code']}"
            f"_f{tag(cell['lambda_fields'])}_c{tag(cell['lambda_coupling'])}")


def strengths(stage: Any) -> list[dict[str, Any]]:
    """The per-(model label, arm) frozen-strength declaration every protocol file carries.

    One frozen strength per (label, arm) and no dose-response strengths: this check
    reads each arm once at the dose calibrated for it, so there is no dose axis to be
    descriptive about and nothing but the frozen pair can reach a contrast.
    """

    out = []
    for model_arm, dtype in RUNGS:
        rung_doses = doses(stage, (model_arm, dtype))
        for code in ARMS:
            fields, coupling = rung_doses[code]
            out.append({"model": label_of(model_arm, dtype), "arm": code,
                        "frozen_strength": {"lambda_fields": fields, "lambda_coupling": coupling,
                                            "pool_size": POOL_SIZE}})
    return out


def protocol(stage: Any, rung: tuple[str, str], cells: list[dict[str, Any]],
             references: Path, roles: dict[str, Any]) -> dict[str, Any]:
    model_arm, dtype = rung
    rung_doses = doses(stage, rung)
    domains = {}
    for name in DOMAINS:
        side = json.loads((references / f"{name}.json").read_text())
        domains[name] = {"L": side["length"],
                         "wt_sha256": hashlib.sha256(side["wildtype"].encode()).hexdigest()}
    grid: list[dict[str, float]] = []
    for code in ARMS:
        fields, coupling = rung_doses[code]
        point = {"lambda_fields": fields, "lambda_coupling": coupling}
        if point not in grid:
            grid.append(point)
    return {
        "schema": SCHEMA,
        "phase": "robustness",
        "model_arms": [model_arm],
        "guidance_arms": [ARMS[code] for code in ARMS],
        "domains": domains,
        "strength_grid": grid,
        "sequences_per_cell": POOL_SIZE,
        "seed_streams": [cell["stream_seed"] for cell in cells],
        "permutation_seed": 20261013,
        "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "batch_size": 1},
        "native_contract": {
            "batch_size": 1, "budget": 1024, "dtype": dtype,
            "score": "native scored-span full-sum log-likelihood (-nll_sum)"},
        "strengths": strengths(stage),
        "roles": roles,
        "amendment": stage.PROGEN3_AMENDMENT["id"],
    }


def protocol_name(rung: tuple[str, str]) -> str:
    return f"guided_generation_protocol_{rung[0]}_{rung[1]}.json"


def row(slot: Any, gpu: Any, stage_name: str, cell_label: str, env: str, expect: str,
        args: str) -> str:
    return "\t".join((str(slot), KEY, str(gpu), stage_name, cell_label, env, expect, args)) + "\n"


def generate_manifest(cells: list[dict[str, Any]], data: Path, references: Path,
                      env: str) -> str:
    grids: dict[tuple[str, str], list[dict[str, float]]] = {}
    out = [
        f"# {CAMPAIGN}_generate: {len(cells)} cells (3 rungs x {len(DOMAINS)} domains x "
        f"{len(ARMS)} arms, {POOL_SIZE} candidates each), {len(GPUS)} per slot on GPUs "
        f"{','.join(str(g) for g in GPUS)}. --device and --out are injected.\n",
        "# slot\tkey\tgpu\tstage\tlabel\tenv\texpect\targs\n",
    ]
    for index, cell in enumerate(cells):
        rung = (cell["model_arm"], cell["dtype"])
        if rung not in grids:
            grids[rung] = json.loads(
                (data / protocol_name(rung)).read_text())["strength_grid"]
        point = {"lambda_fields": cell["lambda_fields"], "lambda_coupling": cell["lambda_coupling"]}
        args = (f"--protocol {data / protocol_name(rung)} --family-reference-dir {references} "
                f"--model-arm {cell['model_arm']} --domain {cell['domain']} "
                f"--guidance-arm {cell['guidance_arm']} "
                f"--strength-index {grids[rung].index(point)} "
                f"--stream-index {cell['stream_index']}")
        out.append(row(index // len(GPUS) + 1, GPUS[index % len(GPUS)],
                       "collect_guided_decoding_pool.py", label(cell), env,
                       "guided_decoding_pool.json", args))
    return "".join(out)


def fold_manifest(cells: list[dict[str, Any]], dtype: str, data: Path, references: Path,
                  wild_type_fasta: Path, msa_dir: Path, wild_type_references: Path,
                  esmfold2: Path, run: str, env_ct: str, env_fold: str) -> str:
    rungs = [rung for rung in RUNGS if rung[1] == dtype]
    selected = [cell for cell in cells if cell["dtype"] == dtype]
    proto = data / protocol_name(rungs[0])
    prefix = f"{KEY}_{dtype}"
    dirs = " ".join(f"{run}/cells/{label(cell)}" for cell in selected)
    out = [
        f"# {CAMPAIGN}_fold_{dtype}: the {dtype} rungs "
        f"({', '.join(label_of(*rung) for rung in rungs)}), {len(selected)} cells. "
        f"merge -> cohort -> fold on {FOLD_SHARDS} cards -> merge -> measure --role evaluation. "
        "A precision is part of the native contract, so each precision is its own terminal line.\n",
        "# slot\tkey\tgpu\tstage\tlabel\tenv\texpect\targs\n",
    ]
    out.append(row(1, "cpu", "merge_e20_cells.py", f"{prefix}_records_cpu", env_ct,
                   "e20_merged_records.json",
                   f"--cells {dirs} --expect-cells {len(selected)} "
                   f"--wild-type-fasta {wild_type_fasta} --protocol {proto} --role evaluation"))
    out.append(row(2, "cpu", "build_e20_fold_cohort.py", f"{prefix}_cohort_cpu", env_ct,
                   "e20_fold_cohort.json",
                   f"--records {run}/{prefix}_records_cpu/records.jsonl "
                   f"--wild-type-fasta {wild_type_fasta}"))
    for shard in range(FOLD_SHARDS):
        out.append(row(3, GPUS[shard], "run_structure_evidence.py",
                       f"{prefix}_fold_shard{shard}_gpu{GPUS[shard]}", env_fold,
                       "structure_evidence.json",
                       f"--cohort {run}/{prefix}_cohort_cpu/fold_cohort.jsonl "
                       f"--model {esmfold2} --shard-index {shard} --num-shards {FOLD_SHARDS}"))
    shards = " ".join(f"{run}/{prefix}_fold_shard{shard}_gpu{GPUS[shard]}"
                      for shard in range(FOLD_SHARDS))
    out.append(row(4, "cpu", "merge_e20_cells.py", f"{prefix}_structures_cpu", env_ct,
                   "e20_merged_structures.json", f"--shards {shards}"))
    out.append(row(5, "cpu", "measure_fold_recapitulation.py", f"{prefix}_measure_cpu", env_ct,
                   "fold_recapitulation.json",
                   f"--records {run}/{prefix}_records_cpu/records.jsonl "
                   f"--structure {run}/{prefix}_structures_cpu "
                   f"--references {wild_type_references} --wild-type-fasta {wild_type_fasta} "
                   f"--msa-dir {msa_dir} --protocol {proto} --role evaluation"))
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / CAMPAIGN,
                        help="directory the protocol files are frozen into")
    parser.add_argument("--run", type=Path, default=ROOT / "results/R6" / CAMPAIGN,
                        help="run directory the manifests address")
    parser.add_argument("--family-reference-dir", type=Path,
                        default=ROOT / "data/pairwise_assets/family_references_jointh_20261013")
    parser.add_argument("--roles-protocol", type=Path,
                        default=ROOT / "data/e20_confirmatory_20261013/guided_generation_protocol.json",
                        help="protocol whose roles block declares the sealed evaluation set")
    parser.add_argument("--wild-type-fasta", type=Path,
                        default=ROOT / "data/pairwise_assets/same12_provenance_20260924/query_12.faa")
    parser.add_argument("--msa-dir", type=Path,
                        default=ROOT / "data/pairwise_assets/same12_provenance_20260924/msa")
    parser.add_argument("--wild-type-references", type=Path,
                        default=ROOT / "results/R6/e20_wild_type_references_20261013/freeze/wild_type_references.json")
    parser.add_argument("--esmfold2", type=Path, required=True,
                        help="staged biohub/ESMFold2-hf checkpoint")
    parser.add_argument("--python", type=Path, default=Path("/home/lzp/miniconda3/envs/ct/bin/python"))
    parser.add_argument("--fold-python", type=Path, default=ROOT / "runtimes/esmfold2/bin/python")
    args = parser.parse_args(argv)

    stage = stage_module()
    roles = json.loads(args.roles_protocol.read_text())["roles"]
    declared = {fr.domain_id(name) for name in roles["evaluation"]}
    if declared != set(DOMAINS):
        raise SystemExit(
            f"the roles protocol declares evaluation domains {sorted(declared)}, not this "
            f"campaign's {list(DOMAINS)}; the sealed set is not this campaign's to choose")
    cells = cell_plan(stage)
    if len(cells) != len(RUNGS) * len(DOMAINS) * len(ARMS):
        raise SystemExit("the cell plan is not the declared design")

    args.data.mkdir(parents=True, exist_ok=True)
    digests = []
    for rung in RUNGS:
        body = protocol(stage, rung, cells, args.family_reference_dir, roles)
        path = args.data / protocol_name(rung)
        path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
        digests.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (args.data / "SHA256SUMS.txt").write_text("".join(digests))

    env_ct = f"TRANSFER_PYTHON={args.python} {PRELUDE}"
    env_fold = f"TRANSFER_PYTHON={args.fold_python} {PRELUDE}"
    queue = args.run / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    (queue / f"campaign_{CAMPAIGN}_generate.tsv").write_text(
        generate_manifest(cells, args.data, args.family_reference_dir, env_ct))
    for dtype in dict.fromkeys(dtype for _, dtype in RUNGS):
        (queue / f"campaign_{CAMPAIGN}_fold_{dtype}.tsv").write_text(
            fold_manifest(cells, dtype, args.data, args.family_reference_dir,
                          args.wild_type_fasta, args.msa_dir, args.wild_type_references,
                          args.esmfold2, str(args.run), env_ct, env_fold))
    (queue / "cell_plan.json").write_text(json.dumps(cells, indent=1) + "\n")
    print("".join(digests), end="")
    print(f"cells: {len(cells)}  streams: {cells[0]['stream_seed']}..{cells[-1]['stream_seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
