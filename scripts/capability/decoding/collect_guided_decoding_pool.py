#!/usr/bin/env python3
"""E20: one cell of guided autoregressive decoding against an external family Potts model.

A cell is one (model arm, domain, guidance arm, strength-grid point, seed stream). It
samples ``sequences_per_cell`` fixed-length, column-aligned sequences from one
admitted protein decoder with that arm's additive logit bias, scores each one
under the project's native full-sum convention, and writes the pool plus its
per-sequence records. The arms, the bias decomposition and the decoding protocol all live in
:mod:`src.capability.decoding.guided_decoding`; this script binds a frozen
protocol to one cell, resolves the handles and owns the artefacts.

The campaign queue runs one cell per GPU. ``h200/h200_campaign_queue.sh`` injects
``--device <cpu|cuda:N> --out <OUT_DIR>`` **before** the manifest's own arguments
and refuses a manifest whose ``args`` column spells either, so this parser is
flat with no sub-commands and both options are top-level. A manifest row names
this file by basename, which ``scripts/capability/entrypoints.py`` resolves.

Protocol schema (``coupling_guided_generation_v1``)
===================================================

``model_arms``
    The admitted protein decoders of this experiment, a non-empty subset of
    ``protgpt2`` and ``progen2-medium``. Another arm needs its own declared
    native generation interface and is refused by name rather than rendered by
    analogy.
``guidance_arms``
    A non-empty subset of
    :data:`~src.capability.decoding.guided_decoding.GUIDANCE_ARMS`. The four
    factorial arms ``N``, ``M``, ``C`` and ``F`` live inside the Potts model's own
    decomposition, so ``M`` is Potts fields and not the profile; ``Pr`` is the
    conventional profile baseline, read pairwise outside the factorial.
``domains``
    Mapping ``domain -> {L, wt_sha256}``. Every declared domain is loaded from
    ``--family-reference-dir``, must be qualified -- the loader refuses an
    unqualified reference and this script never opts out of that -- and must bind
    both its column count and its wild-type bytes. The lengths of this mapping
    are also the donor pool of ``cross_domain_coupling``, so the donor rule is a
    function of the recorded protocol and not of which cell happened to run.
``strength_grid``
    The declared dose grid over the pair ``{lambda_fields, lambda_coupling}``,
    distinct and drawn from :data:`FIELDS_AXIS` and :data:`COUPLING_AXIS`; the full admissible set is
    :data:`DEVELOPMENT_STRENGTH_GRID`. ``--strength-index`` selects one grid point
    by position; a free-floating value cannot be passed. An arm is refused a
    non-zero strength on a component it does not carry, so the manifest pairs each
    arm with the grid points that act on it -- ``native`` only with ``(0, 0)``,
    ``potts_fields`` and ``profile`` only with ``lambda_coupling = 0``, the three
    coupling arms only with ``lambda_fields = 0``, and ``full`` with both positive.
    Two grid points differing only in an inert scalar would otherwise produce
    byte-identical pools under different labels.
``sequences_per_cell``
    Pool size. One attempt per sequence: the protocol emits exactly ``L``
    residues by construction, so there is no attempt budget to match across arms.
``seed_streams``
    Integer stream seeds whose ``[seed, seed + sequences_per_cell)`` intervals
    are pairwise disjoint and non-adjacent. ``--stream-index`` selects one.
``permutation_seed``
    The recorded seed of ``permuted_coupling``'s column permutation.
``sampling``
    ``temperature`` 1.0, ``top_k`` 0, ``top_p`` 1.0 -- the one declared recipe,
    required exactly -- plus a positive ``batch_size`` for the sampling loop.
``native_contract``
    ``batch_size`` 1, ``budget`` 1024, ``dtype`` ``float32`` and a declared
    ``score`` string. Scoring is one row per forward because the native score is
    batch-extent sensitive at the scale of this programme's endpoints.

Two sequence files, joined on ``id``
====================================

``sequences.jsonl`` is the measurement side's input exactly: the twelve keys of
``e20_generated_sequence_v1``, whose declaration is
:mod:`src.capability.generation.fold_recapitulation` and which refuses an unknown
key. ``sequence_diagnostics.jsonl`` carries this stage's own richer record for the
same sequences -- both strengths, both component scores, the token ids, the
per-step entropies -- so nothing is lost and nothing has to be smuggled into the
contract. The contract is imported rather than restated, so the projection cannot
drift from it, and three consequences of importing it are enforced before any
weight is loaded:

* an arm whose code the contract does not list is refused, naming the one-line
  change that would admit it. The two wrong-coupling controls keep distinct codes
  -- ``Cx`` permuted columns, ``Cd`` cross-domain -- because emitting both as one
  code would merge two different interventions into one measured cell;
* whether an arm is guided is read off its own declared components, and the
  contract is then required to classify it the same way, so a contract that lists
  an arm without classifying it refuses here instead of silently adopting the
  wrong dose rule;
* a guided arm at zero strength is refused, because the contract requires a
  positive ``lambda`` for every guided code -- and because an arm at zero strength
  *is* the native arm, which ``N`` already samples. The
  lambda-zero-reproduces-native invariant is a property of the decoding module and
  is verified there, against both real tokenisers, rather than by writing a
  duplicate campaign cell.

Outputs, in a fresh ``--out`` directory: the two ``.jsonl`` files,
``sequences.jsonl.sha256`` (transfer check for the contract file) and
``guided_decoding_pool.json`` (the completion record, with the protocol digest,
the resolved cell, the checkpoint identity, the code digests and
``arm_provenance`` -- the arm's declaration, its two component sources and, where
it has one, its column permutation or its donor). A failure writes an explicit failure record
and re-raises; it never writes a completion record.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context import context_homologue as ch  # noqa: E402
from src.capability.context.family_potts import FamilyModel, load_family_model  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.arms import output_logit_width  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.decoding.guided_decoding import (  # noqa: E402
    ARM_CODES, ARM_COMPONENTS, ATTEMPTS_PER_SEQUENCE, FACTORIAL_ARMS, GUIDANCE_ARMS,
    SAMPLING_RECIPE, CellIdentity, GuidanceStrength, GuidedDecoder, build_arm_bias,
    component_scale, generate_cell, native_layout, nominal_lambda, select_cross_domain_donor,
    token_span_table)
from src.capability.generation import fold_recapitulation as fr  # noqa: E402
from src.capability.readouts.readout_extraction import load_readout_arm  # noqa: E402

SCHEMA = "coupling_guided_generation_v1"
COMPLETION_SCHEMA = "coupling_guided_decoding_pool_v1"
FAILURE_SCHEMA = "coupling_guided_decoding_failure_v1"
COMPLETION_FILENAME = "guided_decoding_pool.json"
FAILURE_FILENAME = "failure_guided_decoding_pool.json"
SEQUENCES_FILENAME = "sequences.jsonl"
DIAGNOSTICS_FILENAME = "sequence_diagnostics.jsonl"

#: The two decoders this experiment admits. Both already carry complete
#: substitution grids on these domains and a verified native float32 handle.
ALLOWED_MODEL_ARMS = ("protgpt2", "progen2-medium")
SETTLED_NATIVE_CONTRACT: dict[str, Any] = {"batch_size": 1, "budget": 1024, "dtype": "float32"}

#: The declared strength axes, one per component. The coupling term is an order of magnitude
#: weaker per unit than the fields term on the qualified (heavily shrunk) fits, so its axis
#: reaches 16 while the fields axis stops at 2. ``native`` sits at ``(0, 0)``. At ``(1, 1)`` the ``full`` *bias* equals
#: the prefix-truncated Potts conditional up to a per-position constant (not the full-joint
#: conditional), and the *arm* samples the product of that with the language model; an evaluation protocol declares the single frozen point
#: chosen on the development domains. A protocol value off its axis is refused rather than
#: silently swept, so a typo cannot become a dose.
FIELDS_AXIS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
COUPLING_AXIS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
#: Every pair the axes admit; a protocol declares the subset it schedules.
DEVELOPMENT_STRENGTH_GRID: tuple[dict[str, float], ...] = tuple(
    {"lambda_fields": fields, "lambda_coupling": coupling}
    for fields in FIELDS_AXIS for coupling in COUPLING_AXIS
)


def require_contract_cell(guidance_arm: str, strength: GuidanceStrength) -> str:
    """The contract code of one arm, refusing a cell the contract cannot represent.

    Whether an arm is guided is read off its own declared components, not off the
    contract's ``GUIDED_ARMS``; the two are then required to agree, so a contract
    that lists an arm without classifying it refuses here instead of silently
    adopting the wrong dose rule.
    """

    code = ARM_CODES[guidance_arm]
    if code not in fr.ARMS:
        raise ValueError(
            f"{guidance_arm} carries arm code {code!r}, which {fr.SCHEMA_VERSION} does not list "
            f"({list(fr.ARMS)}); the measurement stage would refuse every record of this cell. "
            f"Add {code!r} to src/capability/generation/fold_recapitulation.ARMS (and to its "
            "GUIDED_ARMS) before scheduling this arm; this cell will not write records under "
            "another arm's code"
        )
    guided = any(source != "none" for source in ARM_COMPONENTS[guidance_arm].values())
    if guided != (code in fr.GUIDED_ARMS):
        raise ValueError(
            f"{guidance_arm} ({code}) {'carries' if guided else 'carries no'} bias component, but "
            f"{fr.SCHEMA_VERSION} classifies it as "
            f"{'guided' if code in fr.GUIDED_ARMS else 'unguided'}; reconcile "
            "src/capability/generation/fold_recapitulation.GUIDED_ARMS before scheduling it"
        )
    if guided and strength.is_zero:
        raise ValueError(
            f"{guidance_arm} is a guided arm, so {fr.SCHEMA_VERSION} requires a positive lambda; "
            "an arm at zero strength is the native arm and is already sampled as N. The "
            "lambda-zero invariant is verified in tests/decoding/test_guided_decoding.py, not by "
            "a duplicate campaign cell"
        )
    if not guided and not strength.is_zero:
        raise ValueError(f"{guidance_arm} is the unguided arm and takes a zero strength")
    return code


def contract_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one diagnostics record onto the twelve keys of the measurement contract."""

    row = {
        "schema_version": fr.SCHEMA_VERSION,
        "id": record["id"],
        "domain": record["domain"],
        "model": record["model"],
        "arm": record["arm_code"],
        "lambda": record["lambda_nominal"],
        "seed": record["seed"],
        "sequence": record["sequence"],
        "native_loglik": record["native_loglik"],
        "guidance_score": record["guidance_score"],
        "mean_step_entropy": record["mean_step_entropy"],
        "attempt_budget": record["attempt_budget"],
    }
    if tuple(row) != fr.FIELDS:
        raise ValueError(
            f"the contract projection emits {tuple(row)}, not {fr.SCHEMA_VERSION}'s {fr.FIELDS}"
        )
    return row


def _dtos() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sequence_sha(sequence: str) -> str:
    return sha256(sequence.encode("utf-8")).hexdigest()


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """The pool mean of one field, or ``None`` where the arm has no such quantity.

    ``guidance_score`` and the two component scores are null for an arm that
    carries no fields or no coupling component, and a null is the honest summary
    of a quantity the arm does not have rather than a zero to be averaged.
    """

    values = [record[key] for record in records]
    if any(value is None for value in values):
        if not all(value is None for value in values):
            raise ValueError(f"{key} is null for part of the pool only")
        return None
    return float(np.mean(values))


def pool_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """The cell's pool means, with a null where the arm has no such component."""

    return {
        "mean_native_loglik": _mean(records, "native_loglik"),
        "mean_guidance_score": _mean(records, "guidance_score"),
        "mean_fields_score": _mean(records, "fields_score"),
        "mean_coupling_score": _mean(records, "coupling_score"),
        "mean_guidance_bias_applied": _mean(records, "guidance_bias_applied"),
        "mean_step_entropy_nats": _mean(records, "mean_step_entropy"),
        "mean_residues_per_token": _mean(records, "residues_per_token"),
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows).encode("utf-8")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Every source whose bytes decide a number in this cell. The native score comes
#: from the shared extraction kernel rather than a second scorer declared here.
CODE_PATHS = (
    "scripts/capability/decoding/collect_guided_decoding_pool.py",
    "scripts/capability/stability/extract_stability_singles.py",
    "scripts/capability/stages/context_homologue.py",
    "src/capability/decoding/guided_decoding.py",
    "src/capability/decoding/guidance_strength.py",
    "src/capability/context/family_potts.py",
    "src/capability/context/context_homologue.py",
    "src/capability/readouts/readout_extraction.py",
    "src/capability/generation/unconditional_generation.py",
    "src/capability/core/arms.py",
    "src/capability/core/io.py",
)


def prepare_output_dir(path: Path) -> Path:
    """A fresh, owned output directory. An existing non-empty directory is refused."""

    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path {path} exists and is not a directory")
        if any(path.iterdir()):
            raise ValueError(f"refusing to reuse the non-empty output directory {path}")
    else:
        path.mkdir(parents=True)
    return path


# ------------------------------------------------------------- protocol binding


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _require_subset(value: Any, allowed: tuple[str, ...], *, field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(name, str) for name in value):
        raise ValueError(f"{field} must be a non-empty list of names")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} repeats a name")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{field} names {unknown}, which are not among {list(allowed)}")
    return list(value)


def validate_protocol(
    protocol: Any, *, model_arm: str, domain: str, guidance_arm: str, strength_index: int,
    stream_index: int,
) -> dict[str, Any]:
    """Validate the frozen protocol and bind it to one cell, without touching the filesystem."""

    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a JSON object")
    if protocol.get("schema") != SCHEMA:
        raise ValueError(f"protocol schema must be {SCHEMA!r}")

    model_arms = _require_subset(protocol.get("model_arms"), ALLOWED_MODEL_ARMS, field="protocol.model_arms")
    if model_arm not in model_arms:
        raise ValueError(f"model arm {model_arm!r} is not declared by the protocol")
    guidance_arms = _require_subset(protocol.get("guidance_arms"), GUIDANCE_ARMS, field="protocol.guidance_arms")
    if guidance_arm not in guidance_arms:
        raise ValueError(f"guidance arm {guidance_arm!r} is not declared by the protocol")

    sampling = protocol.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("protocol.sampling must be an object")
    for key, expected in SAMPLING_RECIPE.items():
        observed = sampling.get(key)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or float(observed) != float(expected):
            raise ValueError(f"protocol.sampling.{key} must be {expected!r}, got {observed!r}")
    batch_size = _require_int(sampling.get("batch_size"), field="protocol.sampling.batch_size")
    if batch_size < 1:
        raise ValueError("protocol.sampling.batch_size must be positive")

    contract = protocol.get("native_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("protocol.native_contract must be an object")
    for key, expected in SETTLED_NATIVE_CONTRACT.items():
        if contract.get(key) != expected:
            raise ValueError(f"protocol.native_contract.{key} must be {expected!r}")
    if not isinstance(contract.get("score"), str) or not contract["score"]:
        raise ValueError("protocol.native_contract.score must be a non-empty string")

    count = _require_int(protocol.get("sequences_per_cell"), field="protocol.sequences_per_cell")
    if count < 1:
        raise ValueError("protocol.sequences_per_cell must be positive")

    grid = protocol.get("strength_grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("protocol.strength_grid must be a non-empty list of strength pairs")
    strengths = []
    for entry in grid:
        if not isinstance(entry, Mapping) or set(entry) != {"lambda_fields", "lambda_coupling"}:
            raise ValueError(
                "every protocol.strength_grid entry must hold exactly lambda_fields and "
                "lambda_coupling"
            )
        strength = GuidanceStrength(entry["lambda_fields"], entry["lambda_coupling"])
        off_axis = [value for value, axis in ((strength.fields, FIELDS_AXIS), (strength.coupling, COUPLING_AXIS))
                    if value not in axis]
        if off_axis:
            raise ValueError(
                f"protocol.strength_grid holds {off_axis}, which is off the declared strength "
                f"axes fields {list(FIELDS_AXIS)} and coupling {list(COUPLING_AXIS)}"
            )
        strengths.append(strength)
    if len(set(strengths)) != len(strengths):
        raise ValueError("protocol.strength_grid repeats a grid point")
    if not 0 <= strength_index < len(strengths):
        raise ValueError(f"strength index {strength_index} is outside 0..{len(strengths) - 1}")

    seeds = protocol.get("seed_streams")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("protocol.seed_streams must be a non-empty list of integer seeds")
    streams = [_require_int(seed, field="protocol.seed_streams[]") for seed in seeds]
    if len(set(streams)) != len(streams):
        raise ValueError("protocol.seed_streams must be pairwise distinct")
    for left, right in zip(sorted(streams)[:-1], sorted(streams)[1:], strict=True):
        if left + count >= right:
            raise ValueError(
                f"protocol.seed_streams must occupy pairwise disjoint, non-adjacent intervals of "
                f"{count} seeds; {left} and {right} are not separated"
            )
    if not 0 <= stream_index < len(streams):
        raise ValueError(f"stream index {stream_index} is outside 0..{len(streams) - 1}")

    domains = protocol.get("domains")
    if not isinstance(domains, Mapping) or not domains:
        raise ValueError("protocol.domains must be a non-empty object")
    declared: dict[str, dict[str, Any]] = {}
    for name, entry in domains.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"protocol.domains.{name} must be an object")
        length = _require_int(entry.get("L"), field=f"protocol.domains.{name}.L")
        wt_sha = entry.get("wt_sha256")
        if not isinstance(wt_sha, str) or len(wt_sha) != 64:
            raise ValueError(f"protocol.domains.{name}.wt_sha256 must be a sha256 hex digest")
        declared[name] = {"L": length, "wt_sha256": wt_sha}
    if domain not in declared:
        raise ValueError(f"domain {domain!r} is not declared by the protocol")

    return {
        "model_arm": model_arm,
        "model_arms": model_arms,
        "guidance_arm": guidance_arm,
        "guidance_arms": guidance_arms,
        "domain": domain,
        "domains": declared,
        "strength_grid": [{"lambda_fields": item.fields, "lambda_coupling": item.coupling}
                          for item in strengths],
        "strength_index": int(strength_index),
        "strength": strengths[strength_index],
        "stream_index": int(stream_index),
        "stream_seed": streams[stream_index],
        "seed_streams": streams,
        "sequences_per_cell": count,
        "sampling": {**{key: float(value) for key, value in SAMPLING_RECIPE.items()},
                     "batch_size": batch_size},
        "native_contract": {**SETTLED_NATIVE_CONTRACT, "score": contract["score"]},
        "permutation_seed": _require_int(protocol.get("permutation_seed"), field="protocol.permutation_seed"),
    }


def resolve_reference_dir(given: Path, declared: Mapping[str, Mapping[str, Any]]) -> Path:
    """The staged family-reference directory, with every declared domain present.

    Named by ``--family-reference-dir`` and nothing else: the freeze does not
    carry ``data/``, so a path derived from this file's location would point
    inside the code snapshot. A missing directory or a missing domain artefact is
    refused by option name, because the operator needs to know what to stage.
    """

    directory = Path(given).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(
            f"--family-reference-dir {directory} is not a directory; stage the exported family "
            "references there and pass the staged path"
        )
    missing = sorted(
        f"{name}{suffix}" for name in declared for suffix in (".npz", ".json")
        if not (directory / f"{name}{suffix}").is_file()
    )
    if missing:
        raise ValueError(
            f"--family-reference-dir {directory} lacks {missing}; every domain the protocol "
            "declares must have both artefacts staged"
        )
    return directory


def load_declared_models(directory: Path, declared: Mapping[str, Mapping[str, Any]]) -> dict[str, FamilyModel]:
    """Load every declared domain, binding its column count and wild-type bytes.

    All of them, not only the target: the qualified lengths of the whole declared
    set are the donor pool of ``cross_domain_coupling``, so an incomplete load
    would make the donor rule depend on which cell happened to run.
    """

    models: dict[str, FamilyModel] = {}
    for name, entry in sorted(declared.items()):
        model = load_family_model(name, directory)
        if model.length != int(entry["L"]):
            raise ValueError(
                f"{name}: the family reference has {model.length} columns and the protocol "
                f"declares L = {entry['L']}; a column-aligned comparison needs them equal"
            )
        if _sequence_sha(model.wildtype) != entry["wt_sha256"]:
            raise ValueError(f"{name}: protocol wt_sha256 does not bind the reference wild-type bytes")
        models[name] = model
    return models


# ------------------------------------------------------------------ the handles


def _observed_parameter_dtypes(arm: Any) -> list[str]:
    return sorted({str(parameter.dtype).removeprefix("torch.") for parameter in arm.model.parameters()})


def _checkpoint_record(arm: Any) -> dict[str, Any]:
    checkpoint = Path(arm.spec.path)
    small: dict[str, str] = {}
    for item in sorted(checkpoint.rglob("*")):
        if item.is_file() and item.stat().st_size <= 1 << 20:
            small[str(item.relative_to(checkpoint))] = sha256_file(item)
    return {
        "arm": arm.name,
        "path": str(checkpoint),
        "architecture": arm.spec.architecture,
        "input_format": arm.spec.input_format,
        "dtype": arm.dtype,
        "device": str(arm.device),
        "small_file_sha256": small,
        "weight_byte_identity_verified": False,
    }


# --------------------------------------------------------------------- the cell


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    resolved = validate_protocol(
        protocol, model_arm=args.model_arm, domain=args.domain, guidance_arm=args.guidance_arm,
        strength_index=int(args.strength_index), stream_index=int(args.stream_index))

    device = str(args.device)
    if device != "cpu" and not device.startswith("cuda"):
        raise SystemExit(f"--device must be cpu or cuda:N, got {device!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available on this host")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    code = require_contract_cell(resolved["guidance_arm"], resolved["strength"])
    reference_dir = resolve_reference_dir(args.family_reference_dir, resolved["domains"])
    models = load_declared_models(reference_dir, resolved["domains"])
    target = models[resolved["domain"]]

    donor = None
    if resolved["guidance_arm"] == "cross_domain_coupling":
        lengths = {name: model.length for name, model in models.items() if model.qualified}
        donor = models[select_cross_domain_donor(resolved["domain"], lengths)]

    out = prepare_output_dir(Path(args.out))

    stage46 = _load_module("guided_stage46", ROOT / "scripts/capability/stages/context_homologue.py")
    singles = _load_module(
        "guided_stability_singles", ROOT / "scripts/capability/stability/extract_stability_singles.py")
    arm = load_readout_arm(resolved["model_arm"], stage46, device=device, dtype="float32")
    arm.model.eval()
    arm.model.requires_grad_(False)
    observed = _observed_parameter_dtypes(arm)
    if observed != ["float32"]:
        raise ValueError(
            f"{resolved['model_arm']}: the loaded handle carries dtypes {observed}; this cell "
            "requires one float32 handle for both sampling and scoring"
        )
    ch.require_position_budget(arm.model.config, arm=resolved["model_arm"])
    width = output_logit_width(arm)
    budget = int(resolved["native_contract"]["budget"])

    def score_fn(sequence: str) -> float:
        return float(singles.likelihood_score(arm, sequence, stage46, budget))

    layout = native_layout(arm, target.length)
    table = token_span_table(arm, layout, logit_width=int(width["size"]))
    bias = build_arm_bias(
        resolved["guidance_arm"], target,
        permutation_seed=resolved["permutation_seed"] if resolved["guidance_arm"] == "permuted_coupling" else None,
        donor=donor)
    decoder = GuidedDecoder(bias=bias, layout=layout, table=table, strength=resolved["strength"])

    identity = CellIdentity(
        model_arm=resolved["model_arm"], domain=resolved["domain"],
        guidance_arm=resolved["guidance_arm"], strength=resolved["strength"],
        strength_index=resolved["strength_index"], stream_index=resolved["stream_index"],
        stream_seed=resolved["stream_seed"])
    records = generate_cell(
        arm=arm, decoder=decoder, identity=identity, count=resolved["sequences_per_cell"],
        batch_size=int(resolved["sampling"]["batch_size"]), score_fn=score_fn)

    contract_rows = [contract_record(record) for record in records]
    fr.validate_records(contract_rows, {resolved["domain"]: target.wildtype})
    payload = _jsonl(contract_rows)
    (out / SEQUENCES_FILENAME).write_bytes(payload)
    digest = sha256(payload).hexdigest()
    (out / f"{SEQUENCES_FILENAME}.sha256").write_text(f"{digest}  {SEQUENCES_FILENAME}\n")
    diagnostics = _jsonl(records)
    (out / DIAGNOSTICS_FILENAME).write_bytes(diagnostics)

    lengths = {len(record["sequence"]) for record in records}
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "run_directory": Path(args.out).resolve().parent.name,
        "cell": {
            "model_arm": identity.model_arm,
            "domain": identity.domain,
            "guidance_arm": identity.guidance_arm,
            "arm_code": code,
            "lambda": nominal_lambda(bias, identity.strength),
            "lambda_fields": identity.strength.fields,
            "lambda_coupling": identity.strength.coupling,
            "strength_index": identity.strength_index,
            "stream_index": identity.stream_index,
            "stream_seed": identity.stream_seed,
            "sequences_per_cell": resolved["sequences_per_cell"],
            "attempt_budget": ATTEMPTS_PER_SEQUENCE,
        },
        "arm_provenance": bias.provenance,
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_bytes).hexdigest(),
                     "declared_arms": resolved["guidance_arms"],
                     "declared_model_arms": resolved["model_arms"],
                     "strength_grid": resolved["strength_grid"],
                     "seed_streams": resolved["seed_streams"]},
        "sampling": resolved["sampling"],
        "native_contract": resolved["native_contract"],
        "layout": {
            "characters": layout.characters,
            "residue_slots": layout.length,
            "prompt": layout.prompt,
            "structural_literals": sorted(layout.literals.values()),
            "derivation": (
                "residue and literal character slots derived from two renderings of this arm's own "
                "declared input format that differ at every residue"
            ),
        },
        "candidates": {
            "logit_width": int(width["size"]),
            "logit_width_source": width["source"],
            "candidate_tokens": table.size,
            "max_residues_per_token": table.max_residues,
            "suppression_rule": (
                "a token is admissible only where its decoded text matches the native layout and "
                "its residues fit the remaining budget; end-of-sequence, non-canonical and "
                "overshooting tokens are therefore never drawn, identically in every arm"
            ),
        },
        "family_reference": {
            "directory": str(reference_dir),
            "option": "--family-reference-dir",
            "qualification": "load_family_model refuses an unqualified reference; no opt-out is passed",
            "domains": {name: {"length": model.length, "qualified": model.qualified,
                               "npz_sha256": model.sidecar["npz_sha256"]}
                        for name, model in sorted(models.items())},
            "donor": None if donor is None else donor.domain,
        },
        "checkpoint": _checkpoint_record(arm),
        "source": {
            "script": "scripts/capability/decoding/collect_guided_decoding_pool.py",
            "code_sha256": {path: sha256_file(ROOT / path) for path in CODE_PATHS},
        },
        "counts": {
            "sequences": len(records),
            "distinct_sequences": len({record["sequence_sha256"] for record in records}),
            "token_steps": sum(record["token_steps"] for record in records),
            "observed_lengths": sorted(lengths),
            "exact_length": lengths == {target.length},
        },
        "summaries": pool_summaries(records),
        "component_scale": {
            "wildtype": component_scale(target, [target.wildtype]),
            "pool": component_scale(target, [record["sequence"] for record in records]),
            "basis": (
                "wildtype is the domain's own wild type; pool is this cell's sampled sequences, "
                "which for the native cell are the native-arm samples. Both are measured on the "
                "target reference's own h, J and log f, never on a permuted or donor tensor, so "
                "every arm of one domain reports the same quantity"
            ),
        },
        "factorial": {
            "arms": list(FACTORIAL_ARMS),
            "reading": (
                "coupling main effects are full - potts_fields and coupling - native; field main "
                "effects are full - coupling and potts_fields - native. Both fields arms of the "
                "factorial carry the Potts h, so neither contrast mixes a fields swap into a "
                "coupling effect. The profile arm (Pr) is a pairwise practice baseline outside it"
            ),
        },
        "artifacts": {
            "sequences": {"file": SEQUENCES_FILENAME, "sha256": digest, "records": len(records),
                          "sidecar": f"{SEQUENCES_FILENAME}.sha256",
                          "schema": fr.SCHEMA_VERSION,
                          "validated_against": "fold_recapitulation.validate_records"},
            "diagnostics": {"file": DIAGNOSTICS_FILENAME,
                            "sha256": sha256(diagnostics).hexdigest(),
                            "records": len(records), "join_key": "id"},
        },
        "ceiling": (
            "a sampled pool and its model-internal scores; no structural, stability or biological "
            "endpoint is computed or implied here"
        ),
        "invocation": {
            "device": device,
            "family_reference_dir": str(reference_dir),
            "out": str(Path(args.out).resolve()),
        },
        "updated_utc": _dtos(),
    }
    if not completion["counts"]["exact_length"]:
        raise RuntimeError(f"a pool sequence is not exactly {target.length} residues")
    for record in records:
        if set(record["sequence"]) - set(AA20):
            raise RuntimeError("a pool sequence carries a non-canonical residue")
    write_json(out / COMPLETION_FILENAME, completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    """The cell's one flat parser.

    No sub-commands, and ``--device``/``--out`` are top-level: the campaign queue
    injects both ahead of the manifest's own arguments and refuses a manifest that
    spells either, so a sub-command in front of them would make the cell
    unlaunchable.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True,
                        help="cpu or cuda:N; injected by the campaign queue before the cell arguments")
    parser.add_argument("--out", type=Path, required=True,
                        help="fresh output directory; injected by the campaign queue")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family-reference-dir", type=Path, required=True,
                        help="staged directory of the exported family references; absolute in a pod")
    parser.add_argument("--model-arm", required=True, choices=list(ALLOWED_MODEL_ARMS))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--guidance-arm", required=True, choices=list(GUIDANCE_ARMS))
    parser.add_argument("--strength-index", type=int, required=True,
                        help="position in the protocol's strength grid; a free pair cannot be passed")
    parser.add_argument("--stream-index", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out = Path(args.out)
    try:
        completion = run_job(args)
    except Exception as error:
        if out.is_dir():
            try:
                write_json(out / FAILURE_FILENAME, {
                    "schema": FAILURE_SCHEMA,
                    "status": "failed",
                    "model_arm": args.model_arm,
                    "domain": args.domain,
                    "guidance_arm": args.guidance_arm,
                    "strength_index": int(args.strength_index),
                    "stream_index": int(args.stream_index),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "updated_utc": _dtos(),
                })
            except Exception:  # pragma: no cover - the original failure is re-raised
                pass
        raise
    print(json.dumps({
        "completion": str(out / COMPLETION_FILENAME),
        "cell": completion["cell"],
        "counts": completion["counts"],
        "summaries": completion["summaries"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
