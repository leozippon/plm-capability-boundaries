#!/usr/bin/env python3
"""Solve the ProGen3 dose that matches ProGen2-medium's measured guidance strength.

Nominal lambda is not comparable across models: the same bias competes against a
different logit scale in each, so a shared number is a shared label and not a
shared intervention. The comparable statistic is
:data:`~src.capability.decoding.guidance_strength.RECORD_KEYS`'
``kl_per_residue`` -- the summed per-step KL between the guided and the native
sampling distributions divided by the residues emitted -- which
:mod:`src.capability.decoding.guidance_strength` establishes as the cross-tokeniser
quantity. This script measures it for both models and solves, per guidance arm,
the ProGen3 dose at which the two agree.

The rule, which is pre-specified and takes no endpoint as input
=============================================================

For each guidance arm, the ProGen3 dose is the one whose measured median
per-residue KL on the wild-type basis equals the value the reference model
receives at its frozen E20 strength for that arm. The statistic is the **median
over the eight evaluation domains** of each domain's own wild-type-basis
``kl_per_residue``, because the domain is this programme's independence unit.
Nothing downstream of sampling -- no structure, no novelty, no selector -- enters,
so the dose is fixed before any outcome exists.

Both models are measured on the **same basis**: each domain's own wild-type token
path. That is the choice this script makes explicitly, and it is not free. The
reference model's already-recorded in-line strengths are measured on its *sampled
pool*, which is a different conditioning prefix: on ProGen3-112M at
``lambda_coupling = 1`` the 1E0L pool-basis value is about 0.004 nats per residue
against 0.028 on the wild type, a factor of seven. Matching a pool-basis number
against a wild-type-basis number would therefore match two different quantities.
The recorded pool medians are read and reported beside the measurement as
provenance, never as the target.

Why the replay is incremental
=============================

:func:`~src.capability.decoding.guided_decoding.replay_strength` reads every
step's logits from one teacher-forced pass, which is the computation the sampler
ran only for a prefix-exact forward. ProGen3's is not, so this script walks every
path through the sampler's own incremental cache at batch extent one
(:data:`~src.capability.decoding.guided_decoding.INCREMENTAL_REPLAY_DECLARATION`).
One walk per (model, domain, arm) serves the whole dose curve: the native logits
are a property of the path and the two bias components scale linearly in the two
lambdas.

The degenerate-dose refusal
===========================

A dose whose bias cannot outrun the arithmetic it is added to is not an
intervention. The caller declares the measured noise floor of the target's
reduced-precision forward and the margin the intervention must clear; the solved
dose's per-step bias spread -- the component's across-residue standard deviation
at that dose, :func:`~src.capability.decoding.guided_decoding.component_scale` --
must exceed ``margin x floor``, and this script **refuses** the dose rather than
scheduling it and discounting it afterwards.

Outputs, in a fresh ``--out`` directory: ``progen3_dose_calibration.json`` with
the reference targets, the full per-domain dose curve, the solved dose of each
arm, its achieved statistic, the residual mismatch and the floor margins, plus
``.sha256``. A refusal is written into the artefact and the script exits non-zero.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context.family_potts import FamilyModel, load_family_model  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402
from src.capability.decoding.guided_decoding import (  # noqa: E402
    ARM_CODES, INCREMENTAL_REPLAY_DECLARATION, GuidanceStrength, GuidedDecoder, build_arm_bias,
    canonical_token_ids, component_scale, incremental_path_steps, native_layout, path_strength,
    token_span_table)
from src.capability.decoding.progen3_decoding import progen3_decoding_arm  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402
from src.capability.readouts.readout_extraction import load_readout_arm  # noqa: E402

SCHEMA = "progen3_guidance_dose_calibration_v1"
OUTPUT_FILENAME = "progen3_dose_calibration.json"
COMPARABLE = "kl_per_residue"

#: The guidance arms this calibration solves a dose for, by contract code. ``N`` takes no
#: dose by construction and is not listed; ``F`` and ``Cd`` are outside this robustness
#: check's five-arm design and are not solved, so no dose for them can be read off here.
SOLVED_ARMS: dict[str, str] = {
    "Pr": "profile", "M": "potts_fields", "C": "coupling", "Cx": "permuted_coupling",
}
#: Which component of :class:`GuidanceStrength` each solved arm's dose moves.
DOSE_COMPONENT: dict[str, str] = {"Pr": "fields", "M": "fields", "C": "coupling", "Cx": "coupling"}
#: The ``component_scale`` component whose per-step spread each arm's bias has.
SPREAD_COMPONENT: dict[str, str] = {
    "Pr": "profile", "M": "potts_fields", "C": "coupling", "Cx": "coupling",
}

#: The dose curve is reported at these multiples of the solved dose as well as at E20's
#: own settled axis, so the artefact shows both the slope the solution sits on and where
#: the solution falls relative to the doses E20 could schedule.
CURVE_MULTIPLES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
#: Significant digits the solved dose is rounded to before it is declared. A dose is a
#: declared constant, so it is written at a readable precision and the residual mismatch
#: is then measured at the rounded value rather than at the unrounded root.
DOSE_SIGNIFICANT_DIGITS = 4
#: Bracket the bisection searches for the root inside. A root outside it means the two
#: models' strength scales do not overlap on this arm and is reported, not clamped.
SEARCH_BRACKET = (1e-4, 1024.0)
BISECTION_STEPS = 80


def _load_script(relative: str) -> Any:
    path = ROOT / "scripts/capability" / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settled_axis(collect: Any) -> dict[str, tuple[float, ...]]:
    """E20's own positive dose axis per component, read from the stage that enforces it.

    Reported as the curve's backbone so the artefact shows where the solution falls
    relative to the doses E20 could schedule. Read rather than restated: a second
    copy of an axis is a second thing to keep in step with the stage's refusal.
    """

    return {"fields": tuple(value for value in collect.FIELDS_AXIS if value),
            "coupling": tuple(value for value in collect.COUPLING_AXIS if value)}


# ------------------------------------------------------------------ the handles


def load_arm(name: str, dtype: str, *, device: str, stage46: Any) -> Any:
    """One loaded decoding handle, by architecture.

    ProGen3 goes through :func:`progen3_decoding_arm`, which is the only supported
    way in: the generic loader is AutoModel and would sample random experts. Every
    other arm goes through the readout loader this programme already scores with.
    """

    if name.startswith("progen3-"):
        return progen3_decoding_arm(name, device=device, dtype=dtype)
    arm = load_readout_arm(name, stage46, device=device, dtype=dtype)
    arm.model.eval()
    arm.model.requires_grad_(False)
    return arm


def arm_dtypes(arm: Any) -> list[str]:
    handle = (arm.serving_provenance or {}).get("progen3")
    module = handle.model if handle is not None else arm.model
    return sorted({str(p.dtype).removeprefix("torch.") for p in module.parameters()})


# --------------------------------------------------------------- the dose curve


def walk(arm: Any, family: FamilyModel, guidance_arm: str, *, permutation_seed: int,
         logit_width: int) -> list[dict[str, Any]]:
    """One (model, domain, arm) wild-type walk: per-step native logits and unit components."""

    layout = native_layout(arm, family.length)
    bias = build_arm_bias(
        guidance_arm, family,
        permutation_seed=permutation_seed if guidance_arm == "permuted_coupling" else None)
    component = "fields" if bias.has_fields else "coupling"
    decoder = GuidedDecoder(
        bias=bias, layout=layout, table=token_span_table(arm, layout, logit_width=logit_width),
        strength=GuidanceStrength(**{component: 1.0,
                                     "coupling" if component == "fields" else "fields": 0.0}))
    ids, segmentation = canonical_token_ids(arm, decoder, family.wildtype)
    if segmentation != "exact":
        raise ValueError(
            f"{arm.name}/{family.domain}: the wild-type path is {segmentation}, so the walked "
            "path is not the one the sampler would take; this calibration needs an exact "
            "single-character segmentation")
    return incremental_path_steps(arm=arm, decoder=decoder, token_ids=ids)


def statistic(walks: Mapping[str, Sequence[Mapping[str, Any]]], code: str,
              dose: float) -> dict[str, Any]:
    """The per-domain wild-type-basis ``kl_per_residue`` at one dose, and its median."""

    component = DOSE_COMPONENT[code]
    strength = GuidanceStrength(**{component: float(dose),
                                   "coupling" if component == "fields" else "fields": 0.0})
    per_domain = {domain: float(path_strength(steps, strength)[COMPARABLE])
                  for domain, steps in sorted(walks.items())}
    return {"per_domain": per_domain,
            "median": float(np.median(list(per_domain.values()))),
            "min": float(min(per_domain.values())),
            "max": float(max(per_domain.values()))}


def solve_dose(walks: Mapping[str, Sequence[Mapping[str, Any]]], code: str,
               target: float) -> dict[str, Any]:
    """Bisect the dose whose median statistic equals ``target``.

    The statistic is zero at a zero dose and strictly increasing in it, and a
    median of increasing functions is increasing, so the root is unique and
    bisection is exact to the bracket's resolution. A target outside the bracket's
    reach is reported with the bracket's own values rather than clamped.
    """

    low, high = SEARCH_BRACKET
    at_low, at_high = statistic(walks, code, low)["median"], statistic(walks, code, high)["median"]
    if not at_low <= target <= at_high:
        return {"bracketed": False, "bracket": list(SEARCH_BRACKET),
                "median_at_bracket": [at_low, at_high], "target": target}
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2.0
        if statistic(walks, code, middle)["median"] < target:
            low = middle
        else:
            high = middle
    root = (low + high) / 2.0
    rounded = float(f"%.{DOSE_SIGNIFICANT_DIGITS}g" % root)
    achieved = statistic(walks, code, rounded)
    return {"bracketed": True, "root": root, "dose": rounded, "achieved": achieved,
            "target": target,
            "residual_relative": (achieved["median"] - target) / target if target else None}


# ------------------------------------------------------------- the floor refusal


def bias_spread(families: Mapping[str, FamilyModel], code: str, dose: float) -> dict[str, Any]:
    """The solved dose's per-step bias spread per domain, on the wild-type basis.

    ``component_scale``'s spread is the across-residue standard deviation of the
    component's per-position bias vector at unit dose, so the applied spread is
    that times the dose. The decision statistic is its median over the domains --
    one number per arm, matching the one dose per arm -- and the per-domain values
    are reported beside it.
    """

    name = SPREAD_COMPONENT[code]
    per_domain = {
        domain: dose * float(
            component_scale(family, [family.wildtype])["components"][name]["median"])
        for domain, family in sorted(families.items())}
    return {"component": name, "dose": dose, "per_domain": per_domain,
            "median": float(np.median(list(per_domain.values()))),
            "min": float(min(per_domain.values()))}


# ------------------------------------------------------------------- the artefact


def frozen_reference_strengths(protocol: Mapping[str, Any], model: str) -> dict[str, float]:
    """``code -> frozen dose`` of the reference model, read from its own frozen protocol."""

    declaration = fr.parse_strength_declaration(protocol["strengths"])
    out: dict[str, float] = {}
    for code in SOLVED_ARMS:
        key = (model, code)
        if key not in declaration:
            raise ValueError(f"{model}/{code} has no frozen_strength in the reference protocol")
        pair = declaration[key][0]
        fields, coupling = float(pair[0]), float(pair[1])
        dose = coupling if DOSE_COMPONENT[code] == "coupling" else fields
        inert = fields if DOSE_COMPONENT[code] == "coupling" else coupling
        if inert:
            raise ValueError(
                f"{model}/{code}: the frozen strength {pair} moves both components, so one dose "
                "does not describe it and this calibration cannot match it")
        out[code] = dose
    return out


def inline_pool_medians(cells: Path, model: str, domains: Sequence[str],
                        frozen: Mapping[str, float]) -> dict[str, Any]:
    """The reference model's recorded in-line pool medians, as provenance beside the target.

    Read from each frozen cell's own ``sequence_diagnostics.jsonl``, which is where
    the sampler writes the in-line record. Reported, never matched: see the module
    docstring on the basis.
    """

    def tag(value: float) -> str:
        return f"{round(value * 100):03d}"

    out: dict[str, Any] = {}
    for code, dose in sorted(frozen.items()):
        fields = dose if DOSE_COMPONENT[code] == "fields" else 0.0
        coupling = dose if DOSE_COMPONENT[code] == "coupling" else 0.0
        per_domain = {}
        for domain in domains:
            path = (cells / f"e20conf_gen_{model}_{domain}_{code}_f{tag(fields)}_c{tag(coupling)}"
                    / "sequence_diagnostics.jsonl")
            values = [json.loads(line)["guidance_strength"][COMPARABLE]
                      for line in path.read_text().splitlines() if line.strip()]
            per_domain[domain] = {"sequences": len(values), "median": float(np.median(values))}
        out[code] = {"per_domain": per_domain,
                     "median_of_domain_medians": float(
                         np.median([v["median"] for v in per_domain.values()]))}
    return out


def measure(arm: Any, families: Mapping[str, FamilyModel], *, permutation_seed: int,
            logit_width: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Every (domain, arm) wild-type walk of one loaded handle."""

    return {
        code: {domain: walk(arm, family, guidance, permutation_seed=permutation_seed,
                            logit_width=logit_width)
               for domain, family in sorted(families.items())}
        for code, guidance in sorted(SOLVED_ARMS.items())
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_bytes = args.reference_protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    roles = {fr.domain_id(name) for name in protocol["roles"]["evaluation"]}
    domains = tuple(sorted(roles))
    permutation_seed = int(protocol["permutation_seed"])
    families = {name: load_family_model(name, args.family_reference_dir) for name in domains}
    for name, family in families.items():
        if not family.qualified:
            raise ValueError(f"{name}: the family reference is not qualified")

    device = str(args.device)
    if device != "cpu" and not device.startswith("cuda"):
        raise SystemExit(f"--device must be cpu or cuda:N, got {device!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available on this host")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stage46 = _load_script("stages/context_homologue.py")

    frozen = frozen_reference_strengths(protocol, args.reference_model)
    sides: dict[str, Any] = {}
    for role, name, dtype in (("reference", args.reference_model, args.reference_dtype),
                              ("target", args.target_model, args.target_dtype)):
        arm = load_arm(name, dtype, device=device, stage46=stage46)
        observed = arm_dtypes(arm)
        if observed != [dtype]:
            raise ValueError(f"{name}: the loaded handle carries dtypes {observed}, not [{dtype!r}]")
        from src.capability.core.arms import output_logit_width

        width = output_logit_width(arm)
        sides[role] = {
            "model": name, "dtype": dtype, "logit_width": width,
            "walks": measure(arm, families, permutation_seed=permutation_seed,
                             logit_width=int(width["size"])),
        }
        del arm
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    axis = settled_axis(_load_script("decoding/collect_guided_decoding_pool.py"))
    targets = {code: statistic(sides["reference"]["walks"][code], code, frozen[code])
               for code in sorted(SOLVED_ARMS)}
    solved: dict[str, Any] = {}
    curve: dict[str, Any] = {}
    refusals: list[str] = []
    for code in sorted(SOLVED_ARMS):
        walks = sides["target"]["walks"][code]
        solution = solve_dose(walks, code, targets[code]["median"])
        entry: dict[str, Any] = {
            "arm": SOLVED_ARMS[code],
            "component": DOSE_COMPONENT[code],
            "reference_frozen_dose": frozen[code],
            "reference_target": targets[code],
            **solution,
        }
        points = set(axis[DOSE_COMPONENT[code]])
        if solution["bracketed"]:
            points |= {solution["dose"] * multiple for multiple in CURVE_MULTIPLES}
        ordered = sorted(float("%.6g" % value) for value in points)
        curve[code] = [{"dose": point, **statistic(walks, code, point)} for point in ordered]
        if not solution["bracketed"]:
            refusals.append(
                f"{code}: the reference target {targets[code]['median']:.6g} nats per residue is "
                f"outside the dose bracket {SEARCH_BRACKET}, whose medians are "
                f"{solution['median_at_bracket']}; no ProGen3 dose matches it")
            solved[code] = entry
            continue
        tolerance = float(args.relative_tolerance)
        entry["relative_tolerance"] = tolerance
        entry["within_tolerance"] = abs(solution["residual_relative"]) <= tolerance
        spread = bias_spread(families, code, solution["dose"])
        required = float(args.floor_margin) * float(args.noise_floor)
        entry["bias_spread"] = spread
        entry["noise_floor_nats"] = float(args.noise_floor)
        entry["floor_margin_required"] = float(args.floor_margin)
        entry["floor_margin_achieved"] = spread["median"] / float(args.noise_floor)
        entry["clears_floor"] = spread["median"] > required
        entry["per_domain_floor_margin"] = {
            domain: value / float(args.noise_floor) for domain, value in spread["per_domain"].items()}
        if not entry["within_tolerance"]:
            refusals.append(
                f"{code}: the solved dose {solution['dose']} achieves "
                f"{solution['achieved']['median']:.6g} against the target "
                f"{targets[code]['median']:.6g}, a relative residual of "
                f"{solution['residual_relative']:+.4f} outside the declared {tolerance}")
        if not entry["clears_floor"]:
            refusals.append(
                f"{code}: the solved dose {solution['dose']} gives a median per-step bias spread "
                f"of {spread['median']:.4g} nats, which is {entry['floor_margin_achieved']:.2f}x "
                f"the declared {args.noise_floor} nat floor and below the required "
                f"{args.floor_margin}x. This dose is degenerate; it is refused rather than run "
                "and discounted")
        solved[code] = entry

    return {
        "schema": SCHEMA,
        "status": "refused" if refusals else "solved",
        "comparable_statistic": COMPARABLE,
        "decision_statistic": (
            "median over the evaluation domains of each domain's wild-type-basis kl_per_residue; "
            "the domain is the independence unit, and one dose per arm matches one number per arm"),
        "basis": (
            "each domain's own wild-type token path, walked through the sampler's incremental "
            "cache at batch extent one, for both models. The reference model's recorded in-line "
            "strengths are pool-basis and are reported as provenance only"),
        "incremental_replay": INCREMENTAL_REPLAY_DECLARATION,
        "arm_codes": {code: ARM_CODES[name] for code, name in SOLVED_ARMS.items()},
        "domains": list(domains),
        "reference": {
            "model": sides["reference"]["model"], "dtype": sides["reference"]["dtype"],
            "logit_width": sides["reference"]["logit_width"],
            "frozen_doses": frozen,
            "protocol": {"path": str(args.reference_protocol),
                         "sha256": sha256(protocol_bytes).hexdigest()},
            "wild_type_basis_targets": targets,
            "recorded_inline_pool_medians": (
                None if args.reference_cells is None
                else inline_pool_medians(args.reference_cells, args.reference_model, domains, frozen)),
        },
        "target": {"model": sides["target"]["model"], "dtype": sides["target"]["dtype"],
                   "logit_width": sides["target"]["logit_width"]},
        "curve": curve,
        "solved": solved,
        "refusals": refusals,
        "family_reference": {
            "directory": str(args.family_reference_dir),
            "domains": {name: {"length": family.length, "qualified": family.qualified,
                               "npz_sha256": family.sidecar["npz_sha256"]}
                        for name, family in sorted(families.items())},
        },
        "permutation_seed": permutation_seed,
        "invocation": {"device": device, "out": str(Path(args.out).resolve())},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True, help="cpu or cuda:N")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--family-reference-dir", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path, required=True,
                        help="the reference model's frozen protocol; its strengths and roles are read")
    parser.add_argument("--reference-cells", type=Path, default=None,
                        help="optional directory of the reference campaign's cells, for the "
                             "recorded in-line pool medians reported as provenance")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-dtype", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-dtype", required=True)
    parser.add_argument("--noise-floor", type=float, required=True,
                        help="measured noise floor of the target's forward, nats")
    parser.add_argument("--floor-margin", type=float, required=True,
                        help="multiple of the floor the per-step bias spread must exceed")
    parser.add_argument("--relative-tolerance", type=float, required=True,
                        help="declared relative tolerance on the matched statistic")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    collect = _load_script("decoding/collect_guided_decoding_pool.py")
    out = collect.prepare_output_dir(args.out)
    result = run(args)
    path = out / OUTPUT_FILENAME
    write_json(path, result)
    (out / f"{OUTPUT_FILENAME}.sha256").write_text(
        f"{sha256(path.read_bytes()).hexdigest()}  {OUTPUT_FILENAME}\n")
    print(json.dumps({"output": str(path), "status": result["status"],
                      "solved": {code: entry.get("dose") for code, entry in result["solved"].items()},
                      "refusals": result["refusals"]}, indent=1))
    return 1 if result["refusals"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
