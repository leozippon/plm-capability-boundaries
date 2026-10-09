#!/usr/bin/env python3
"""Single-mutation effect landscapes: generated products against matched naturals.

E15. On natural proteins this project measured where a single substitution moves
a model's likelihood: the site's own term, the one-sided downstream propagation a
left-to-right model must have, and the log-linear decay of that propagation with
sequence separation (half-distance 29.9 residues for ProGen2-small, 21.1 for
GPT-2-large). This stage asks whether the same profile holds on proteins the
models themselves generated.

The comparison is controlled, not descriptive. Every generated product in the
cohort is paired one-to-one with a length-matched natural sequence scored by the
same arm in the same run, the independence group of the generated member is the
bootstrap unit of the pair, and the realised length and separation imbalances are
reported beside every estimate. Three readings are produced for each arm:

* the scalar endpoints, which need no token alignment -- the absolute likelihood
  difference of a substitution, and the wild type's own negative log likelihood
  per *residue*, which is the predictability covariate that would explain a
  difference in propagation without anything about mutation at all. Per residue
  rather than per token, so that a merged-piece interface's segmentation of a
  repetitive sequence cannot move the covariate on its own;
* the position-resolved endpoints -- the site's own term, the summed absolute
  downstream response, and the share of the total the downstream part carries --
  on the mutations whose token grid is shared between the two states, with the
  retained fraction reported because a merged-piece interface loses some;
* the propagation profile and its decay fit, for each origin, plus the
  generated-minus-natural contrast inside each separation stratum.

The degeneracy sensitivity repeats the primary contrast on the non-degenerate
generated products and their partners. A homopolymeric run is trivially
predictable and would move every quantity here for reasons that have nothing to
do with being generated; the degenerate products stay in the outcome-blind
primary estimate and are reported separately rather than quietly removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.interactions import generated_mutation as gm  # noqa: E402
from src.capability.interactions.contact_enrichment import (  # noqa: E402
    ADDITIVE_RESPONSE_BINS,
    cross_fit_binned_residuals,
)
from src.capability.position.contact_response import ProfileAccumulator  # noqa: E402

COMPLETION = "generated_mutation_analysis.json"
SCHEMA = "generated_mutation_analysis_v1"

#: Scalar endpoints, defined without any token alignment.
SCALAR_ENDPOINTS = ("absolute_likelihood_nats", "wild_type_nll_per_residue_nats")

#: Position-resolved endpoints, defined only on an aligned mutation.
ALIGNED_ENDPOINTS = ("site_term_nats", "downstream_absolute_nats", "downstream_share")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _site_of(label: str) -> int:
    if ":" in label:
        raise ValueError(f"{label}: a single-substitution scan carries no multi-site state")
    return int(label[1:-1]) - 1


def profile_control_block(rows: list, strata: list, control: dict, assays: dict, contrasts,
                          *, draws: int, seed: int) -> dict:
    """The evolutionary-profile control, three ways, with its own support counts.

    The asymmetry this controls for is real and not subtle: a natural protein has
    relatives in a sequence corpus and a generated product mostly does not, so
    "the model responds less to mutating a generated protein" could simply be
    "the model has no family knowledge to lose". Three readings, in increasing
    strength and decreasing support:

    *retrievability* quotes the control stage's own census, which is the
    measurement that the asymmetry exists at all;

    *the channel itself* is the generated-minus-natural contrast of the
    mutation-local LOOKUP score -- if the evolutionary statistic does not differ
    between the origins, it cannot explain a difference in likelihood response;

    *both without a profile* re-estimates every primary endpoint on the pairs
    where **neither** member has a retrievable profile. A likelihood contrast
    that survives there is not an evolutionary-retrieval effect, and this is the
    cleanest of the three because it removes the channel rather than adjusting
    for it;

    *LOOKUP-adjusted* residualises each endpoint on the mutation-local LOOKUP
    score with the project's own leave-one-group-out binned residualizer. It is
    reported last and with its support, because it is defined only where a
    LOOKUP score exists, which is exactly where the asymmetry is weakest.
    """

    scores = {
        (row["sequence_id"], row["mutation"]): row["lookup_score_nats"]
        for row in control["mutations"]
    }
    profiled = {
        row["sequence_id"]: row["profile"] is not None for row in control["sequences"]
    }
    for row in rows:
        row["lookup_score_nats"] = scores.get((row["sequence_id"], row["mutation"]))

    both_absent = {
        identity
        for identity in profiled
        if not profiled[identity]
        and not profiled.get(assays.get(identity, {}).get("paired_with"), True)
    }
    block: dict = {
        "retrievability": control.get("summary"),
        "lookup_score_nats": gm.paired_origin_contrast(
            rows, value="lookup_score_nats", draws=draws, seed=seed
        ),
        "sequences_with_a_profile": {
            origin: sum(
                1 for row in control["sequences"]
                if row["origin"] == origin and row["profile"] is not None
            )
            for origin in gm.ORIGINS
        },
    }
    block["lookup_score_nats"]["precision"] = gm.precision_record(
        block["lookup_score_nats"]["difference"]
    )

    without = [row for row in rows if row["sequence_id"] in both_absent]
    block["both_without_profile"] = dict(
        contrasts(without, [row for row in strata if row["sequence_id"] in both_absent]),
        sequences=len(both_absent),
        reading=(
            "neither member of these pairs has a retrievable evolutionary profile, so a "
            "contrast here cannot be an evolutionary-retrieval effect"
        ),
    ) if both_absent else {
        "undefined": "no matched pair has both members without a retrievable profile"
    }

    scored = [row for row in rows if row.get("lookup_score_nats") is not None]
    if len({row["group"] for row in scored}) >= 2:
        covariate = np.asarray([row["lookup_score_nats"] for row in scored], dtype=np.float64)
        groups = np.asarray([row["group"] for row in scored])
        adjusted = {}
        for endpoint in SCALAR_ENDPOINTS + ALIGNED_ENDPOINTS:
            usable = [row for row in scored if row[endpoint] is not None]
            if len({row["group"] for row in usable}) < 2:
                adjusted[endpoint] = {"undefined": "too few groups carry this endpoint"}
                continue
            mask = np.asarray([row[endpoint] is not None for row in scored])
            residual = cross_fit_binned_residuals(
                covariate[mask],
                np.asarray([row[endpoint] for row in usable], dtype=np.float64),
                groups[mask],
                ADDITIVE_RESPONSE_BINS,
            )
            shaped = [dict(row, residual=float(value)) for row, value in zip(usable, residual)]
            estimate = gm.paired_origin_contrast(
                shaped, value="residual", draws=draws, seed=seed
            )
            estimate["precision"] = gm.precision_record(estimate["difference"])
            estimate["mutations"] = len(usable)
            adjusted[endpoint] = estimate
        block["lookup_adjusted"] = dict(
            adjusted,
            bins=ADDITIVE_RESPONSE_BINS,
            method=(
                "endpoint minus the mean endpoint of its LOOKUP-score quantile bin, the "
                "curve fit on every independence group except the row's own"
            ),
        )
    else:
        block["lookup_adjusted"] = {
            "undefined": "fewer than two independence groups carry a LOOKUP score"
        }
    return block


def triad_block(rows: list, strata: list, contrasts, *, draws: int, seed: int) -> dict:
    """The three contrasts of the retrievability-controlled design, and their identity.

    The original comparison carried two effects at once. Here the generation
    effect is read against a natural arm in the same identity band, the
    retrievability effect is read between two natural arms, and the original
    contrast is read unchanged. Because every triple is retained only when both
    natural partners exist, the three are on one group support and the third
    point estimate is arithmetically the sum of the first two -- which is checked,
    not assumed, and a nonzero residual means the supports have diverged.
    """

    out: dict = {"contrasts": {}}
    for left, right, question in gm.TRIAD_CONTRASTS:
        key = f"{left}__minus__{right}"
        selected = gm.relabelled_pair(rows, left, right)
        selected_strata = gm.relabelled_pair(strata, left, right)
        record = contrasts(selected, selected_strata)
        record["question"] = question
        record["left"], record["right"] = left, right
        out["contrasts"][key] = record

    residuals = {}
    for endpoint in SCALAR_ENDPOINTS + ALIGNED_ENDPOINTS:
        def point(pair: tuple[str, str]) -> float | None:
            record = out["contrasts"][f"{pair[0]}__minus__{pair[1]}"][endpoint]
            return record["difference"]["point"]

        parts = [
            point(("generated", "natural_low_homology")),
            point(("natural_low_homology", "natural_high_homology")),
            point(("generated", "natural_high_homology")),
        ]
        if any(value is None for value in parts):
            residuals[endpoint] = {"checked": False, "reason": "a contrast has no point estimate"}
            continue
        residuals[endpoint] = {
            "checked": True,
            "generation_plus_retrievability": parts[0] + parts[1],
            "original_contrast": parts[2],
            "residual": parts[0] + parts[1] - parts[2],
        }
    out["decomposition_identity"] = residuals
    out["reading"] = (
        "the generation contrast is the one that answers whether a model behaves "
        "differently on its own products once retrievability is matched; the "
        "retrievability contrast measures how much of the original difference was "
        "reference-database coverage rather than generation"
    )
    return out


def analyse_arm(extraction: gm.Extraction, assays: dict, control: dict | None,
                *, draws: int, seed: int, triad: bool = False) -> dict:
    """Every endpoint of one arm over the whole cohort."""

    per_mutation: list[dict] = []
    origins = gm.TRIAD_ORIGINS if triad else gm.ORIGINS
    profiles = {origin: ProfileAccumulator(direction="downstream") for origin in origins}
    cells: dict[tuple, dict[str, float]] = {}
    aligned = unaligned = 0
    worst_upstream = 0.0
    unaligned_reasons: dict[str, int] = {}
    present, absent = extraction.covered(sorted(assays))
    unexplained = [row for row in absent if "without a recorded reason" in row["reason"]]
    if unexplained:
        raise SystemExit(
            f"{extraction.arm}: {len(unexplained)} cohort assays are absent from the extraction "
            f"without a recorded reason, first {[row['assay'] for row in unexplained][:5]}"
        )

    for assay in present:
        row = assays[assay]
        payload = extraction.payload(assay)
        mutants = [str(value) for value in payload["mutants"]]
        if mutants != list(row["mutants"]):
            raise SystemExit(f"{assay}: archive mutant order differs from the cohort")
        states = extraction.states(payload, [row["wildtype"], *row["sequences"]])
        likelihoods = gm.state_likelihoods(payload)
        wild_terms = np.asarray(
            payload["position_nats"][
                int(payload["position_offsets"][0]) : int(payload["position_offsets"][1])
            ],
            dtype=np.float64,
        )
        shared = {
            "origin": row["origin"],
            "group": row["group"],
            "sequence_id": assay,
        }
        per_mutation.append(
            dict(
                shared,
                mutation="<wild-type>",
                absolute_likelihood_nats=None,
                wild_type_nll_per_residue_nats=float(wild_terms.sum()) / len(row["wildtype"]),
                site_term_nats=None,
                downstream_absolute_nats=None,
                downstream_share=None,
            )
        )
        misaligned_here: list[str] = []
        for index, label in enumerate(mutants):
            site = _site_of(label)
            census, reason = gm.mutation_receivers(
                payload,
                states=states,
                paradigm=extraction.paradigm,
                index=index,
                site=site,
            )
            record = dict(
                shared,
                mutation=label,
                absolute_likelihood_nats=abs(float(likelihoods[label])),
                wild_type_nll_per_residue_nats=None,
                site_term_nats=None,
                downstream_absolute_nats=None,
                downstream_share=None,
            )
            if census is None:
                unaligned += 1
                misaligned_here.append(label)
                unaligned_reasons[reason or "unstated"] = (
                    unaligned_reasons.get(reason or "unstated", 0) + 1
                )
                per_mutation.append(record)
                continue
            aligned += 1
            worst_upstream = max(worst_upstream, float(census["upstream_max_abs_nats"]))
            downstream = [item for item in census["receivers"] if item["direction"] == "downstream"]
            total = float(sum(abs(float(item["response"])) for item in downstream))
            site_term = abs(float(census["site_response"]))
            record["site_term_nats"] = site_term
            record["downstream_absolute_nats"] = total
            denominator = site_term + total
            record["downstream_share"] = (total / denominator) if denominator > 0 else None
            per_mutation.append(record)
            profiles[row["origin"]].add(downstream)
            profiles[row["origin"]].count_mutation()
            for item in downstream:
                separation = abs(int(item["separation"]))
                key = (row["group"], row["origin"], assay, gm.separation_stratum(separation))
                cell = cells.setdefault(key, {"absolute": 0.0, "separation": 0.0, "n": 0})
                cell["absolute"] += abs(float(item["response"]))
                cell["separation"] += float(separation)
                cell["n"] += 1
        extraction.agrees_on_alignment(assay, misaligned_here)

    if extraction.paradigm == "causal_next_token" and worst_upstream != 0.0:
        raise SystemExit(
            f"{extraction.arm}: upstream terms differ by {worst_upstream} nats on a causal arm"
        )

    stratum_rows = [
        {
            "group": group,
            "origin": origin,
            "sequence_id": assay,
            "stratum": stratum,
            "absolute_response": cell["absolute"] / cell["n"],
            "separation": cell["separation"] / cell["n"],
            "receivers": cell["n"],
        }
        for (group, origin, assay, stratum), cell in cells.items()
    ]

    def contrasts(rows: list[dict], strata: list[dict]) -> dict:
        record = {}
        for endpoint in SCALAR_ENDPOINTS + ALIGNED_ENDPOINTS:
            estimate = gm.paired_origin_contrast(rows, value=endpoint, draws=draws, seed=seed)
            estimate["precision"] = gm.precision_record(estimate["difference"])
            record[endpoint] = estimate
        record["profile"] = gm.profile_contrast(strata, draws=draws, seed=seed)
        for stratum in record["profile"]["strata"]:
            stratum["precision"] = gm.precision_record(stratum["difference"])
        return record

    non_degenerate = {
        assay
        for assay in present
        if not (
            assays[assay].get("degenerate")
            or assays.get(assays[assay].get("paired_with"), {}).get("degenerate")
        )
    }
    return {
        "arm": extraction.arm,
        "paradigm": extraction.paradigm,
        "dtype": extraction.completion["identity"]["dtype"],
        "assays": len(present),
        "assays_absent": absent,
        "complete_pairs": sum(
            1
            for assay in present
            if assays[assay]["origin"] == "generated" and assays[assay]["paired_with"] in present
        ),
        "mutations": {
            "aligned": aligned,
            "unaligned": unaligned,
            "aligned_share": (
                aligned / (aligned + unaligned) if aligned + unaligned else None
            ),
            "unaligned_reasons": unaligned_reasons,
        },
        "upstream_max_abs_nats": worst_upstream,
        "retention_max_abs_nats": float(
            extraction.completion["totals"]["retention_max_abs_nats"]
        ),
        "primary": (
            triad_block(per_mutation, stratum_rows, contrasts, draws=draws, seed=seed)
            if triad
            else contrasts(per_mutation, stratum_rows)
        ),
        "profile_control": (
            {
                "by_design": (
                    "this cohort matches retrievability by construction; the measured band "
                    "and admissible-relative count of every arm are in the cohort design's "
                    "retrievability census, so no post-hoc covariate adjustment is applied"
                )
            }
            if triad
            else profile_control_block(
                per_mutation, stratum_rows, control, assays, contrasts, draws=draws, seed=seed
            )
            if control is not None
            else {
                "undefined": (
                    "no evolutionary-profile control was supplied; the generated-versus-natural "
                    "contrast is therefore not controlled for sequence retrievability"
                )
            }
        ),
        "non_degenerate": dict(
            (
                triad_block(
                    [row for row in per_mutation if row["sequence_id"] in non_degenerate],
                    [row for row in stratum_rows if row["sequence_id"] in non_degenerate],
                    contrasts, draws=draws, seed=seed,
                )
                if triad
                else contrasts(
                    [row for row in per_mutation if row["sequence_id"] in non_degenerate],
                    [row for row in stratum_rows if row["sequence_id"] in non_degenerate],
                )
            ),
            sequences=len(non_degenerate),
        ),
        "decay": {origin: profiles[origin].profile() for origin in origins},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=Path, required=True,
                        help="the singles cohort this extraction was run on")
    parser.add_argument("--archives", type=Path, nargs="+", required=True,
                        help="one completed extraction directory per arm")
    parser.add_argument("--profile-control", type=Path,
                        help="the evolutionary-profile control table "
                             "(measure_generated_profile_control.py)")
    parser.add_argument("--draws", type=int, default=gm.BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=gm.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="cpu",
                        help="accepted because the campaign queue injects it; unused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws is positive")

    out = gm.prepare_output_directory(args.out, COMPLETION)
    cohort = json.loads(Path(args.cohort).read_text())
    if cohort.get("schema") != gm.COHORT_SCHEMA or cohort.get("mode") not in ("singles", "triad"):
        raise SystemExit(f"{args.cohort}: not a singles or triad cohort of {gm.COHORT_SCHEMA}")
    triad = cohort["mode"] == "triad"
    assays = {row["assay"]: row for row in cohort["assays"]}
    if len(assays) != len(cohort["assays"]):
        raise SystemExit("the cohort carries a duplicate assay identity")

    control = None
    if args.profile_control is not None:
        control = json.loads(Path(args.profile_control).read_text())
        if control.get("schema") != "generated_profile_control_v1":
            raise SystemExit(f"{args.profile_control}: unexpected schema")
        summary = Path(args.profile_control).parent / "generated_profile_control.json"
        if summary.is_file():
            control["summary"] = json.loads(summary.read_text())["summary"]

    arms = []
    for root in args.archives:
        extraction = gm.open_extraction(root)
        arms.append(analyse_arm(extraction, assays, control, draws=args.draws,
                                seed=args.seed, triad=triad))
    if len({record["arm"] for record in arms}) != len(arms):
        raise SystemExit("two extraction directories report the same arm")

    write_json(out / COMPLETION, {
        "status": "complete",
        "schema": SCHEMA,
        "created_utc": _now(),
        "question": (
            "do the single-mutation likelihood patterns a model shows on natural proteins "
            "also hold on proteins it generated?"
        ),
        "design": cohort["design"],
        "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
        "mode": cohort["mode"],
        "endpoints": {
            "scalar": list(SCALAR_ENDPOINTS),
            "aligned": list(ALIGNED_ENDPOINTS),
            "contrast": (
                "three paired contrasts decomposing generation from retrievability, "
                "independence group as the paired unit"
                if triad
                else "generated minus matched natural, independence group as the paired unit"
            ),
        },
        "bootstrap": {"draws": args.draws, "seed": args.seed,
                      "method": "group percentile bootstrap on the group-equal mean"},
        "arms": arms,
        "limitations": [
            "A generated product has no experimental measurement, so every quantity here is "
            "the model's own likelihood behaviour and not its accuracy.",
            "The natural partners are Swiss-Prot entries matched on length alone; composition, "
            "repetitiveness and structural content are not matched and the realised "
            "wild-type predictability of each origin is reported for that reason.",
            "Position-resolved endpoints are defined only where the two states share a token "
            "grid, so on a merged-piece interface they describe a tokenisation-selected "
            "subset of the mutations; the retained share travels with the estimate.",
            "Intervals condition on the extracted likelihoods and omit generation, sampling "
            "and checkpoint variation.",
            "If a generated product has no retrievable evolutionary profile at all, then "
            "'adjust the contrast for the mutation-local evolutionary statistic' is not a "
            "well-posed operation -- the covariate does not exist on one arm of a paired "
            "comparison, and the LOOKUP-adjusted block reports that rather than a number. "
            "The well-posed control in that case is the restriction to pairs where neither "
            "member has a profile, whose support is reported with it.",
        ],
        "code_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/capability/interactions/analyse_generated_mutation.py",
                "src/capability/interactions/generated_mutation.py",
                "src/capability/position/contact_response.py",
            )
        },
    })


if __name__ == "__main__":
    main()
