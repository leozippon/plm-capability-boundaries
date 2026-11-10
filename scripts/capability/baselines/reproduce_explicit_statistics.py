#!/usr/bin/env python3
"""Prove the cohort-parameterised ladder reproduces the completed R1 run.

The explicit-statistics producer was generalised so that one code path serves
R1 and any other wild-type cohort. Nothing new may be produced through it until
it is shown to return R1's own published numbers, so this stage replays them and
writes one artefact saying by how much.

Five checks, each exact where exactness is available and measured where it is
not:

``cohort``
    Resolve ``r1`` through the cohort descriptor and compare the resulting
    variant rows -- identifier, assay, mutation, substitutions and the recomputed
    LOOKUP delta -- against the frozen export's own ``variant_table.json``.

``support``
    Rebuild the selected wild types' homologous support through the generalised
    reader at the declared depth cap and require the alignment and weight files
    to be byte-identical to the frozen export's, by digest.

``potts_replay``
    Score every R1 variant of a wild type whose fitted parameters are retained
    locally, and compare with the published ``potts_delta``. The parameters of
    ``q00025`` are byte-identical to the cluster run's, so that comparison is
    exact; ``q00126`` was refitted locally from the same alignment, weights and
    regularization on a different host, so its difference measures plmc's
    cross-host reproducibility and is reported rather than asserted.

``potts_refit``
    Refit declared cheap wild types from scratch through the generalised fit
    path, one on each regularization branch, and compare the delivered scores
    with the published ones.

``gemme_replay``
    Rebuild the GEMME plan through the generalised path and require the mutation
    list handed to the official entry point to be byte-identical to the retained
    one, then re-aggregate the three retained official runs and compare with the
    published table. GEMME exposes no seed, so a rerun is not reproducible and is
    not claimed to be: what is replayed is everything around the stochastic step.

``gemme_additivity``
    The official ``-m`` multi-mutant score is the sum of its single-site scores.
    Every R1 double mutant whose two single substitutions were scored in the same
    runs is checked against that sum, which is what makes the interaction lane's
    accepted limitation a measurement rather than a reading of the source.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.capability.baselines import r1_msa_baselines as R  # noqa: E402
from scripts.capability.baselines import score_r1_gemme as G  # noqa: E402
from src.capability.baselines import msa_baselines as B  # noqa: E402
from src.capability.context.homology import ALIGNMENT_FIELDS, parse_hits  # noqa: E402

#: Declared before the gate was run. ``q00025`` and ``q00126`` are the only R1
#: wild types whose fitted parameters are retained on this host; ``q00174`` and
#: ``q00027`` are the cheapest cross-validated backgrounds, so refitting them
#: exercises the whole lambda sweep for seconds; ``q00011`` and ``q00017`` carry
#: double mutants whose single substitutions were scored in the same runs.
GATE_WILDTYPES = ("q00011", "q00017", "q00025", "q00027", "q00126", "q00174")
REFIT_WILDTYPES = ("q00025", "q00027", "q00174")

#: A replayed score must match the published one to the last bit where the model
#: is the same bytes. Where the model was refitted, the tolerance is plmc's own
#: optimizer termination, not a modelling choice, and the measured difference is
#: reported beside it.
EXACT = 0.0
REFIT_TOLERANCE = 1e-6


def published(root: Path) -> dict:
    scores = json.loads((root / "potts-explicit-complete" / "scores.json").read_text())
    gemme = json.loads((root / "gemme" / "scores.json").read_text())
    return {
        "potts": {row["sample_id"]: row for row in scores["samples"]},
        "potts_backgrounds": scores["backgrounds"],
        "gemme": {row["sample_id"]: row for row in gemme["samples"]},
        "gemme_wildtypes": gemme["wildtypes"],
    }


def check_cohort(cohort, frozen: dict) -> dict:
    """The generalised reader's rows against the frozen export's own table."""

    by_id = {row["sample_id"]: row for row in frozen["samples"]}
    if len(by_id) != len(frozen["samples"]):
        raise RuntimeError("the frozen variant table holds a duplicate sample_id")
    rebuilt = {row["sample_id"]: row for row in cohort.rows}
    missing = sorted(set(by_id) - set(rebuilt))
    extra = sorted(set(rebuilt) - set(by_id))
    drift = []
    for sample_id, row in rebuilt.items():
        original = by_id.get(sample_id)
        if original is None:
            continue
        for field in ("assay", "wildtype_id", "mutation", "n_substitutions", "variant_index",
                      "original_index"):
            if row[field] != original[field]:
                drift.append(f"{sample_id}.{field}")
        if [list(token) for token in row["substitutions"]] != [
                list(token) for token in original["substitutions"]]:
            drift.append(f"{sample_id}.substitutions")
    return {
        "rows_rebuilt": len(rebuilt),
        "rows_frozen": len(by_id),
        "missing": missing[:5],
        "unexpected": extra[:5],
        "field_drift": drift[:5],
        "counts_match": cohort.counts()["samples"] == frozen["counts"]["samples"],
        "passed": not missing and not extra and not drift,
    }


def check_profile_rung(cohort, frozen: dict, log: R.Log) -> dict:
    """The recomputed LOOKUP delta against the frozen table's own values."""

    deltas = R.profile_rung(cohort, log)
    frozen_deltas = {row["sample_id"]: row["profile_delta"] for row in frozen["samples"]}
    worst, worst_id = 0.0, None
    for sample_id, value in deltas.items():
        difference = abs(value - frozen_deltas[sample_id])
        if difference > worst:
            worst, worst_id = difference, sample_id
    return {"samples": len(deltas), "max_absolute_difference": worst, "at": worst_id,
            "passed": len(deltas) == len(frozen_deltas) and worst <= EXACT}


def check_support(cohort, support: dict, selected, scratch: Path, log: R.Log) -> dict:
    """Rebuild the selected wild types' support and compare the exported bytes."""

    wanted = {entry.query_id: entry for entry in cohort.references
              if entry.wildtype_id in selected}
    hits: dict[str, list] = {name: [] for name in wanted}
    for hit in parse_hits(cohort.hits, fields=ALIGNMENT_FIELDS):
        if hit.query in hits:
            hits[hit.query].append(hit)
    scratch.mkdir(parents=True, exist_ok=True)
    rebuilt = {}
    for query_id, entry in sorted(wanted.items()):
        name = entry.wildtype_id
        expectation = R.comparable_store(cohort, query_id)
        rows, weights, _, statistics_ = B.reconstruct_support(
            entry.sequence, query_id, hits[query_id], max_sequences=cohort.max_sequences,
            expect=expectation)
        fasta, weight_path = scratch / f"{name}.fa", scratch / f"{name}.weights"
        B.write_query_first_alignment(fasta, weight_path, entry.sequence, rows, weights)
        frozen = support[name]
        rebuilt[name] = {
            "rows": statistics_["rows"],
            "rows_frozen": frozen["rows"],
            "profile_cross_checked": statistics_["frequencies_verified"],
            "msa_identical": R.digest(fasta) == frozen["msa_sha256"],
            "weights_identical": R.digest(weight_path) == frozen["weights_sha256"],
            "neff_difference": abs(statistics_["neff_at_30_percent_identity"]
                                  - frozen["neff_at_30_percent_identity"]),
        }
        log.write(f"gate support {name} identical="
                  f"{rebuilt[name]['msa_identical'] and rebuilt[name]['weights_identical']}")
    return {
        "wildtypes": rebuilt,
        "passed": all(entry["msa_identical"] and entry["weights_identical"]
                      and entry["rows"] == entry["rows_frozen"] for entry in rebuilt.values()),
    }


def _score_from_params(params: Path, support: dict, rows: list[dict]) -> dict[str, float]:
    from src.capability.stability.pairwise_stability import read_plmc

    wildtype = B.read_fasta(ROOT / support["msa"])[0][1]
    scorer = B.potts_scorer(read_plmc(params), wildtype)
    return {row["sample_id"]: scorer.delta(B.mutant_codes(wildtype, row["substitutions"]))
            for row in rows}


def _difference(scores: dict[str, float], reference: dict) -> dict:
    worst, worst_id = 0.0, None
    for sample_id, value in scores.items():
        published_value = reference["potts"][sample_id]["potts_delta"]
        difference = abs(value - published_value)
        if difference > worst:
            worst, worst_id = difference, sample_id
    return {"samples": len(scores), "max_absolute_difference": worst, "at": worst_id}


def _retained_status(params: Path) -> str | None:
    """The recorded outcome of the fit that wrote this retained parameter file.

    A retained parameter file sits beside the checkpoint that produced it, and
    that checkpoint says whether the delivered model converged. The distinction
    decides what a difference means: a comparison against a fit that stopped at
    its iteration cap measures non-convergence, not the fitter's run-to-run
    spread, and pooling the two would overstate the instrument's precision bound
    by more than an order of magnitude.
    """

    checkpoint = params.parents[1] / "backgrounds" / f"{params.stem}.json"
    if not checkpoint.is_file():
        return None
    return json.loads(checkpoint.read_text()).get("status")


def check_potts_replay(cohort, support: dict, reference: dict, root: Path, log: R.Log) -> dict:
    """Published scores, rescored from the parameter files retained on this host."""

    retained = {}
    for tree in sorted(path for path in root.glob("potts*/params") if path.is_dir()):
        for params in sorted(tree.glob("*.params")):
            retained.setdefault(params.stem, []).append(params)
    outcome = {}
    for name, candidates in sorted(retained.items()):
        if name not in support:
            continue
        rows = cohort.by_reference[name]
        recorded = (reference["potts_backgrounds"].get(name) or {}).get("params") or {}
        for params in candidates:
            found = R.digest(params)
            entry = _difference(_score_from_params(params, support[name], rows), reference)
            entry["params"] = str(params.relative_to(ROOT))
            entry["params_sha256"] = found
            entry["is_published_model"] = found == recorded.get("sha256")
            entry["recorded_status"] = _retained_status(params)
            entry["exact"] = entry["max_absolute_difference"] <= EXACT
            outcome.setdefault(name, []).append(entry)
            log.write(f"gate potts-replay {name} {params.name} published_model="
                      f"{entry['is_published_model']} max_diff="
                      f"{entry['max_absolute_difference']:.3e}")
    same = [entry for entries in outcome.values() for entry in entries
            if entry["is_published_model"]]
    other = [entry for entries in outcome.values() for entry in entries
             if not entry["is_published_model"]]
    converged = [entry for entry in other if entry["recorded_status"] in B.COMPLETE]
    return {
        "wildtypes": outcome,
        "models_identical_to_published": len(same),
        "models_identical_to_published_exact": all(entry["exact"] for entry in same),
        "cross_host_refit_models": len(converged),
        "cross_host_refit_max_difference": max(
            (entry["max_absolute_difference"] for entry in converged), default=None),
        "non_converged_models": [
            {"params": entry["params"], "recorded_status": entry["recorded_status"],
             "max_absolute_difference": entry["max_absolute_difference"]}
            for entry in other if entry["recorded_status"] not in B.COMPLETE],
        "passed": bool(same) and all(entry["exact"] for entry in same),
    }


def check_potts_refit(args, cohort, support: dict, reference: dict, scratch: Path,
                      log: R.Log) -> dict:
    """Fit the declared cheap wild types again, through the generalised fit path."""

    outcome = {}
    for name in args.refit:
        if name not in support:
            raise SystemExit(f"{name} carries no exported support in {args.root}")
        rows = cohort.by_reference[name]
        job = {
            "wildtype_id": name,
            "wildtype": B.read_fasta(ROOT / support[name]["msa"])[0][1],
            "support": support[name],
            "variants": [{"sample_id": row["sample_id"],
                          "substitutions": [tuple(token) for token in row["substitutions"]]}
                         for row in rows],
            "msa": str(ROOT / support[name]["msa"]),
            "weights": str(ROOT / support[name]["weights"]),
            "msa_sha256": support[name]["msa_sha256"],
            "weights_sha256": support[name]["weights_sha256"],
            "plmc": str(ROOT / R.PLMC),
            "plmc_record": R.plmc_record(),
            "work": str(scratch / "work" / name),
            "params": str(scratch / "params" / f"{name}.params"),
            "threads": args.threads,
        }
        record = B.fit_background(job)
        reference_background = reference["potts_backgrounds"][name]
        entry = _difference(record["sample_scores"], reference)
        entry |= {
            "status": record["status"],
            "status_published": reference_background["status"],
            "selected_lambda_e": record["selected_lambda_e"],
            "selected_lambda_e_published": reference_background["selected_lambda_e"],
            "policy": record["regularization_policy"],
            "policy_published": reference_background["regularization_policy"],
            "params_sha256": record["params"]["sha256"],
            "params_sha256_published": (reference_background.get("params") or {}).get("sha256"),
            "seconds": sum(attempt["seconds"]
                           for block in list(record.get("sweep", []))
                           + [record.get("final", {})]
                           for attempt in block.get("attempts", [])),
        }
        entry["params_identical"] = entry["params_sha256"] == entry["params_sha256_published"]
        entry["within_tolerance"] = entry["max_absolute_difference"] <= REFIT_TOLERANCE
        outcome[name] = entry
        log.write(f"gate potts-refit {name} lambda={entry['selected_lambda_e']} "
                  f"params_identical={entry['params_identical']} "
                  f"max_diff={entry['max_absolute_difference']:.3e} "
                  f"seconds={entry['seconds']:.1f}")
    return {
        "wildtypes": outcome,
        "tolerance": REFIT_TOLERANCE,
        "passed": all(entry["within_tolerance"]
                      and entry["status"] == entry["status_published"]
                      and entry["selected_lambda_e"] == entry["selected_lambda_e_published"]
                      and entry["policy"] == entry["policy_published"]
                      for entry in outcome.values()),
    }


def _published_run(root: Path, name: str, run_key: str) -> tuple[Path, dict] | None:
    """The retained checkpoint the published table was assembled from.

    A wild type can hold checkpoints of more than one run key, because the key
    changes with the image digest and the runtime's own schema. Only the one the
    published score names may be replayed against that score.
    """

    for checkpoint in sorted((root / "gemme" / name).glob("*/checkpoint.json")):
        record = json.loads(checkpoint.read_text())
        if record.get("status") == "complete" and record.get("run_key") == run_key:
            return checkpoint.parent, record
    return None


def check_gemme_replay(cohort, support: dict, selected, reference: dict, root: Path,
                       log: R.Log) -> dict:
    """The mutation list and the aggregate, replayed from the retained official runs."""

    outcome = {}
    for name in sorted(selected):
        key = (reference["gemme_wildtypes"].get(name) or {}).get("run_key")
        found = None if key is None else _published_run(root, name, key)
        if found is None:
            outcome[name] = {"replayed": False,
                             "reason": f"no retained complete run of the published key {key}"}
            continue
        directory, checkpoint = found
        entry = G.plan_wildtype(support[name], cohort.by_reference[name])
        # The checkpoint names the file the official entry point was given; the
        # retained layout puts it beside the repeats of one attempt, so it is
        # read from the record rather than guessed at.
        mutants = Path(checkpoint["mutants_file"])
        if not mutants.is_absolute():
            mutants = ROOT / mutants
        retained = mutants.read_text()
        rebuilt = "\n".join(entry["mutations"]) + "\n"
        aggregate = G.aggregate(entry, checkpoint)
        worst, worst_id = 0.0, None
        for variant in aggregate["variants"]:
            row = reference["gemme"][variant["sample_id"]]
            for field in ("gemme_score", "gemme_sd", "gemme_run1", "gemme_run2", "gemme_run3"):
                difference = abs(variant[field] - row[field])
                if difference > worst:
                    worst, worst_id = difference, f"{variant['sample_id']}.{field}"
        outcome[name] = {
            "replayed": True,
            "run_key": checkpoint["run_key"],
            "mutations": len(entry["mutations"]),
            "mutants_file": str(mutants.relative_to(ROOT)),
            "mutation_list_identical": rebuilt == retained,
            "alignment_identical": support[name]["msa_sha256"] == checkpoint["alignment_sha256"],
            "variants": len(aggregate["variants"]),
            "max_absolute_difference": worst,
            "at": worst_id,
            "degenerate": aggregate["degenerate"],
            "exact": worst <= EXACT,
        }
        log.write(f"gate gemme-replay {name} list_identical="
                  f"{outcome[name]['mutation_list_identical']} max_diff={worst:.3e}")
    replayed = [entry for entry in outcome.values() if entry["replayed"]]
    return {
        "wildtypes": outcome,
        "replayed": len(replayed),
        "passed": bool(replayed) and all(
            entry["mutation_list_identical"] and entry["alignment_identical"] and entry["exact"]
            for entry in replayed),
    }


def check_gemme_additivity(reference: dict) -> dict:
    """Every R1 double mutant whose two singles were scored in the same runs."""

    by_wildtype: dict[str, dict[str, dict]] = {}
    for row in reference["gemme"].values():
        by_wildtype.setdefault(row["wildtype_id"], {})[row["mutation"]] = row
    pairs, worst, worst_at = [], 0.0, None
    for name, mutations in sorted(by_wildtype.items()):
        for mutation, row in sorted(mutations.items()):
            if row["n_substitutions"] < 2:
                continue
            sites = mutation.split(":")
            if any(site not in mutations for site in sites):
                continue
            for field in ("gemme_run1", "gemme_run2", "gemme_run3"):
                difference = abs(row[field] - sum(mutations[site][field] for site in sites))
                if difference > worst:
                    worst, worst_at = difference, f"{name}/{mutation}.{field}"
            pairs.append({"wildtype_id": name, "mutation": mutation, "sites": sites})
    return {
        "multi_mutants_with_every_single_scored": len(pairs),
        "max_absolute_difference_from_the_sum": worst,
        "at": worst_at,
        "official_code": "pred.R normalizePred/normalizePredWithNbSeqs: "
                         "normPred[i] = sum(predTMP * trace[myPos]), one term per mutated site "
                         "and no cross-term",
        "consequence": R.GEMME_ADDITIVITY,
        "passed": bool(pairs) and worst < 1e-12,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", default="r1")
    parser.add_argument("--config", type=Path, default=R.CONFIG)
    parser.add_argument("--root", type=Path, default=None,
                        help="the frozen export to replay; defaults to the cohort's own")
    parser.add_argument("--wildtypes", nargs="*", default=list(GATE_WILDTYPES))
    parser.add_argument("--refit", nargs="*", default=list(REFIT_WILDTYPES))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=Path("logs/operations/explicit-statistics-cohorts-20261109"))
    arguments = parser.parse_args()
    out = arguments.out if arguments.out.is_absolute() else ROOT / arguments.out
    log = R.Log(out / "reproduction-gate.log")
    namespace = argparse.Namespace(cohort=arguments.cohort, config=arguments.config)
    cohort = R.load_cohort(namespace)
    root = arguments.root or cohort.out
    root = root if root.is_absolute() else ROOT / root
    frozen = json.loads((root / "inputs" / "variant_table.json").read_text())
    support = json.loads((root / "inputs" / "support.json").read_text())["wildtypes"]
    reference = published(root)
    selected = [name for name in arguments.wildtypes if name in support]
    unknown = sorted(set(arguments.wildtypes) - set(selected))
    if unknown:
        raise SystemExit(f"{unknown} carry no exported support in {root}")
    scratch = out / "gate-scratch"
    log.write(f"gate cohort={cohort.name} root={root} wildtypes={selected} refit={arguments.refit}")

    checks = {
        "cohort": check_cohort(cohort, frozen),
        "profile_rung": check_profile_rung(cohort, frozen, log),
        "support": check_support(cohort, support, set(selected), scratch / "support", log),
        "potts_replay": check_potts_replay(cohort, support, reference, root, log),
        "potts_refit": check_potts_refit(arguments, cohort, support, reference,
                                         scratch / "refit", log),
        "gemme_replay": check_gemme_replay(cohort, support, selected, reference, root, log),
        "gemme_additivity": check_gemme_additivity(reference),
    }
    document = {
        "schema": "explicit_statistics_reproduction_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort.name,
        "replayed_export": str(root.relative_to(ROOT)),
        "code": R.code_record(),
        "plmc": R.plmc_record(),
        "selected": selected,
        "refit": list(arguments.refit),
        "exact_tolerance": EXACT,
        "refit_tolerance": REFIT_TOLERANCE,
        "checks": checks,
        "passed": all(entry["passed"] for entry in checks.values()),
    }
    R.save(out / "reproduction-gate.json", document)
    log.write(f"gate {'PASSED' if document['passed'] else 'FAILED'}: "
              + json.dumps({name: entry["passed"] for name, entry in checks.items()}))
    print(json.dumps({name: entry["passed"] for name, entry in checks.items()}, indent=1))
    raise SystemExit(0 if document["passed"] else 1)


if __name__ == "__main__":
    main()
