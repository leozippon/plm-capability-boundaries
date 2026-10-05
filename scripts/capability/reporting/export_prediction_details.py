#!/usr/bin/env python3
"""Export retained sample details, predictions and split receipts; never fit models.

Only --root source files are read. --out must be new or empty. Help is available
without inspecting the repository or importing numerical/model libraries.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import resource
import shutil

MUTATION = "results/R5/context_mutation_rescue_20260923/cohort.json"
ABUNDANCE = "results/R1/external_confirmation_20260924"
STABILITY = "results/R4/gate_stability_20260924"
REPLAY = "results/R4/stability_likelihood_replay_20260927"
READOUT = "results/R2/readout_expansion_20260923/complete/results/external_baseline"
LOCAL = "results/R5/local_context_20260923/20260923233257_e429ce7f31e4/lcgp_admission/local_context_measurement.json"
LINEAGES = "results/shared/followup_support_20260927/lineage_robustness.json"
SEEDS = (20260923, 20260924, 20260925)
FITTED = ("P", "M", "P_M", "B", "B_R", "R", "sequence", "profile_sequence")
RAW = ("raw_M", "raw_P")
PERMUTED = ("permuted_B", "permuted_B_R")
DESIGNS = FITTED + RAW + PERMUTED
MUTATION_ANALYSIS = "mutation_supporting_R2_anchor201"
LOCAL_ANALYSIS = "mutation_headline_R5_local_context"
ABUNDANCE_ANALYSIS = "abundance_R1_external_confirmation"
STABILITY_ANALYSIS = "stability_R4_primary_likelihood_replay"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pointer(*parts):
    return "".join("/" + str(x).replace("~", "~0").replace("/", "~1") for x in parts)


def sample_id(cohort_sha, unit, mutation):
    """Content-bound, unambiguous IDs; no dependence on row sorting or model."""
    return hashlib.sha256(dumps([cohort_sha, unit, mutation]).encode()).hexdigest()


def finite_vector(values, length, label):
    require(isinstance(values, list) and len(values) == length, f"unaligned vector: {label}")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values),
            f"nonfinite/non-numeric vector: {label}")


def standardized_ranks(values):
    """Same average-tie ranks and population SD as the retained producer.

    This is a label-scale conversion, not a prediction, split, fit or resample.
    """
    finite_vector(values, len(values), "measured")
    require(bool(values), "empty measured vector")
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[start]] == values[order[end]]:
            end += 1
        rank = (start + 1 + end) / 2
        for i in order[start:end]:
            ranks[i] = rank
        start = end
    mean = sum(ranks) / len(ranks)
    sd = math.sqrt(sum((v - mean) ** 2 for v in ranks) / len(ranks))
    return [(v - mean) / sd if sd else 0.0 for v in ranks]


def positions(mutation, wildtype):
    result = []
    for item in mutation.split(":"):
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", item)
        if match is None:
            raise ValueError(f"invalid mutation: {mutation}")
        before, pos, _ = match.groups()
        pos = int(pos)
        require(1 <= pos <= len(wildtype) and wildtype[pos - 1] == before,
                f"mutation disagrees with WT: {mutation}")
        require(pos not in result, f"duplicate position: {mutation}")
        result.append(pos)
    return ":".join(map(str, result))


def outer_mapping(folds, groups, *, held="held_groups", training="training_groups"):
    """Validate saved memberships without drawing or reconstructing any split."""
    groups = set(groups)
    mapping = {}
    fold_ids = set()
    for record in folds:
        fold = record["fold"]
        require(fold not in fold_ids, "duplicate outer fold")
        fold_ids.add(fold)
        test = record[held]
        require(len(test) == len(set(test)), "duplicate group within outer fold")
        require(set(test) <= groups, "unknown held group")
        require(not (set(test) & mapping.keys()), "overlapping outer held folds")
        if training in record:
            train = record[training]
            purge = record.get("purged_training_groups", [])
            require(len(train) == len(set(train)), "duplicate training group")
            require(not set(train) & set(test), "outer held/training overlap")
            require(not set(purge) & (set(train) | set(test)), "purge overlap")
            require(set(train) | set(test) | set(purge) == groups, "incomplete outer partition")
            if "inner_folds" in record:
                inner = [g for x in record["inner_folds"] for g in x["validation_families"]]
                require(len(inner) == len(set(inner)) and set(inner) == set(train),
                        "inner folds must partition outer training families")
        mapping.update({g: fold for g in test})
    require(set(mapping) == groups, "outer folds do not partition cohort groups")
    return mapping


def aligned_predictions(record, cohort, designs):
    require(record["mutants"] == cohort["mutants"], "exact mutation order mismatch")
    require(set(record) == {"assay", "mutants", *designs}, "unexpected prediction design/schema")
    for design in designs:
        finite_vector(record[design], len(cohort["mutants"]), design)


@contextmanager
def gzip_text(path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                yield text


class Table:
    def __init__(self, stream, fields):
        self.fields = fields
        self.writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        self.writer.writeheader()
        self.count = 0

    def row(self, **values):
        def cell(value):
            if value is None:
                return ""
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (dict, list)):
                return dumps(value)
            if isinstance(value, float):
                require(math.isfinite(value), "nonfinite CSV value")
            return value
        self.writer.writerow({k: cell(v) for k, v in values.items()})
        self.count += 1


class Export:
    def __init__(self, root, out, stack):
        self.root, self.out, self.stack = root, out, stack
        self.sources = {}
        self.tables = {}
        self.records = stack.enter_context(gzip_text(out / "provenance.jsonl.gz"))
        self.record_count = 0
        self.availability = self.table("availability", [
            "analysis_id", "cohort_sha256", "model_id", "seed", "design", "role",
            "prediction_status", "fold_status", "samples", "source_id", "source_pointer", "reason"])
        self.folds = self.table("fold-membership", [
            "analysis_id", "model_id", "seed", "design", "outer_fold", "inner_fold",
            "group_id", "role", "held_out", "source_id", "source_pointer"])

    def table(self, name, fields):
        table = Table(self.stack.enter_context(gzip_text(self.out / f"{name}.csv.gz")), fields)
        self.tables[name] = table
        return table

    def read(self, relative):
        path = self.root / relative
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        source_id = str(path.relative_to(self.root))
        self.sources[source_id] = {"source_id": source_id, "path": source_id,
                                   "sha256": sha, "bytes": len(raw)}
        data = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON token in {source_id}: {value}")))
        return data, source_id, sha

    def record(self, source, path, kind, data, **context):
        self.records.write(dumps(dict(source_id=source, source_pointer=path, kind=kind,
                                      data=data, **context)) + "\n")
        self.record_count += 1

    def failures(self, source, data, path=""):
        """Retain original failure/exclusion/skip fields, including nulls and counts."""
        if isinstance(data, dict):
            for key, value in data.items():
                child = path + pointer(key)
                if any(word in key.lower() for word in ("fail", "exclu", "skipp", "missing")):
                    self.record(source, child, "failure_exclusion_or_missing_record", value)
                else:
                    self.failures(source, value, child)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                if isinstance(value, (dict, list)):
                    self.failures(source, value, path + pointer(i))

    def metadata(self, source, data, exclude=()):
        self.record(source, "", "source_metadata", {k: v for k, v in data.items() if k not in exclude})
        self.failures(source, data)

    def availability_row(self, analysis, cohort_sha, model, seed, design, status,
                         source, path, *, role="recorded_design", folds="not_serialized", samples=None,
                         reason=""):
        require(status in {"retained", "absent_local_referenced_artifact", "not_serialized", "not_applicable"},
                "unknown availability status")
        self.availability.row(analysis_id=analysis, cohort_sha256=cohort_sha, model_id=model,
                              seed=seed, design=design, role=role, prediction_status=status,
                              fold_status=folds, samples=samples, source_id=source,
                              source_pointer=path, reason=reason)

    def saved_folds(self, analysis, model, seed, design, source, path, folds, groups,
                    held="held_groups", training="training_groups"):
        mapping = outer_mapping(folds, groups, held=held, training=training)
        self.record(source, path, "original_fold_records", folds, analysis_id=analysis,
                    model_id=model, seed=seed, design=design)
        for i, fold in enumerate(folds):
            for key, role in ((held, "held_out"), (training, "training"),
                              ("purged_training_groups", "purged")):
                for j, group in enumerate(fold.get(key, [])):
                    self.folds.row(analysis_id=analysis, model_id=model, seed=seed, design=design,
                                   outer_fold=fold["fold"], group_id=group, role=role,
                                   held_out=role == "held_out", source_id=source,
                                   source_pointer=path + pointer(i, key, j))
            for j, inner in enumerate(fold.get("inner_folds", [])):
                for k, group in enumerate(inner["validation_families"]):
                    self.folds.row(analysis_id=analysis, model_id=model, seed=seed, design=design,
                                   outer_fold=fold["fold"], inner_fold=j, group_id=group,
                                   role="inner_validation", held_out=False, source_id=source,
                                   source_pointer=path + pointer(i, "inner_folds", j, "validation_families", k))
        return mapping


def export_cohorts(ex, common, headline):
    mutation, ms, mh = ex.read(MUTATION)
    abundance, ads, ah = ex.read(f"{ABUNDANCE}/cohort.json")
    stability, ss, sh = ex.read(f"{STABILITY}/cohort/cohort.json")
    ex.metadata(ms, mutation, ("assays",))
    ex.metadata(ads, abundance, ("domains",))
    ex.metadata(ss, stability, ("backgrounds",))
    base = ["sample_id", "cohort_sha256", "unit_id", "protein_id", "family_id", "group_id",
            "mutation", "positions_1based", "cohort_member", "source_id", "source_pointer"]
    mt = ex.table("mutation-samples", base + ["measured", "profile_score", "rank_target",
                  "supporting_common_member", "headline_common_member", "mutant_digest", "variant_index"])
    assay_map = {}
    for i, assay in enumerate(mutation["assays"]):
        name = assay["assay"]
        require(name not in assay_map, "duplicate assay")
        assay_map[name] = assay
        n = len(assay["mutants"])
        require(n == len(set(assay["mutants"])), "duplicate mutation sample")
        finite_vector(assay["measured"], n, "measured")
        finite_vector(assay["profile_scores"], n, "profile_scores")
        ranks = standardized_ranks(assay["measured"])
        ex.record(ms, pointer("assays", i), "mutation_unit_metadata",
                  {k: v for k, v in assay.items() if k not in
                   ("wildtype", "sequences", "mutants", "measured", "profile_scores",
                    "homolog_candidates", "unrelated_candidates")})
        for j, mutant in enumerate(assay["mutants"]):
            mt.row(sample_id=sample_id(mh, name, mutant), cohort_sha256=mh, unit_id=name,
                   protein_id=assay["wildtype_id"], family_id=assay["cluster"], group_id=assay["cluster"],
                   mutation=mutant, positions_1based=positions(mutant, assay["wildtype"]),
                   cohort_member=True, source_id=ms, source_pointer=pointer("assays", i),
                   measured=assay["measured"][j], profile_score=assay["profile_scores"][j],
                   rank_target=ranks[j], supporting_common_member=name in common,
                   headline_common_member=name in headline, mutant_digest=assay["mutant_digest"], variant_index=j)
    require(common <= assay_map.keys() and headline <= assay_map.keys(), "support absent from cohort")
    for data, source, sha, name, key, extra in [
        (abundance, ads, ah, "abundance", "domains", ["pfam", "quality_rank", "target", "uncertainty"]),
        (stability, ss, sh, "stability", "backgrounds", ["ddg", "ddg_trypsin", "ddg_chymotrypsin",
          "state", "rows", "distinct_dna", "wildtype_combined_kcal_mol"]),
    ]:
        table = ex.table(f"{name}-samples", base + ["variant_index"] + extra)
        units = set()
        for i, unit in enumerate(data[key]):
            require(unit["name"] not in units, "duplicate cohort unit")
            units.add(unit["name"])
            ex.record(source, pointer(key, i), "cohort_unit_metadata",
                      {k: v for k, v in unit.items() if k not in ("wildtype", "sequences", "variants")})
            mutations = set()
            for j, variant in enumerate(unit["variants"]):
                pos, aa = variant["position"], variant["mutant"]
                require(type(pos) is int and 1 <= pos <= len(unit["wildtype"]), "invalid variant position")
                require(isinstance(aa, str) and len(aa) == 1, "invalid mutant residue")
                mutant = f"{unit['wildtype'][pos - 1]}{pos}{aa}"
                require(mutant not in mutations, "duplicate cohort variant")
                mutations.add(mutant)
                values = {k: (variant[k] if k in variant else unit.get(k)) for k in extra}
                table.row(sample_id=sample_id(sha, unit["name"], mutant), cohort_sha256=sha,
                          unit_id=unit["name"], protein_id=unit.get("accession", unit["name"]),
                          family_id=unit.get("pfam", unit.get("cluster")), group_id=unit["group"],
                          mutation=mutant, positions_1based=pos, cohort_member=True,
                          source_id=source, source_pointer=pointer(key, i, "variants", j),
                          variant_index=j, **values)
    require(len(assay_map) == 217 and mt.count == 27711, "unexpected mutation cohort size")
    require(len(abundance["domains"]) == 428 and ex.tables["abundance-samples"].count == 109568,
            "unexpected abundance cohort size")
    require(len(stability["backgrounds"]) == 101 and ex.tables["stability-samples"].count == 25856,
            "unexpected stability cohort size")
    return assay_map, mh, abundance, ah, stability, sh


def export_models(ex):
    data, source, _ = ex.read(LINEAGES)
    table = ex.table("models", ["model_id", "release_family", "protein_pretraining",
                               "source_id", "source_pointer"])
    models = set()
    for family, record in data["family_records"].items():
        for model in record["checkpoints"]:
            require(model not in models, "duplicate historical model grouping")
            models.add(model)
            table.row(model_id=model, release_family=family,
                      protein_pretraining=record["protein_pretraining"], source_id=source,
                      source_pointer=pointer("family_records", family))
    require(len(models) == 33, "expected 33 historically grouped checkpoints")
    ex.record(source, "/training_ancestry_limit", "grouping_qualification", data["training_ancestry_limit"])
    return models


def is_supporting_readout(data):
    return (data.get("status") == "complete" and data.get("support", {}).get("definition") == "common"
            and (data.get("n_assays"), data.get("n_families"), data.get("n_variants")) == (201, 163, 25728)
            and bool(data.get("predictions")))


def export_mutation(ex, paths, assays, cohort_sha, models, common):
    fields = ["analysis_id", "sample_id", "model_id", "seed", "assay", "group_id",
              "fitted_held_out", "raw_comparator_held_out", "source_id", "source_pointer", "variant_index"]
    table = ex.table("mutation-supporting-predictions", fields + list(DESIGNS)
                     + [f"fold_{d}" for d in FITTED + PERMUTED])
    cells = set()
    groups = {assays[a]["cluster"] for a in common}
    require(len(common) == 201 and len(groups) == 163, "unexpected common support")
    for path in paths:
        data, source, _ = ex.read(path)
        if not is_supporting_readout(data):
            ex.record(source, "", "outside_shared_panel_scope", {
                "status": data.get("status"), "support": data.get("support"),
                "n_assays": data.get("n_assays"), "n_families": data.get("n_families"),
                "n_variants": data.get("n_variants"), "arm": data.get("arm"),
                "reason": "not a complete common 201-assay/163-family/25728-variant readout"})
            continue
        require(data["schema_version"] == "d1_readout_v1", "unexpected supporting schema")
        model, seed = data["arm"], data["fold_seed"]
        require((model, seed) not in cells, "duplicate supporting mutation model/seed cell")
        cells.add((model, seed))
        require(data["status"] == "complete" and data["n_variants"] == 25728, "incomplete mutation source")
        require(data["n_assays"] == 201 and data["n_families"] == 163, "unexpected mutation counts")
        require(cohort_sha in data["source_sha256"].values(), "mutation cohort hash mismatch")
        designs = FITTED + RAW + (PERMUTED if seed == SEEDS[0] else ())
        require(set(data["folds"]) == set(designs) - set(RAW), "unexpected fitted designs")
        mapping = {design: ex.saved_folds(MUTATION_ANALYSIS, model, seed, design, source,
                   pointer("folds", design), records, groups,
                   held="held_families", training="training_families")
                   for design, records in data["folds"].items()}
        ex.metadata(source, data, ("predictions", "folds", "assays", "summaries"))
        seen = set()
        for i, prediction in enumerate(data["predictions"]):
            name = prediction["assay"]
            require(name in common and name not in seen, "unknown or duplicate prediction assay")
            seen.add(name)
            assay = assays[name]
            aligned_predictions(prediction, assay, designs)
            group = assay["cluster"]
            folds = {f"fold_{d}": mapping[d][group] for d in mapping}
            for j, mutant in enumerate(assay["mutants"]):
                table.row(analysis_id=MUTATION_ANALYSIS, sample_id=sample_id(cohort_sha, name, mutant),
                          model_id=model, seed=seed, assay=name, group_id=group,
                          fitted_held_out=True, raw_comparator_held_out=None,
                          source_id=source, source_pointer=pointer("predictions", i), variant_index=j,
                          **{d: prediction[d][j] for d in designs}, **folds)
        require(seen == common, "incomplete supporting common support")
        for design in DESIGNS:
            ex.availability_row(MUTATION_ANALYSIS, cohort_sha, model, seed, design,
                "retained" if design in designs else "not_applicable", source, "/predictions",
                folds="not_applicable" if design in RAW or design not in designs else "retained",
                samples=25728 if design in designs else 0,
                role="raw_rank_comparator" if design in RAW else "negative_control" if design in PERMUTED else "fitted",
                reason="negative controls were run only at the first seed" if design not in designs else "")
        print(dumps({"stage": "mutation", "cell": f"{model}/{seed}", "rows": table.count}), flush=True)
    require(cells == {(m, s) for m in models for s in SEEDS}, "incomplete 33 x 3 mutation panel")
    require(table.count == 99 * 25728, "mutation prediction row count")
    # Manifests are receipts, not checkpoint/NPZ reads. Keep all recorded skips and version references.
    for path in sorted((ex.root / READOUT).glob("**/manifest_*.json")):
        data, source, _ = ex.read(path)
        if data["identity"]["arm"] in models:
            ex.metadata(source, data, ("assays",))
            ex.record(source, "/assays", "extraction_assay_receipts", data["assays"])


def export_local_gap(ex, data, source, cohort_sha, models):
    require(data["cohort_sha256"] == cohort_sha, "headline mutation cohort hash mismatch")
    require(set(data["cells"]) == {f"{m}/{s}" for m in models for s in SEEDS}, "incomplete headline panel")
    ex.metadata(source, data, ("summaries",))
    require(len(data["report_sha256"]) == 99, "incomplete headline artifact references")
    for model in sorted(models):
        for seed in SEEDS:
            refs = [(p, sha) for p, sha in data["report_sha256"].items()
                    if Path(p).name == f"local_context_measure_{model}_fold{seed}.json"]
            require(len(refs) == 1, "headline source reference mismatch")
            original, sha = refs[0]
            # This frozen source-only exporter intentionally does not mix later local reproductions.
            candidates = list((ex.root / "results/R5/local_context_20260923").glob(f"**/{Path(original).name}"))
            require(not candidates, "headline original artifact is now local; update extraction explicitly")
            ex.record(source, pointer("report_sha256", original), "absent_local_referenced_artifact",
                      {"original_path": original, "recorded_sha256": sha},
                      analysis_id=LOCAL_ANALYSIS, model_id=model, seed=seed)
            for design in sorted(data["feature_dimensions"][model]):
                ex.availability_row(LOCAL_ANALYSIS, cohort_sha, model, seed, design,
                    "absent_local_referenced_artifact", source, pointer("report_sha256", original),
                    samples=data["support"]["n_variants"], folds="absent_local_referenced_artifact",
                    reason="admission retained; per-cell JSON and prediction NPZ not retained locally")


def export_abundance(ex, cohort, cohort_sha, models):
    groups = {unit["group"] for unit in cohort["domains"]}
    require(len(groups) == 96, "unexpected abundance groups")
    seen = set()
    for path in sorted((ex.root / ABUNDANCE / "fits").glob("fit_*.json")):
        data, source, _ = ex.read(path)
        model = data["arm"]
        require(model not in seen, "duplicate abundance model")
        seen.add(model)
        require(data["cohort_sha256"] == cohort_sha and data["rows"] == 109568, "abundance cohort mismatch")
        require(set(data["per_seed"]) == set(map(str, SEEDS)), "abundance seeds mismatch")
        ex.metadata(source, data, ("per_seed",))
        for seed, record in data["per_seed"].items():
            path = pointer("per_seed", seed)
            original = record["held_groups_per_fold"]
            folds = [dict(fold=i, held_groups=groups) for i, groups in enumerate(original)]
            # Fold numbers here are original list indices, never redrawn assignments.
            outer_mapping(folds, groups)
            ex.record(source, path + "/held_groups_per_fold", "original_fold_records", original,
                      analysis_id=ABUNDANCE_ANALYSIS, model_id=model, seed=int(seed))
            for i, held in enumerate(original):
                for j, group in enumerate(held):
                    ex.folds.row(analysis_id=ABUNDANCE_ANALYSIS, model_id=model, seed=seed,
                        design="all_saved_designs", outer_fold=i, group_id=group, role="held_out",
                        held_out=True, source_id=source, source_pointer=path + pointer("held_groups_per_fold", i, j))
            ex.record(source, path, "fit_design_and_tuning_metadata",
                      {k: record[k] for k in ("alpha", "dimensions", "nuisance")},
                      analysis_id=ABUNDANCE_ANALYSIS, model_id=model, seed=int(seed))
            for design in sorted(record["dimensions"]):
                primary, secondary = data["matched_baseline"]["primary"], data["matched_baseline"]["secondary"]
                role = "primary" if design in (primary, primary + "_M", primary + "_R") else (
                    "secondary" if design in (secondary, secondary + "_M", secondary + "_R") else "other_recorded_design")
                ex.availability_row(ABUNDANCE_ANALYSIS, cohort_sha, model, int(seed), design,
                    "not_serialized", source, path, folds="retained_outer_only", samples=109568, role=role,
                    reason="fit serializer retained metrics and outer held groups, not prediction vectors or inner memberships")
    require(seen == models, "incomplete abundance 33-model panel")
    for file in ("support_declaration.json", "endpoint_qualification.json", "panel/panel.json"):
        data, source, _ = ex.read(f"{ABUNDANCE}/{file}")
        ex.metadata(source, data, ("panel", "arms", "baseline_mse_values"))


def export_stability(ex, cohort, cohort_sha, models):
    groups = {unit["group"] for unit in cohort["backgrounds"]}
    seen = set()
    table = ex.table("stability-group-metrics", ["analysis_id", "model_id", "seed", "group_id",
        "baseline", "augmented", "baseline_mse", "augmented_mse", "source_id", "source_pointer", "group_index"])
    for path in sorted((ex.root / REPLAY).glob("*.json")):
        if path.name in ("panel.json", "declaration.json"):
            data, source, _ = ex.read(path)
            ex.metadata(source, data)
            continue
        data, source, _ = ex.read(path)
        model = data["arm"]
        require(model not in seen, "duplicate stability model")
        seen.add(model)
        require(data["cohort_sha256"] == cohort_sha and data["baseline"] == "S", "stability replay mismatch")
        require(set(data["seeds"]) == set(map(str, SEEDS)), "stability seeds mismatch")
        ex.metadata(source, data, ("seeds",))
        for seed, record in data["seeds"].items():
            require(len(record["groups"]) == len(groups) and set(record["groups"]) == groups, "stability metric groups mismatch")
            for design in ("baseline_mse", "augmented_mse"):
                finite_vector(record[design], len(groups), design)
            ex.saved_folds(STABILITY_ANALYSIS, model, int(seed), "S|S_M", source,
                           pointer("seeds", seed, "folds"), record["folds"], groups)
            for i, group in enumerate(record["groups"]):
                table.row(analysis_id=STABILITY_ANALYSIS, model_id=model, seed=seed, group_id=group,
                          baseline="S", augmented="S_M", baseline_mse=record["baseline_mse"][i],
                          augmented_mse=record["augmented_mse"][i], source_id=source,
                          source_pointer=pointer("seeds", seed), group_index=i)
            for design in ("S", "S_M"):
                ex.availability_row(STABILITY_ANALYSIS, cohort_sha, model, int(seed), design,
                    "not_serialized", source, pointer("seeds", seed), samples=25856,
                    folds="retained_outer_only", role="primary",
                    reason="group MSE arrays are retained, not sample predictions")
    require(seen == models, "incomplete stability 33-model replay panel")
    data, source, _ = ex.read(f"{STABILITY}/panel.json")
    ex.metadata(source, data, ("panel",))
    # Retain every primary/secondary and remote-purge record, never relabel it as primary replay.
    for contrast, block in data["panel"].items():
        require(set(block["arms"]) == models, f"stability historical contrast coverage: {contrast}")
        ex.record(source, pointer("panel", contrast), "historical_panel_contrast", block)
        for model, cell in block["arms"].items():
            for seed in SEEDS:
                ex.availability_row("stability_R4_historical_" + contrast, cohort_sha, model, seed,
                    contrast, "not_serialized", source, pointer("panel", contrast, "arms", model),
                    samples=25856, reason="historical panel summary; original per-model fits mostly absent locally")
                for threshold in cell.get("remote", {}):
                    ex.availability_row("stability_R4_remote_" + threshold, cohort_sha, model, seed,
                        contrast, "not_serialized", source,
                        pointer("panel", contrast, "arms", model, "remote", threshold), samples=25856,
                        reason="distinct remote-purge sensitivity; retained panel has counts, not sample predictions or exact memberships")
    for path in sorted((ex.root / STABILITY / "fits").glob("fit_*.json")):
        data, source, _ = ex.read(path)
        ex.metadata(source, data, ("primary", "remote_stratification"))
        for key in ("primary", "remote_stratification"):
            # This single surviving fit contains no sample vectors; keep its actual tuning/remote receipt.
            ex.record(source, pointer(key), "surviving_historical_fit_receipt", data[key])
    data, source, _ = ex.read(f"{STABILITY}/exclusions/exclusion_profile.json")
    ex.metadata(source, data)


README = """# Retained-source prediction details

This CPU-only export preserves existing records. It performs no inference, fitting, bootstrap, resampling, split generation, checkpoint access or downloads. It does not make missing predictions into negative outcomes. `coverage.json` is written only after all completeness checks pass.

## Tables and joins

- `mutation-samples.csv.gz`, `abundance-samples.csv.gz`, `stability-samples.csv.gz`: all measured rows in each frozen cohort, including mutation assays outside common analysis support. `unit_id` is the original assay/domain/background name. `protein_id`, `family_id` and `group_id` retain their source identifiers; the latter is the original split grouping, not a new homology claim. Cohort membership and analysis-support membership are independent of measured outcomes. `mutation` is the exact saved colon-separated substitution string for mutation assays; for single-site endpoints it combines the saved WT residue, saved 1-based position and saved mutant AA. Domain positions are domain-local, not full-protein coordinates. No raw sequences are exported.
- `mutation-supporting-predictions.csv.gz`: one common-support sample × model × seed row, containing ALL saved native design columns, including raw comparators and first-seed negative controls. Join `sample_id` to mutation samples and `model_id` to models. The 33 × 3 cells all contain the same 25,728 samples. `fold_<design>` is the actual outer fold for that design; `fitted_held_out=true` applies to fitted designs only. Raw comparators have no fitted split; their held-out indicator is blank/not applicable. Blank permuted columns at later seeds mean not run, not zero.
- `fold-membership.csv.gz`: original group assignments keyed by analysis, model, seed, design and outer fold. `all_saved_designs` applies to every saved abundance design; `S|S_M` applies only to the stability primary likelihood replay pair. Join cohort `group_id` to `role=held_out` for a per-sample held-out fold even when predictions are unavailable. Training and inner-validation memberships appear only if serialized. Abundance fold numbers and inner fold numbers are original list indices. Original fold records are also retained in provenance, including tuned alphas and purged groups. No fold is reconstructed.
- `stability-group-metrics.csv.gz`: original paired group MSE values, in squared kcal/mol. These are NOT sample predictions and cannot be joined as though they were.
- `models.csv.gz`: original release-family and historical binary `protein_pretraining` labels from the retained lineage receipt. These are historical labels, not reclassifications or a new claim about joint training. Checkpoint identities, metadata hashes, retained code/version references and extraction skips are in provenance.
- `availability.csv.gz`: every analysis/model/seed/design cell; `retained`, `absent_local_referenced_artifact`, `not_serialized`, or `not_applicable`. Availability is not a scientific outcome. Primary/secondary abundance roles remain distinct. Historical stability contrasts and remote-purge sensitivities are separately named.

## Scales

Mutation `measured` is the original assay score, not a common physical unit; `profile_score` preserves the cohort score. `rank_target` is the deterministic within-assay average-tie rank standardized using population SD (zero for a constant assay), matching the original producer. All supporting mutation prediction columns, including `raw_M` and `raw_P`, are on this rank scale, NOT nats or assay phenotype units. Native designs: P=profile; M=likelihood; P_M=both; sequence=sequence controls; profile_sequence=profile plus sequence; B=profile PLUS likelihood PLUS sequence; B_R=B plus representation; R=representation alone. `permuted_B`/`permuted_B_R` are the saved one-realization negative controls, not new permutations.

Abundance `target` is source `normalized_fitness`, dimensionless and WT-centered at 0; higher means greater cellular abundance. `uncertainty` is original `normalized_fitness_sigma`, not a confidence interval, reweighting or threshold. Stability `ddg` is mutant minus WT on the combined unfolding free-energy scale in kcal/mol; positive is stabilizing. Original trypsin/chymotrypsin deltas, state, row count and DNA count remain separate, as does the WT combined measurement. No scale is pooled across cohorts.

## Provenance, missingness and limits

`sample_id = sha256(compact JSON [cohort SHA256, original unit name, mutation])`; IDs are bound to the exact cohort and independent of model or file ordering. All row `source_id` values join `sources.jsonl.gz`, which supplies relative path, SHA256 and byte length. RFC 6901 `source_pointer` values identify original records. Mutation sample `variant_index` locates the identical entry in `mutants`; other aligned vectors share that index. Prediction `variant_index` and group metric `group_index` identify aligned vector entries. `provenance.jsonl.gz` carries exact original metadata, exclusion/failure/skip records, artifact references, hashes and fold records, rather than full-fit JSON duplicates. Missing/null CSV values are blank; booleans are explicit `true`/`false`. JSON nulls are retained. NaN and infinity are rejected.

The headline local-control mutation analysis is DISTINCT from supporting R2 readouts. Its 99 per-cell JSON paths/hashes survive in the admission receipt, but those files and associated NPZ vectors are absent locally. No R2 value substitutes for C_P_wall or its augmentations. Abundance serializers did not retain sample predictions or inner-fold memberships. Stability replay retained group losses and exact outer fold receipts but not sample predictions or inner memberships. Historical stability remote-purge counts are not exact group assignments. Cohort exclusions may contain identifiers/reasons without measured rows; those records remain in provenance, not invented sample rows. Availability refers only to the source inventory at export, not to successful inference.

Native-interface and 40-assay pilot panels lie outside this shared-panel dataset. No new model selection or regrouping is performed. Later local reproductions must use a NEW analysis_id and separately provenanced prediction table referencing these sample IDs; never overwrite original H200 source hashes or claim reproduced values were retained originals.

Gzip uses an empty filename and mtime=0. Tables and provenance are deterministic for identical source/exporter bytes. `resource-log.json` is an operational CPU/memory/disk log and is intentionally run-dependent. `coverage.json` records table counts, output hashes and exporter hash. Run with `PYTHONDONTWRITEBYTECODE=1 /home/lzp/miniconda3/envs/ct/bin/python scripts/capability/reporting/export_prediction_details.py --root ORIGINAL_REPOSITORY --out NEW_OR_EMPTY_DIRECTORY`.
"""


def export(root, out):
    require(root.is_dir(), "source root does not exist")
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "--out must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    resources = {"compute": "CPU only; no numerical/model library imported", "disk_free_bytes": shutil.disk_usage(out).free,
                 "memory": Path("/proc/meminfo").read_text(), "peak_rss_kib_before": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    require(resources["disk_free_bytes"] > 2_000_000_000, "need at least 2 GB free for export")
    (out / "resource-log.json").write_text(dumps(resources) + "\n")
    print(dumps({"stage": "resource_preflight", "disk_free_bytes": resources["disk_free_bytes"], "device": "CPU"}), flush=True)
    with ExitStack() as stack:
        ex = Export(root, out, stack)
        paths = sorted((root / READOUT).glob("**/readout_*common*.json"))
        require(bool(paths), "no supporting mutation source files")
        local, source, _ = ex.read(LOCAL)
        common = set(local["support"]["assay_ids"])
        models = export_models(ex)
        assays, mh, abundance, ah, stability, sh = export_cohorts(ex, common, set(local["support"]["assay_ids"]))
        export_local_gap(ex, local, source, mh, models)
        export_mutation(ex, paths, assays, mh, models, common)
        export_abundance(ex, abundance, ah, models)
        export_stability(ex, stability, sh, models)
        with gzip_text(out / "sources.jsonl.gz") as stream:
            for source in sorted(ex.sources):
                stream.write(dumps(ex.sources[source]) + "\n")
    (out / "README.md").write_text(README)
    resources["peak_rss_kib_after"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resources["disk_free_bytes_after"] = shutil.disk_usage(out).free
    (out / "resource-log.json").write_text(dumps(resources) + "\n")
    coverage = {"schema": "retained_prediction_details_v1", "complete": True,
                "exporter_sha256": digest(Path(__file__)), "models": len(models), "seeds": list(SEEDS),
                "source_count": len(ex.sources), "provenance_records": ex.record_count,
                "row_counts": {k: v.count for k, v in ex.tables.items()},
                "availability_statuses": ["retained", "absent_local_referenced_artifact", "not_serialized", "not_applicable"],
                "excluded_scopes": ["native-interface panels", "40-assay pilot panels", "later local reproductions"],
                "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                            for p in sorted(out.iterdir()) if p.name != "resource-log.json"}}
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2, allow_nan=False) + "\n")
    print(dumps({"stage": "complete", "rows": coverage["row_counts"], "output_bytes":
                 sum(p.stat().st_size for p in out.iterdir())}), flush=True)
    return coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="original repository (read only)")
    parser.add_argument("--out", type=Path, required=True, help="new or empty flat dataset directory")
    args = parser.parse_args(argv)
    export(args.root.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
