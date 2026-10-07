"""Frozen-panel descriptive cross-phenotype associations; no checkpoint-iid inference."""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (20260923, 20260924, 20260925)
METRICS = ("proteingym_rank", "abundance_rank", "abundance_mse", "stability_mse")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_families(path: Path) -> dict:
    """Read the authoritative literal without importing GPU/fitting dependencies."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FAMILIES" for t in node.targets):
            families = ast.literal_eval(node.value)
            arms = [a for _, members in families.values() for a in members]
            if len(families) != 16 or len(arms) != 33 or len(set(arms)) != 33:
                raise ValueError("invalid frozen release-family mapping")
            return families
    raise ValueError("FAMILIES declaration not found")


def validate_cells(frame: pd.DataFrame, roster: list[str], split: bool) -> None:
    keys = ["arm", "split_seed"] if split else ["arm"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate endpoint cells")
    expected = set(itertools.product(roster, SEEDS)) if split else {(a,) for a in roster}
    if set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("missing cells or identifier/split mismatch")
    if not np.isfinite(frame[["estimate", "ci_low", "ci_high"]].to_numpy(dtype=float)).all():
        raise ValueError("missing/nonfinite endpoint values")
    if (frame.ci_low > frame.ci_high).any():
        raise ValueError("reversed source intervals")


def correlations(frame: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for x, y in itertools.combinations(METRICS, 2):
        if not np.isfinite(frame[[x, y]].to_numpy(dtype=float)).all():
            raise ValueError("incomplete paired effects; no silent pairwise deletion")
        constant = frame[x].nunique() < 2 or frame[y].nunique() < 2
        rows.append({"summary": label, "x": x, "y": y, "n": len(frame),
                     "spearman": None if constant else float(np.corrcoef(pd.Series(frame[x].to_numpy()).rank(method="average"), pd.Series(frame[y].to_numpy()).rank(method="average"))[0, 1]),
                     "pearson": None if constant else float(np.corrcoef(frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float))[0, 1]),
                     "status": "nonestimable_constant" if constant else "descriptive_only"})
    return rows


def release_means(joined: pd.DataFrame, families: dict) -> pd.DataFrame:
    records = []
    for family, (category, arms) in families.items():
        subset = joined.set_index("arm").loc[list(arms)]
        records.append({"release_family": family, "release_category": category,
                        "n_checkpoints": len(arms), "checkpoints": ";".join(arms),
                        **{m: float(subset[m].mean()) for m in METRICS}})
    return pd.DataFrame(records)


def within_release_categories(releases: pd.DataFrame, families: dict, labels: dict) -> list[dict]:
    """Stratify existing release means without redefining the frozen categories."""
    if releases.release_family.duplicated().any() or set(releases.release_family) != set(families):
        raise ValueError("incomplete or duplicate release-category support")
    declared = {family: category for family, (category, _) in families.items()}
    if not (releases.release_category == releases.release_family.map(declared)).all():
        raise ValueError("release category differs from frozen declaration")
    if set(releases.release_category) != set(labels):
        raise ValueError("category-label coverage mismatch")
    records = []
    for category, label in labels.items():
        subset = releases.loc[releases.release_category == category]
        members = subset.release_family.tolist()
        records.extend({**record, "release_category": category, "category_label": label,
                        "release_families": members}
                       for record in correlations(subset, f"within_release_category:{category}"))
    return records


def frozen_category_labels(path: Path) -> dict:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GROUPING" for t in node.targets):
            return ast.literal_eval(node.value)["labels"]
    raise ValueError("GROUPING declaration not found")


def analyse(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hashes, cache = {}, {}

    def load(relative: str, expected: str | None = None) -> dict:
        path = root / relative
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"source digest mismatch: {relative}")
        hashes[relative] = actual
        if relative not in cache:
            cache[relative] = json.loads(path.read_text())
        return cache[relative]

    family_path = "scripts/capability/reporting/audit_followup_support.py"
    families = release_families(root / family_path)
    hashes[family_path] = digest(root / family_path)
    roster = sorted(a for _, arms in families.values() for a in arms)
    specs = [("proteingym_rank", "local-increments-source.csv", "increment_M_C_P_wall", True),
             ("abundance_rank", "external-abundance-source.csv", "primary_likelihood_spearman", True),
             ("abundance_mse", "external-abundance-source.csv", "primary_likelihood", True),
             ("stability_mse", "stability-panel-source.csv", None, False)]
    joined = pd.DataFrame({"arm": roster})
    source_rows = []
    for metric, filename, endpoint, split in specs:
        relative = f"manuscript/data/{filename}"
        hashes[relative] = digest(root / relative)
        frame = pd.read_csv(root / relative)
        if endpoint:
            frame = frame.loc[frame.endpoint == endpoint].copy()
        validate_cells(frame, roster, split)
        for row in frame.to_dict("records"):
            receipt = load(row["source_path"], row["source_sha256"])
            arm = row["arm"]
            if metric == "proteingym_rank":
                if row["source_pointer"] != f'summaries.{arm}/{row["split_seed"]}.{endpoint}':
                    raise ValueError("ProteinGym source pointer mismatch")
                cell = receipt["summaries"][f'{arm}/{row["split_seed"]}'][endpoint]
                point, interval = cell["point"], cell["interval"]
                if receipt["status"] != "admitted":
                    raise ValueError("ProteinGym source not admitted")
            elif split:
                if receipt["arm"] != arm or set(map(int, receipt["per_seed"])) != set(SEEDS):
                    raise ValueError("abundance receipt identifier/split mismatch")
                if row["source_pointer"] != f'per_seed.{row["split_seed"]}.{endpoint}':
                    raise ValueError("abundance source pointer mismatch")
                cell = receipt["per_seed"][str(row["split_seed"])][endpoint]
                point, interval = cell["point"], cell["interval"]
                if cell["baseline"] not in ("S", "S_T") or cell["baseline"] != receipt["matched_baseline"]["primary"]:
                    raise ValueError("abundance baseline mismatch")
                row["matched_baseline"] = cell["baseline"]
            else:
                if len(receipt["arms"]) != len(set(receipt["arms"])) or set(receipt["arms"]) != set(roster):
                    raise ValueError("stability panel identifier mismatch")
                index = receipt["arms"].index(arm)
                point, interval = receipt["primary"]["point"][index], receipt["primary"]["interval"][index]
                if row["source_pointer"] != f"primary.point.{index}" or row["baseline_source_pointer"] != "mean(seeds.*.baseline_mse)":
                    raise ValueError("stability source pointer mismatch")
                replay = load(row["baseline_source_path"], row["baseline_source_sha256"])
                if replay["arm"] != arm or replay["baseline"] != "S" or set(map(int, replay["seeds"])) != set(SEEDS):
                    raise ValueError("stability replay identity/baseline/split mismatch")
                split_points = [float(np.mean(np.array(replay["seeds"][str(s)]["baseline_mse"]) - np.array(replay["seeds"][str(s)]["augmented_mse"]))) for s in SEEDS]
                if not np.isclose(np.mean(split_points), point, atol=1e-12, rtol=0):
                    raise ValueError("stability split mean does not reproduce panel")
            if not np.allclose([row["estimate"], row["ci_low"], row["ci_high"]], [point, *interval], atol=1e-12, rtol=0):
                raise ValueError("source table differs from receipt")
            source_rows.append({"metric": metric, **row})
        records = []
        for arm, rows in frame.groupby("arm", sort=True):
            if split:
                points = rows.sort_values("split_seed").estimate.tolist()
            else:
                replay = load(rows.iloc[0].baseline_source_path)
                points = [float(np.mean(np.array(replay["seeds"][str(s)]["baseline_mse"]) - np.array(replay["seeds"][str(s)]["augmented_mse"]))) for s in SEEDS]
            records.append({"arm": arm, metric: float(rows.estimate.mean()),
                            f"{metric}_missing": False,
                            f"{metric}_positive": bool((rows.ci_low > 0).all()),
                            f"{metric}_point_positive": bool(rows.estimate.mean() > 0),
                            f"{metric}_source_paths": ";".join(sorted(set(rows.source_path))),
                            f"{metric}_source_pointers": ";".join(rows.source_pointer.astype(str)),
                            f"{metric}_split_points": json.dumps(points)})
        joined = joined.merge(pd.DataFrame(records), on="arm", validate="one_to_one")
    mapping = {a: f for f, (_, arms) in families.items() for a in arms}
    joined.insert(1, "release_family", joined.arm.map(mapping))
    releases = release_means(joined, families)
    main = correlations(joined, "checkpoints_descriptive") + correlations(releases, "release_means_primary")
    loo = [r for f in families for r in correlations(releases.loc[releases.release_family != f], f"leave_out:{f}")]
    ranges = []
    for x, y in itertools.combinations(METRICS, 2):
        cells = [r for r in loo if (r["x"], r["y"]) == (x, y)]
        ranges.append({"x": x, "y": y, **{f"{stat}_{edge}": fn(r[stat] for r in cells) for stat in ("spearman", "pearson") for edge, fn in (("min", min), ("max", max))}})
    excluded = releases.loc[~releases.release_family.isin(["Llama2", "ProLLaMA"])]
    positives = {m: joined.loc[joined[f"{m}_positive"], "arm"].tolist() for m in METRICS}
    if positives["proteingym_rank"] != positives["abundance_rank"] or len(positives["proteingym_rank"]) != 14:
        raise ValueError("frozen rank positivity sets changed")
    if set(positives["stability_mse"]) != {"progen3-3b", "prollama"} or set(positives["abundance_mse"]) != {"progen2-xlarge", "proteinglm-7b-clm"}:
        raise ValueError("frozen MSE positivity sets changed")
    gates = {"proteingym_rank": {"baseline": "C_P_wall", "unit": "dimensionless Spearman increment", "criterion": "source interval lower > 0 in all three splits", "gate": "admitted local-context measurement"},
             "abundance_rank": {"baseline": "receipt-specific S or S_T (qualified controls plus admitted tokenisation where carried)", "unit": "dimensionless within-domain Spearman increment", "criterion": "source interval lower > 0 in all three splits; pointwise unadjusted"},
             "abundance_mse": {"baseline": "receipt-specific S or S_T (qualified controls plus admitted tokenisation where carried)", "unit": "squared normalised abundance-PCA fitness reduction", "criterion": "source interval lower > 0 in all three splits; pointwise unadjusted"},
             "stability_mse": {"baseline": "S = ident + geom + chem + G", "unit": "(kcal/mol)^2 MSE reduction", "criterion": "33-model simultaneous biological-group interval lower > 0; splits averaged"}}
    abundance_panel_path = "results/R1/external_confirmation_20260924/panel/panel.json"
    abundance_panel = load(abundance_panel_path)
    if set(abundance_panel["arms"]) != set(roster) or abundance_panel["arms_fitted"] != 33 or not abundance_panel["roster_complete"]:
        raise ValueError("abundance panel incomplete")
    gates["abundance_rank"]["licensed"] = abundance_panel["panel"]["primary_likelihood_spearman"]["licensed_for_a_verdict"]
    gates["abundance_mse"]["licensed"] = abundance_panel["panel"]["primary_likelihood"]["licensed_for_a_verdict"]
    stability_original_path = "results/R4/gate_stability_20260924/panel.json"
    stability_original = load(stability_original_path)
    original_cells = stability_original["panel"]["primary_likelihood"]["arms"]
    if set(original_cells) != set(roster) or stability_original["arms_missing"]:
        raise ValueError("original stability panel incomplete")
    for arm, point in joined[["arm", "stability_mse"]].itertuples(index=False, name=None):
        if not np.isclose(np.mean(original_cells[arm]["per_seed_point"]), point, atol=1e-12, rtol=0):
            raise ValueError("original stability panel mean differs from replay")
    gates["stability_mse"]["original_panel"] = stability_original_path
    gates["stability_mse"]["original_pointwise_multiplicity"] = stability_original["multiplicity"]
    gates["abundance_rank"]["panel_verdict_rule"] = abundance_panel["verdict_rule"]
    gates["abundance_mse"]["panel_verdict_rule"] = abundance_panel["verdict_rule"]
    return joined, releases, {"schema": "phenotype_consistency_descriptive_v1", "metrics": gates,
        "families": families, "source_sha256": hashes, "source_rows": source_rows,
        "pairwise": main, "leave_one_release_out": loo, "leave_one_release_out_ranges": ranges,
        "shared_parent_joint_exclusion": correlations(excluded, "exclude_Llama2_and_ProLLaMA_release_means"),
        "positives": positives, "missing_cells": 0,
        "limitations": ["Three split point estimates are averaged within each endpoint, not independent replicates.",
            "Equal checkpoints within each of 16 declared releases; correlations of release means are primary descriptive summaries, not independent training-history inference.",
            "Selected checkpoints and releases are not iid; no p-values or confidence intervals computed.",
            "Different controls, endpoint units and positivity criteria; rank and MSE are not pooled into a score. Pearson is descriptive linear association only, not commensurate effect magnitude.",
            "ProteinGym includes Tsuboyama stability assays; protein/family overlap is not excluded or independence established.",
            "Historical stability ranks under S/S2 not included; no ranking requalification or new fits."],
        "biological_bootstrap": {"status": "not_computed", "reason": "No complete jointly aligned cross-endpoint biological-group increment vectors with audited protein/family overlap were supplied. Original abundance fit receipts retain scalar summaries rather than complete group increment vectors. Scalar intervals cannot reconstruct covariance; available stability vectors alone are insufficient."}}
