"""CPU nested-CV evaluation of R3 double-mutant epistasis cycles for general PLMs.

Matches the frozen interaction lane protocol in
``src/capability/interactions/pairwise_epistasis.py`` and
``scripts/capability/extensions/explicit_statistics_lanes/explicit_statistics_lane_adapters.py``:
- Target: measured-singles-residualized double-mutant epistasis cycle
  ``epsilon = y_AB - y_A - y_B + y_WT`` (kcal/mol) on the indel-excluded
  complete-case Potts support (61 groups, 7,805 cycles) using the exact cached
  3-seed x 5-outer x 4-inner held-group splits and cross-fitted nonlinear-additive
  nuisance block ``G``.
- Control hierarchy (3 rungs):
  1. ``nuisance``: ``ident, geom, comp, local, G, T``
  2. ``nuisance+profile``: ``ident, geom, comp, local, prof, G, T``
  3. ``nuisance+profile+couplings``: ``ident, geom, comp, local, prof, G, T, POTTS``
- Channels evaluated bidirectionally (forward ``model | controls`` and reverse
  ``controls | model``) in both group-equal cycle MSE (kcal^2/mol^2, site-pair-equal
  within group) and within-background Spearman:
  - ``M_cycle``: scalar exact-PLL double-mutant cycle ``eps_PLL``
  - ``R_cycle``: 1024-d frozen hidden-state cycle ``R(AB) - R(A) - R(B) + R(WT)``
  - ``MR_cycle``: joint ``M_cycle + R_cycle`` over ``nuisance+profile+couplings``
- Also computes zero-shot within-background Spearman vs raw measured ``epsilon``
  (overall and stratified by sequence separation ``1-2``, ``3-9``, ``10+``)
  alongside the Potts coupling contrast on the identical support.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.capability.baselines.cohorts import SAMPLE_SEPARATOR
from src.capability.context.profiles import Profile
from src.capability.extensions import explicit_statistics_lanes as lanes
from src.capability.general_plm.eval_utils import safe_spearman, sha256_file
from src.capability.interactions import pairwise_epistasis as mod
from src.capability.readouts.readout_analysis import family_folds, ridge_predict, row_weights

SCHEMA = "general_plm_r3_cycle_eval_v1"

INTERACTION_RUNGS: dict[str, tuple[str, ...]] = {
    "nuisance": ("ident", "geom", "comp", "local", "G", "T"),
    "nuisance+profile": ("ident", "geom", "comp", "local", "prof", "G", "T"),
    "nuisance+profile+couplings": ("ident", "geom", "comp", "local", "prof", "G", "T", "POTTS"),
}
STRATA = ("1-2", "3-9", "10+")


def _load_profiles(path: Path, plan: dict[str, Any]) -> dict[str, Profile | None]:
    wildtypes = {row["name"]: row["wildtype"] for row in plan["backgrounds"]}
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        profiles: dict[str, Profile | None] = {}
        for record in meta["backgrounds"]:
            name = record["name"]
            if record["status"] != "present":
                profiles[name] = None
                continue
            scalars = data[f"{name}|scalars"]
            profiles[name] = Profile(
                query_id=name,
                wildtype=wildtypes[name],
                n_hits=int(scalars[4]),
                n_sequences=int(scalars[3]),
                saturated=False,
                frequencies=data[f"{name}|frequencies"].astype(np.float64),
                column_weight=data[f"{name}|column_weight"].astype(np.float64),
                neff=float(scalars[0]),
                max_identity_over_query=float(scalars[2]),
            )
    return profiles


def _sha256_array(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    h = hashlib.sha256(str((arr.dtype.str, arr.shape)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _verify_shared_inputs(panel: dict[str, Any], cache_dir: Path) -> None:
    shared_path = cache_dir / "shared-inputs.json"
    if not shared_path.is_file():
        return
    expected = json.loads(shared_path.read_text(encoding="utf-8"))
    observed = {
        **{f"states_{k}": _sha256_array(v) for k, v in panel["states"].items()},
        **{
            k: _sha256_array(panel[k])
            for k in ("group", "site_pair", "background", "cycle_index", "epsilon", "cycle_states")
        },
        **{f"C_{k}": _sha256_array(panel["blocks"][k]) for k in mod.C_BLOCKS},
    }
    mismatches = {
        k: (observed[k], expected[k])
        for k in observed
        if k in expected and observed[k] != expected[k]
    }
    if mismatches:
        first = next(iter(mismatches))
        raise ValueError(
            f"R3 pairwise panel does not match cached fold shared-inputs.json on {first}: "
            f"got {mismatches[first][0]}, expected {mismatches[first][1]}"
        )


def _aa_tokenisation_features(
    sequences: list[str], states: tuple[int, int, int, int], length: int
) -> np.ndarray:
    """Exact 1-token-per-residue tokenisation descriptor block matching AA-tokenized arms."""
    pooled_counts = np.asarray([len(s) for s in sequences], dtype=np.float64)
    token_ids = [[ord(ch) for ch in s] for s in sequences]
    return mod.tokenisation_features(pooled_counts, token_ids, states, length)


@dataclass
class PairwiseCycleContext:
    """Frozen R3 pairwise cohort, controls, Potts couplings, and cached fold partitions."""

    root: Path
    plan_path: Path
    plan: dict[str, Any]
    panel: dict[str, Any]
    kept_global_indices: np.ndarray  # index into the full 8,192-cycle extraction_plan order
    raw_all_epsilon: np.ndarray  # length 8,192
    raw_all_groups: np.ndarray  # length 8,192
    raw_all_separations: np.ndarray  # length 8,192
    raw_all_backgrounds: np.ndarray  # length 8,192
    potts_cycle_raw: np.ndarray  # length len(kept_global_indices)
    weights: np.ndarray
    group_order: list[Any]
    cache_dir: Path
    fold_entries: dict[tuple[int, int], dict[str, Any]]
    support_sha256: str
    _control_fit_cache: dict[str, Any] | None = field(default=None, repr=False)


def load_pairwise_cycle_context(root: Path) -> PairwiseCycleContext:
    """Assemble and hash-verify the 61-group complete-case R3 pairwise evaluation context."""
    plan_path = root / "archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json"
    cohort_path = root / "archive/logs/R3/pairwise_cohort_20260924/cohort.json"
    profiles_path = root / "archive/logs/R3/pairwise_epistasis_20260924/profile_features.npz"
    excl_path = root / "results/R3/pairwise_indel_exclusion_20260928/pairwise_indel_exclusion.json"
    potts_path = root / "results/extensions/explicit_statistics_pairwise_20261109/potts/scores.json"

    for p in (plan_path, cohort_path, profiles_path, excl_path, potts_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required R3 evaluation input missing: {p}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    profiles = _load_profiles(profiles_path, plan)
    excl_decl = json.loads(excl_path.read_text(encoding="utf-8"))
    potts_doc = json.loads(potts_path.read_text(encoding="utf-8"))

    potts_vals = {
        s["sample_id"]: float(s["potts_delta"])
        for s in potts_doc["samples"]
        if s.get("status") == "fitted" and s.get("potts_delta") is not None
    }
    refused_bgs = set(potts_doc.get("refused", {}).keys())
    kept_groups = {b["group"] for b in plan["backgrounds"] if b["name"] not in refused_bgs}

    measurements = {row["name"]: row["measurements"] for row in cohort["backgrounds"]}
    eps_by_bg = {
        row["name"]: [float(c["epsilon"]) for c in row["cycles"]]
        for row in cohort["backgrounds"]
    }

    # Global 8,192-cycle raw metadata (for zero-shot correlation on full plan order).
    raw_all_eps: list[float] = []
    raw_all_grp: list[Any] = []
    raw_all_sep: list[str] = []
    raw_all_bg: list[str] = []

    # Build pre-indel-exclusion panel on kept_groups, tracking global cycle index.
    row_group: list[Any] = []
    row_pair: list[str] = []
    row_bg: list[str] = []
    row_sep: list[str] = []
    row_cycle: list[int] = []
    row_eps: list[float] = []
    row_global_idx: list[int] = []
    control_rows: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    cycle_states_list: list[list[int]] = []
    state_feat_rows: list[np.ndarray] = []
    state_y: list[float] = []
    state_group: list[Any] = []
    state_bg: list[str] = []

    global_cursor = 0
    for bg in plan["backgrounds"]:
        name = bg["name"]
        n_cyc = len(bg["cycles"])
        bg_eps = eps_by_bg[name]
        for c_idx, cyc in enumerate(bg["cycles"]):
            raw_all_eps.append(bg_eps[c_idx])
            raw_all_grp.append(bg["group"])
            raw_all_sep.append(str(cyc["separation"]))
            raw_all_bg.append(name)

        if bg["group"] not in kept_groups:
            global_cursor += n_cyc
            continue

        wildtype = bg["wildtype"]
        sequences = bg["sequences"]
        profile = profiles[name]
        base = len(state_y)
        for seq in sequences:
            state_feat_rows.append(mod.state_features(wildtype, seq, profile))
            state_y.append(float(measurements[name][seq]["value"]))
            state_group.append(bg["group"])
            state_bg.append(name)

        for c_idx, cyc in enumerate(bg["cycles"]):
            states = tuple(int(s) for s in cyc["states"])
            positions = tuple(int(p) for p in cyc["positions"])
            quartet = tuple(sequences[s] for s in states)
            control_rows.append(
                mod.cycle_control_features(wildtype, quartet, positions, cyc["separation"], profile)
            )
            token_rows.append(_aa_tokenisation_features(sequences, states, len(wildtype)))
            cycle_states_list.append([base + s for s in states])
            row_group.append(bg["group"])
            row_pair.append(f"{name}:{positions[0]}-{positions[1]}")
            row_bg.append(name)
            row_sep.append(str(cyc["separation"]))
            row_cycle.append(c_idx)
            row_eps.append(bg_eps[c_idx])
            row_global_idx.append(global_cursor + c_idx)
        global_cursor += n_cyc

    ctrl = np.asarray(control_rows, dtype=np.float64)
    raw_panel: dict[str, Any] = {
        "group": np.asarray(row_group),
        "site_pair": np.asarray(row_pair),
        "background": np.asarray(row_bg),
        "separation": np.asarray(row_sep),
        "cycle_index": np.asarray(row_cycle, dtype=np.int64),
        "epsilon": np.asarray(row_eps, dtype=np.float64),
        "cycle_states": np.asarray(cycle_states_list, dtype=np.int64),
        "independent_site_max_absolute": None,
        "blocks": {
            "ident": ctrl[:, :400],
            "geom": ctrl[:, 400:410],
            "comp": ctrl[:, 410:450],
            "local": ctrl[:, 450:850],
            "prof": ctrl[:, 850:],
            "T": np.asarray(token_rows, dtype=np.float64),
            "_global_idx": np.asarray(row_global_idx, dtype=np.int64)[:, None],
        },
        "states": {
            "features": np.asarray(state_feat_rows, dtype=np.float64),
            "y": np.asarray(state_y, dtype=np.float64),
            "group": np.asarray(state_group),
            "background": np.asarray(state_bg),
        },
    }
    raw_panel["states"]["weight"] = row_weights(
        raw_panel["states"]["background"], raw_panel["states"]["group"]
    )

    panel, _accounting = mod.indel_excluded_support(raw_panel, plan, excl_decl, kept_groups)
    kept_global_indices = panel["blocks"].pop("_global_idx")[:, 0].astype(np.int64)

    # Recover Potts cycle contrast v(AB) - v(A) - v(B) matching InteractionAdapter._corner_keys.
    absent = {
        (entry["background"], entry["sequence"])
        for entry in excl_decl["states"]
        if entry["value_without_indel_rows_kcal_mol"] is None
    }
    retained_state_keys = [
        key for key in mod.panel_state_keys(plan, kept_groups) if key not in absent
    ]
    if [name for name, _ in retained_state_keys] != panel["states"]["background"].tolist():
        raise ValueError("Retained state keys do not align with indel-excluded panel states")

    wt_of = {row["name"]: row["wildtype"] for row in plan["backgrounds"]}
    mutations: list[str] = []
    for name, seq in retained_state_keys:
        wt = wt_of[name]
        mutations.append(
            ":".join(f"{wt[i]}{i + 1}{seq[i]}" for i in range(len(wt)) if wt[i] != seq[i])
        )

    potts_raw = np.empty(len(panel["epsilon"]), dtype=np.float64)
    for idx, row in enumerate(panel["cycle_states"]):
        wild_i, low_i, high_i, double_i = (int(x) for x in row)
        if mutations[wild_i]:
            raise ValueError(f"Non-empty wild-type mutation string {mutations[wild_i]!r}")
        bg_name = str(panel["states"]["background"][double_i])
        k_ab = f"{bg_name}{SAMPLE_SEPARATOR}{mutations[double_i]}"
        k_a = f"{bg_name}{SAMPLE_SEPARATOR}{mutations[low_i]}"
        k_b = f"{bg_name}{SAMPLE_SEPARATOR}{mutations[high_i]}"
        potts_raw[idx] = potts_vals[k_ab] - potts_vals[k_a] - potts_vals[k_b]

    panel["blocks"]["POTTS"] = lanes.prepare_scalar(potts_raw, panel["background"])

    group_digest = lanes.identity_digest(sorted(kept_groups))
    cache_dir = (
        root / "results/extensions/explicit_statistics_lanes_20261110/interaction/cache" / group_digest
    )
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Precomputed interaction fold cache not found at {cache_dir}; "
            "cannot evaluate residualized interaction controls."
        )
    _verify_shared_inputs(panel, cache_dir)

    fold_entries: dict[tuple[int, int], dict[str, Any]] = {}
    for seed in mod.SPLIT_SEEDS:
        for outer, held in enumerate(family_folds(panel["group"], mod.OUTER_SPLITS, seed)):
            path = cache_dir / f"{seed}-{outer}.npz"
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if list(meta["held_groups"]) != list(held):
                raise ValueError(f"{path}: cached held_groups do not match family_folds")
            fold_entries[(seed, outer)] = {"path": path, "meta": meta}

    cycle_keys = [
        f"{bg}|{idx}" for bg, idx in zip(panel["background"].tolist(), panel["cycle_index"].tolist())
    ]

    return PairwiseCycleContext(
        root=root,
        plan_path=plan_path,
        plan=plan,
        panel=panel,
        kept_global_indices=kept_global_indices,
        raw_all_epsilon=np.asarray(raw_all_eps, dtype=np.float64),
        raw_all_groups=np.asarray(raw_all_grp),
        raw_all_separations=np.asarray(raw_all_sep),
        raw_all_backgrounds=np.asarray(raw_all_bg),
        potts_cycle_raw=potts_raw,
        weights=row_weights(panel["site_pair"], panel["group"]),
        group_order=sorted(set(panel["group"].tolist())),
        cache_dir=cache_dir,
        fold_entries=fold_entries,
        support_sha256=lanes.identity_digest(cycle_keys),
    )


def _zero_shot_spearman_summary(
    scores: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    separations: np.ndarray,
    *,
    draws: int = mod.BOOTSTRAP_DRAWS,
    seed: int = mod.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _, per_group = mod.group_spearman(target, scores, groups)
    overall = mod.interval(per_group, draws=draws, seed=seed)
    by_sep: dict[str, Any] = {}
    for stratum in STRATA:
        mask = separations == stratum
        if int(mask.sum()) >= 3:
            _, sp_vals = mod.group_spearman(target[mask], scores[mask], groups[mask])
            by_sep[stratum] = mod.interval(sp_vals, draws=draws, seed=seed)
    # Also pooled Spearman across all rows for quick diagnostic visibility.
    pooled = safe_spearman(scores, target)
    return {
        "within_background_spearman": overall,
        "by_separation": by_sep,
        "pooled_spearman": pooled,
    }


def _fit_designs_cached_folds(
    ctx: PairwiseCycleContext,
    blocks_base: dict[str, np.ndarray],
    designs: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Run exact 3-seed x 5-outer x 4-inner nested-CV ridge using cached folds & residual targets."""
    # Cap PyTorch CPU threads at 8 to avoid 192-core OpenMP barrier thrashing on ~1k-2k eigh.
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    panel = ctx.panel
    groups = panel["group"]
    site_pairs = panel["site_pair"]
    weights = ctx.weights
    names = list(designs)

    per_seed_mse: list[dict[str, np.ndarray]] = []
    per_seed_sp: list[dict[str, list[float | None]]] = []
    per_seed_stratum_mse: dict[str, list[dict[str, np.ndarray]]] = {s: [] for s in STRATA}
    oof_targets: list[np.ndarray] = []
    dimensions: dict[str, int] = {}

    for seed in mod.SPLIT_SEEDS:
        predictions = {name: np.full(len(groups), np.nan, dtype=np.float64) for name in names}
        target = np.full(len(groups), np.nan, dtype=np.float64)

        for outer, held in enumerate(family_folds(groups, mod.OUTER_SPLITS, seed)):
            entry = ctx.fold_entries[(seed, outer)]
            with np.load(entry["path"], allow_pickle=False) as cached:
                losses = {name: np.zeros(len(mod.ALPHAS), dtype=np.float64) for name in names}
                for idx, inner in enumerate(entry["meta"]["inner"]):
                    fitting = cached[f"fit_{idx}"]
                    valid = cached[f"valid_{idx}"]
                    yfit = cached[f"yfit_{idx}"]
                    yvalid = cached[f"yvalid_{idx}"]
                    blocks = dict(blocks_base, G=cached[f"G_{idx}"])
                    for name in names:
                        mat = np.column_stack([blocks[k] for k in designs[name]])
                        pred = ridge_predict(
                            mat[fitting], yfit, weights[fitting], mat[valid], mod.ALPHAS, "cpu"
                        )
                        losses[name] += len(inner["validation_groups"]) * (
                            weights[valid][:, None] * (pred - yvalid[:, None]) ** 2
                        ).sum(0)

                train = cached["train"]
                test = cached["test"]
                target[test] = cached["ytest"]
                blocks = dict(blocks_base, G=cached["G"])
                for name in names:
                    alpha = mod.ALPHAS[mod._select_alpha(losses[name])]
                    mat = np.column_stack([blocks[k] for k in designs[name]])
                    predictions[name][test] = ridge_predict(
                        mat[train], cached["ytrain"], weights[train], mat[test], [alpha], "cpu"
                    )[:, 0]
                    dimensions[name] = int(mat.shape[1])

        if not np.isfinite(target).all() or not all(
            np.isfinite(v).all() for v in predictions.values()
        ):
            raise RuntimeError(f"Incomplete out-of-fold predictions on seed {seed}")

        oof_targets.append(target)
        seed_mse: dict[str, np.ndarray] = {}
        seed_sp: dict[str, list[float | None]] = {}
        for name in names:
            labels, mse_vals = mod.group_errors(target, predictions[name], groups, site_pairs)
            if list(labels) != ctx.group_order:
                raise RuntimeError(f"Group support mismatch on design {name}")
            seed_mse[name] = np.asarray(mse_vals, dtype=np.float64)
            _, sp_vals = mod.group_spearman(target, predictions[name], groups)
            seed_sp[name] = sp_vals
        per_seed_mse.append(seed_mse)
        per_seed_sp.append(seed_sp)

        for stratum in STRATA:
            rows = np.flatnonzero(panel["separation"] == stratum)
            if not len(rows):
                continue
            s_mse: dict[str, np.ndarray] = {}
            for name in names:
                _, vals = mod.group_errors(
                    target[rows], predictions[name][rows], groups[rows], site_pairs[rows]
                )
                s_mse[name] = np.asarray(vals, dtype=np.float64)
            per_seed_stratum_mse[stratum].append(s_mse)

    # Average per-group errors and Spearmans across the 3 split seeds (matching InteractionAdapter).
    avg_mse = {
        name: np.mean([d[name] for d in per_seed_mse], axis=0) for name in names
    }
    avg_sp: dict[str, list[float | None]] = {}
    for name in names:
        n_g = len(ctx.group_order)
        g_vals: list[float | None] = []
        for g_idx in range(n_g):
            vals = [d[name][g_idx] for d in per_seed_sp if d[name][g_idx] is not None]
            g_vals.append(float(np.mean(vals)) if len(vals) == len(per_seed_sp) else None)
        avg_sp[name] = g_vals

    avg_stratum_mse: dict[str, dict[str, np.ndarray]] = {}
    for stratum, seed_list in per_seed_stratum_mse.items():
        if seed_list:
            avg_stratum_mse[stratum] = {
                name: np.mean([d[name] for d in seed_list], axis=0) for name in names
            }

    mean_residual_target = np.mean(oof_targets, axis=0)
    return {
        "avg_mse": avg_mse,
        "avg_spearman": avg_sp,
        "avg_stratum_mse": avg_stratum_mse,
        "mean_residual_target": mean_residual_target,
        "dimensions": dimensions,
    }


def _summarize_channel_rungs(
    fit_out: dict[str, Any],
    channel_key: str,
    *,
    draws: int = mod.BOOTSTRAP_DRAWS,
    seed: int = mod.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    avg_mse = fit_out["avg_mse"]
    avg_sp = fit_out["avg_spearman"]
    avg_stratum = fit_out["avg_stratum_mse"]

    model_only_key = channel_key
    rungs_summary: dict[str, Any] = {}
    for rung_name in INTERACTION_RUNGS:
        ctrl_key = f"ctrl::{rung_name}"
        joint_key = f"joint::{rung_name}::{channel_key}"

        ctrl_mse = avg_mse[ctrl_key]
        mod_mse = avg_mse[model_only_key]
        joint_mse = avg_mse[joint_key]

        fwd_mse_red = ctrl_mse - joint_mse
        rev_mse_red = mod_mse - joint_mse

        ctrl_sp = avg_sp[ctrl_key]
        mod_sp = avg_sp[model_only_key]
        joint_sp = avg_sp[joint_key]
        fwd_sp_inc = [
            None if a is None or b is None else b - a for a, b in zip(ctrl_sp, joint_sp)
        ]
        rev_sp_inc = [
            None if a is None or b is None else b - a for a, b in zip(mod_sp, joint_sp)
        ]

        strata_fwd: dict[str, Any] = {}
        for stratum, s_map in avg_stratum.items():
            strata_fwd[stratum] = mod.interval(
                s_map[ctrl_key] - s_map[joint_key], draws=draws, seed=seed
            )

        rungs_summary[rung_name] = {
            "control_mse_kcal2": mod.interval(ctrl_mse, draws=draws, seed=seed),
            "model_only_mse_kcal2": mod.interval(mod_mse, draws=draws, seed=seed),
            "joint_mse_kcal2": mod.interval(joint_mse, draws=draws, seed=seed),
            "control_spearman": mod.interval(ctrl_sp, draws=draws, seed=seed),
            "model_only_spearman": mod.interval(mod_sp, draws=draws, seed=seed),
            "joint_spearman": mod.interval(joint_sp, draws=draws, seed=seed),
            "forward_model_given_controls": {
                "mse_reduction_kcal2": mod.interval(fwd_mse_red, draws=draws, seed=seed),
                "spearman_increment": mod.interval(fwd_sp_inc, draws=draws, seed=seed),
                "by_separation_mse_reduction_kcal2": strata_fwd,
            },
            "reverse_controls_given_model": {
                "mse_reduction_kcal2": mod.interval(rev_mse_red, draws=draws, seed=seed),
                "spearman_increment": mod.interval(rev_sp_inc, draws=draws, seed=seed),
            },
        }
    return rungs_summary


def evaluate_r3_pairwise_cycles_arrays(
    eps_pll_all: np.ndarray,
    arm: str,
    *,
    ctx: PairwiseCycleContext,
    r_cycle_all: np.ndarray | None = None,
    source_path: str = "",
    source_sha256: str = "",
    bootstrap_draws: int = mod.BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Evaluate R3 double-mutant epistasis cycle arrays on CPU against the interaction hierarchy."""
    eps_pll_all = np.asarray(eps_pll_all, dtype=np.float64)
    if eps_pll_all.shape != ctx.raw_all_epsilon.shape:
        raise ValueError(
            f"{arm}: cycle_eps_pll shape {eps_pll_all.shape} != expected {ctx.raw_all_epsilon.shape}"
        )
    if r_cycle_all is not None:
        r_cycle_all = np.asarray(r_cycle_all, dtype=np.float64)
        mod.require_projected_width(r_cycle_all.shape[1], arm=arm, source=source_path or arm)

    # Extract exact complete-case indel-excluded rows.
    idx = ctx.kept_global_indices
    eps_pll_cc = eps_pll_all[idx]
    panel = ctx.panel

    blocks_base: dict[str, np.ndarray] = dict(panel["blocks"])
    blocks_base["M"] = eps_pll_cc[:, None]
    if r_cycle_all is not None:
        blocks_base["R"] = r_cycle_all[idx]

    # Fit arm-invariant control rungs once per context and reuse across arms.
    if ctx._control_fit_cache is None:
        ctrl_designs = {f"ctrl::{k}": v for k, v in INTERACTION_RUNGS.items()}
        ctx._control_fit_cache = _fit_designs_cached_folds(ctx, panel["blocks"], ctrl_designs)
    ctrl_fit = ctx._control_fit_cache

    # Fit only arm-specific designs.
    arm_designs: dict[str, tuple[str, ...]] = {"M": ("M",)}
    for rung_name, cols in INTERACTION_RUNGS.items():
        arm_designs[f"joint::{rung_name}::M"] = (*cols, "M")
    if r_cycle_all is not None:
        arm_designs["R"] = ("R",)
        for rung_name, cols in INTERACTION_RUNGS.items():
            arm_designs[f"joint::{rung_name}::R"] = (*cols, "R")
        top_cols = INTERACTION_RUNGS["nuisance+profile+couplings"]
        arm_designs["joint::nuisance+profile+couplings::MR"] = (*top_cols, "M", "R")

    arm_fit = _fit_designs_cached_folds(ctx, blocks_base, arm_designs)
    fit_out = {
        "avg_mse": {**ctrl_fit["avg_mse"], **arm_fit["avg_mse"]},
        "avg_spearman": {**ctrl_fit["avg_spearman"], **arm_fit["avg_spearman"]},
        "avg_stratum_mse": {
            s: {**ctrl_fit["avg_stratum_mse"].get(s, {}), **arm_fit["avg_stratum_mse"].get(s, {})}
            for s in STRATA
            if s in ctrl_fit["avg_stratum_mse"] or s in arm_fit["avg_stratum_mse"]
        },
        "mean_residual_target": ctrl_fit["mean_residual_target"],
        "dimensions": {**ctrl_fit["dimensions"], **arm_fit["dimensions"]},
    }

    # Zero-shot Spearmans (full 64-group raw, 61-group indel-excluded raw, 61-group residualized, and Potts ref).
    zs_full_raw = _zero_shot_spearman_summary(
        eps_pll_all,
        ctx.raw_all_epsilon,
        ctx.raw_all_groups,
        ctx.raw_all_separations,
        draws=bootstrap_draws,
    )
    zs_cc_raw = _zero_shot_spearman_summary(
        eps_pll_cc,
        panel["epsilon"],
        panel["group"],
        panel["separation"],
        draws=bootstrap_draws,
    )
    zs_cc_residual = _zero_shot_spearman_summary(
        eps_pll_cc,
        fit_out["mean_residual_target"],
        panel["group"],
        panel["separation"],
        draws=bootstrap_draws,
    )
    zs_potts_raw = _zero_shot_spearman_summary(
        ctx.potts_cycle_raw,
        panel["epsilon"],
        panel["group"],
        panel["separation"],
        draws=bootstrap_draws,
    )

    channels: dict[str, Any] = {
        "M_cycle": _summarize_channel_rungs(fit_out, "M", draws=bootstrap_draws),
    }
    if r_cycle_all is not None:
        channels["R_cycle"] = _summarize_channel_rungs(fit_out, "R", draws=bootstrap_draws)
        ctrl_top = fit_out["avg_mse"]["ctrl::nuisance+profile+couplings"]
        joint_mr = fit_out["avg_mse"]["joint::nuisance+profile+couplings::MR"]
        ctrl_top_sp = fit_out["avg_spearman"]["ctrl::nuisance+profile+couplings"]
        joint_mr_sp = fit_out["avg_spearman"]["joint::nuisance+profile+couplings::MR"]
        channels["MR_cycle_over_nuisance_profile_couplings"] = {
            "joint_mse_kcal2": mod.interval(joint_mr, draws=bootstrap_draws),
            "joint_spearman": mod.interval(joint_mr_sp, draws=bootstrap_draws),
            "forward_mse_reduction_kcal2": mod.interval(
                ctrl_top - joint_mr, draws=bootstrap_draws
            ),
            "forward_spearman_increment": mod.interval(
                [
                    None if a is None or b is None else b - a
                    for a, b in zip(ctrl_top_sp, joint_mr_sp)
                ],
                draws=bootstrap_draws,
            ),
        }

    return {
        "schema": SCHEMA,
        "arm": arm,
        "npz_path": source_path,
        "npz_sha256": source_sha256,
        "support": {
            "name": "complete_case_indel_excluded",
            "support_sha256": ctx.support_sha256,
            "n_groups": len(ctx.group_order),
            "n_cycles": int(len(panel["epsilon"])),
            "n_site_pairs": int(len(set(panel["site_pair"].tolist()))),
            "target": "measured_singles_residualized_epsilon_kcal_mol",
            "split_seeds": list(mod.SPLIT_SEEDS),
            "outer_splits": mod.OUTER_SPLITS,
            "inner_splits": mod.INNER_SPLITS,
            "bootstrap_draws": bootstrap_draws,
            "control_rungs": {k: list(v) for k, v in INTERACTION_RUNGS.items()},
        },
        "zero_shot": {
            "all_64_groups_raw_epsilon": zs_full_raw,
            "complete_case_61_groups_raw_epsilon": zs_cc_raw,
            "complete_case_61_groups_residualized_epsilon": zs_cc_residual,
            "potts_reference_complete_case_raw_epsilon": zs_potts_raw,
        },
        "nested_cv_channels": channels,
        "design_dimensions": fit_out["dimensions"],
    }


def evaluate_r3_pairwise_cycles_npz(
    npz_path: Path,
    arm: str,
    *,
    ctx: PairwiseCycleContext,
    bootstrap_draws: int = mod.BOOTSTRAP_DRAWS,
    include_r_cycle: bool = True,
) -> dict[str, Any]:
    """Evaluate a completed ``r3_pairwise_cycles_<arm>.npz`` on CPU against the interaction hierarchy."""
    with np.load(npz_path, allow_pickle=False) as data:
        if "cycle_eps_pll" not in data:
            raise KeyError(f"{npz_path}: missing required array 'cycle_eps_pll'")
        eps_pll_all = np.asarray(data["cycle_eps_pll"], dtype=np.float64)
        has_r_cycle = include_r_cycle and ("R_cycle" in data)
        r_cycle_all = (
            np.asarray(data["R_cycle"], dtype=np.float64) if has_r_cycle else None
        )
    return evaluate_r3_pairwise_cycles_arrays(
        eps_pll_all,
        arm,
        ctx=ctx,
        r_cycle_all=r_cycle_all,
        source_path=str(npz_path),
        source_sha256=sha256_file(npz_path),
        bootstrap_draws=bootstrap_draws,
    )
