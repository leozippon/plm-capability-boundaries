#!/usr/bin/env python3
"""Publication figures from completed frozen analyses and separately pinned references."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation.generation_biology_analysis import index_rows, merge_reference_annotations, read_jsonl


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No source records for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def finish(fig, out, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=350, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def reference_panel(ax, rows, prefix, title):
    aligned = [r for r in rows if r.get(prefix + "search_status") == "aligned_query_coverage_available"]
    display = ax.scatter([r[prefix + "identity"] for r in aligned], [r["mean_ca_plddt"] for r in aligned],
                         c=[r[prefix + "coverage"] for r in aligned], vmin=0, vmax=1,
                         cmap="viridis", s=17, alpha=.8)
    absent = [r for r in rows if r.get(prefix + "search_status") == "no_reported_alignment"]
    if len(aligned) + len(absent) != len(rows):
        raise ValueError("Every plotted reference outcome must be aligned or explicitly no-hit")
    divider = make_axes_locatable(ax)
    no_hit_axis = divider.append_axes("left", size="22%", pad=.18, sharey=ax)
    no_hit_axis.scatter(np.linspace(-.2, .2, len(absent)), [r["mean_ca_plddt"] for r in absent],
                        color="#777777", s=13, alpha=.5)
    no_hit_axis.set(xlim=(-.5, .5), ylim=(0, 100), ylabel="Mean CA-pLDDT")
    no_hit_axis.set_xticks([0], ["No hit"])
    no_hit_axis.tick_params(axis="x", labelsize=8)
    no_hit_axis.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set(xlim=(-3, 103), ylim=(0, 100), xlabel="Query-normalized\nidentity (%)",
           title=f"{title}\n{len(aligned)}/{len(rows)} aligned")
    ax.spines[["top", "right"]].set_visible(False)
    return display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--reference-sidecar", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    if report["phase"] != "main" or not report["terminal_predictions_complete"]:
        raise ValueError("Publication structural figures require the complete main analysis")
    attempts = read_jsonl(args.attempts)
    sources = [args.analysis, args.attempts]
    if args.reference_sidecar:
        sidecar = read_jsonl(args.reference_sidecar)
        if set(index_rows(attempts)) != set(index_rows(sidecar)):
            raise ValueError("Fresh reference sidecar must cover every ledger ID exactly")
        attempts = merge_reference_annotations(attempts, sidecar)
        sources.append(args.reference_sidecar)
    native_annotations = bool(attempts and all(r["arm"] == "progen3-3b"
        and isinstance(r.get("any_profile_hit"), bool)
        and r.get("reference_search_status") in {"aligned_query_coverage_available", "no_reported_alignment"}
        for r in attempts))
    population = index_rows(attempts)
    pairs = []
    for row in report["pair_sufficient_statistics"]:
        base = population[row["id"]]
        if base["sequence_sha256"] != row["sequence_sha256"]:
            raise ValueError("Structural statistic and reference row have different sequences")
        metadata = {k: v for k, v in base.items() if k.startswith("reference_") or k in {
            "any_profile_hit", "official_compilation_valid", "source_budget_censored", "source_stop_reason",
            "residue_prefix_is_whole_generated_sequence", "source_termination_observed"}}
        pairs.append(row | metadata)
    arms = sorted(report["arms"], key=lambda arm: ({"zymctrl": 0, "prollama": 1, "progen3-3b": 2}.get(arm, 3), arm))
    condition = report["primary_condition"]
    args.out.mkdir(parents=True, exist_ok=True)
    for font_path in font_manager.findSystemFonts():
        if Path(font_path).name.lower() in {"arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"}:
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8.5,
                         "axes.linewidth": 1, "lines.linewidth": 1, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, len(arms), figsize=(2.55 * len(arms), 5.8), squeeze=False)
    colors = {condition: "#0072B2", "mismatched": "#D55E00", "natural": "#555555"}
    for column, arm in enumerate(arms):
        scatter, contrasts = axes[:, column]
        for role, cond, marker, label in [("generation", condition, "o", "Generated"),
                                           ("natural_reference", None, "^", "Natural reference")]:
            subset = [r for r in pairs if r["arm"] == arm and r["role"] == role
                      and (cond is None or r["condition"] == cond)
                      and r["status"] == r["shuffle_status"] == "ok"]
            if not subset:
                continue
            scatter.scatter([r["shuffle_mean_ca_plddt"] for r in subset],
                            [r["mean_ca_plddt"] for r in subset], s=12, marker=marker,
                            alpha=.5, color=colors[condition] if role == "generation" else colors["natural"],
                            label=label)
        scatter.plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1)
        scatter.set(xlim=(0, 100), ylim=(0, 100), xlabel="Shuffle mean CA-pLDDT",
                    ylabel="Original mean CA-pLDDT", title={"zymctrl": "ZymCTRL", "prollama": "ProLLaMA", "progen3-3b": "ProGen3-3B"}.get(arm, arm))
        scatter.legend(fontsize=8, frameon=False, loc="lower right")
        values = report["arms"][arm]["class_values"]
        classes = sorted(values, key=str)
        offsets = {condition: -.16, "mismatched": .16}
        for cond in dict.fromkeys((condition, "mismatched")):
            rows = [r for r in report["structural_cells"] if r["arm"] == arm
                    and r["role"] == "generation" and r["condition"] == cond]
            if not rows:
                continue
            contrasts.scatter([r["paired_mean_ca_plddt_delta"] for r in rows],
                              [classes.index(str(r["class_key"])) + offsets[cond] for r in rows],
                              s=17, color=colors[cond], label=cond.capitalize())
        contrasts.axvline(0, color="black", linestyle="--", linewidth=1)
        contrasts.set_yticks(range(len(classes)), [c if c not in (None, "None") else "Unconditional task" for c in classes])
        contrasts.set_xlabel("Original − own shuffle\nCA-pLDDT")
        contrasts.legend(fontsize=8, frameon=False)
        result = report["arms"][arm]
        interval = result["ci95"] if report["uncertainty_unit"] == "sequence_cluster" else result["ci97_5"]
        interval_label = "95%" if report["uncertainty_unit"] == "sequence_cluster" else "97.5%"
        if interval is not None:
            contrasts.set_title(f"Mean {result['mean']:.2f}\n{interval_label} CI [{interval[0]:.2f}, {interval[1]:.2f}]", fontsize=8.5)
        contrasts.invert_yaxis()
    if report["uncertainty_unit"] == "sequence_cluster":
        for column, arm in enumerate(arms):
            ax = axes[1, column]
            ax.clear()
            result = report["arms"][arm]
            ci = result["ci95"]
            ax.errorbar(result["mean"], 0, xerr=[[result["mean"] - ci[0]], [ci[1] - result["mean"]]],
                        fmt="o", color="#0072B2", capsize=4, markersize=5)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set(ylim=(-.5, .5), xlabel="Original − shuffle CA-pLDDT", title="Native task: mean and 95% interval")
            ax.set_yticks([])
        fig.set_size_inches(5.1, 3)
        for column in range(len(arms)):
            axes[0, column].set_position([.09, .2, .36, .67])
            axes[1, column].set_position([.60, .34, .36, .42])
    for index, ax in enumerate(axes.flat):
        ax.text(-.14, 1.04, chr(97 + index), transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=1)
    if report["uncertainty_unit"] != "sequence_cluster":
        fig.tight_layout(h_pad=1.8, w_pad=1.2)
    finish(fig, args.out, "structural-evidence")
    write_csv(args.out / "structural-pair-source.csv", pairs)
    write_csv(args.out / "structural-cell-source.csv", report["structural_cells"])

    if report["uncertainty_unit"] == "class":
        class_source = []
        for cell in report["structural_cells"]:
            if cell["role"] != "generation" or cell["condition"] != condition:
                continue
            profile = next(row for row in report["profile_ledger"] if row["arm"] == cell["arm"]
                           and row["class_key"] == cell["class_key"] and row["role"] == "generation"
                           and row["condition"] == condition)
            class_source.append({"arm": cell["arm"], "class_key": cell["class_key"],
                "n_attempts": profile["n_attempts"], "n_target_profile": profile["n_target_profile"],
                "n_any_profile": round(profile["any_profile_rate"] * profile["n_attempts"]),
                "n_distinct_target_groups": profile["n_distinct_target_groups"],
                "mean_ca_plddt": cell["mean_ca_plddt"], "paired_delta": cell["paired_mean_ca_plddt_delta"],
                "event_lower": cell["confidence_event_all_attempt_lower_estimate"],
                "event_upper": cell["confidence_event_all_attempt_upper_estimate"],
                "joint_lower": cell["joint_profile_confidence"]["lower_estimate"],
                "joint_upper": cell["joint_profile_confidence"]["upper_estimate"]})
        class_source.sort(key=lambda row: (row["arm"] != "zymctrl", row["arm"], row["class_key"]))
        write_csv(args.out / "class-evidence-source.csv", class_source)
        def percent_range(low, high):
            return f"{100 * low:.1f}" if abs(low - high) < 1e-9 else f"{100 * low:.1f}--{100 * high:.1f}"
        lines = []
        last_arm = None
        for row in class_source:
            if row["arm"] != last_arm:
                name = {"zymctrl": "ZymCTRL", "prollama": "ProLLaMA"}.get(row["arm"], row["arm"])
                lines.append(r"\rowcolor{tableblue}\multicolumn{8}{l}{\textbf{" + name + r"}} \\")
                last_arm = row["arm"]
            lines.append(" & ".join([row["class_key"], str(row["n_target_profile"]), str(row["n_any_profile"]),
                str(row["n_distinct_target_groups"]), f"{row['mean_ca_plddt']:.1f}", f"{row['paired_delta']:.1f}",
                percent_range(row["event_lower"], row["event_upper"]), percent_range(row["joint_lower"], row["joint_upper"])]) + r" \\")
        (args.out / "class-evidence-table-rows.tex").write_text("\n".join(lines) + "\n")
        fig, axes = plt.subplots(2, len(arms), figsize=(2.55 * len(arms), 4.7), squeeze=False)
        distribution_source = []
        for column, arm in enumerate(arms):
            for role, cond, shuffle, label, color, style in [
                ("generation", condition, False, "Requested", "#0072B2", "-"),
                ("generation", "mismatched", False, "Mismatched", "#D55E00", "-"),
                ("natural_reference", None, False, "Natural reference", "#555555", "-"),
                ("generation", condition, True, "Requested: own shuffle", "#0072B2", "--"),
            ]:
                rows = [r for r in pairs if r["arm"] == arm and r["role"] == role
                        and (cond is None or r["condition"] == cond)
                        and r["status"] == r["shuffle_status"] == "ok"]
                mass = defaultdict(float)
                for row in rows:
                    mass[row["class_key"]] += 1 / row["inclusion_probability"]
                for row in rows:
                    weight = 1 / row["inclusion_probability"] / mass[row["class_key"]] / len(mass)
                    distribution_source.append({"id": row["id"], "arm": arm, "class_key": row["class_key"],
                        "curve": label, "class_balanced_weight": weight,
                        "mean_ca_plddt": row[("shuffle_" if shuffle else "") + "mean_ca_plddt"],
                        "ptm": row[("shuffle_" if shuffle else "") + "ptm"]})
                curve = distribution_source[-len(rows):]
                for ax, metric in zip(axes[:, column], ("mean_ca_plddt", "ptm")):
                    ordered = sorted(curve, key=lambda r: r[metric])
                    ax.step([r[metric] for r in ordered], np.cumsum([r["class_balanced_weight"] for r in ordered]),
                            where="post", color=color, linestyle=style, label=label)
            axes[0, column].set(xlim=(0, 100), xlabel="Mean CA-pLDDT",
                                title={"zymctrl": "ZymCTRL", "prollama": "ProLLaMA"}.get(arm, arm))
            axes[1, column].set(xlim=(0, 1), xlabel="pTM")
        for number, ax in enumerate(axes.flat):
            ax.set(ylim=(0, 1.02), ylabel="Cumulative fraction")
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(-.14, 1.04, chr(97 + number), transform=ax.transAxes, fontweight="bold", fontsize=10)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, [label.replace("Requested: own shuffle", "Own shuffle") for label in labels],
                   frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(.5, .01))
        fig.tight_layout(h_pad=1.8, w_pad=1.2, rect=(0, .15, 1, 1))
        finish(fig, args.out, "confidence-distributions")
        write_csv(args.out / "confidence-distribution-source.csv", distribution_source)

    native_pairs = [r for r in pairs if r["role"] == "generation"
                    and isinstance(r.get("official_compilation_valid"), bool)]
    if native_pairs:
        native_summary = []
        if native_annotations:
            fig, matrix = plt.subplots(2, 2, figsize=(5.1, 5.4))
            axes = matrix.flat
        else:
            fig, axes = plt.subplots(1, 2, figsize=(5.1, 3.2))
        for valid, label, color, marker in [(True, "Compiled", "#0072B2", "o"),
                                            (False, "Censored prefix", "#D55E00", "^")]:
            full = [r for r in attempts if r["role"] == "generation"
                    and r.get("official_compilation_valid") is valid]
            selected = [r for r in native_pairs if r["official_compilation_valid"] is valid]
            complete = [r for r in selected if r["status"] == r["shuffle_status"] == "ok"]
            if any(not r.get("source_budget_censored") for r in full if not valid):
                raise ValueError("Additional native failure categories need their own reported stratum")
            weights = np.array([1 / r["inclusion_probability"] for r in complete])
            denominator = sum(1 / r["inclusion_probability"] for r in native_pairs)
            native_summary.append({"output_category": label, "n_population": len(full),
                                   "n_selected": len(selected), "n_complete_pairs": len(complete),
                                   "n_any_profile_full_population": sum(r.get("any_profile_hit") is True for r in full),
                                   "n_reference_aligned_full_population": sum(r.get("reference_search_status") == "aligned_query_coverage_available" for r in full),
                                   "weighted_sample_mass": float(weights.sum()),
                                   "weighted_confidence_event_contribution_all_attempts":
                                   sum(w * r["confidence_event"] for w, r in zip(weights, complete)) / denominator,
                                   **{key: float(np.average([r[key] for r in complete], weights=weights))
                                      if len(complete) else None for key in
                                      ("mean_ca_plddt", "paired_delta", "ptm")},
                                   "scope": "posthoc_compilation_stratum_descriptive_fixed_sampling_weights"})
            axes[0].scatter([r["length"] for r in complete], [r["mean_ca_plddt"] for r in complete],
                            s=18, marker=marker, alpha=.65, color=color,
                            label=f"{label} (n={len(complete)})")
            axes[1].scatter([r["shuffle_mean_ca_plddt"] for r in complete],
                            [r["mean_ca_plddt"] for r in complete], s=18,
                            marker=marker, alpha=.65, color=color)
        axes[0].set(xlabel="Sequence length (aa)", ylabel="Mean CA-pLDDT", ylim=(0, 100))
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot([0, 100], [0, 100], "--", color="black", linewidth=1)
        axes[1].set(xlabel="Shuffle mean CA-pLDDT", ylabel="Original mean CA-pLDDT", xlim=(0, 100), ylim=(0, 100))
        if native_annotations:
            original_rows = [r for r in native_pairs if r["status"] == "ok"]
            display = reference_panel(axes[2], original_rows, "reference_", "Reference support")
            fig.colorbar(display, ax=axes[2], label="Query coverage")
            ordered = sorted(original_rows, key=lambda r: r["paired_delta"])
            weights = np.array([1 / r["inclusion_probability"] for r in ordered])
            axes[3].step([r["paired_delta"] for r in ordered], np.cumsum(weights) / weights.sum(),
                         where="post", color="#0072B2", label="Paired differences")
            result = report["arms"][arms[0]]
            lo, hi = result["ci95"]
            axes[3].axvspan(lo, hi, color="#0072B2", alpha=.14, label="Mean: 95% interval")
            axes[3].axvline(result["mean"], color="#0072B2", linewidth=1)
            axes[3].axvline(0, color="black", linestyle="--", linewidth=1)
            axes[3].set(xlabel="Original − own shuffle\nCA-pLDDT", ylabel="Cumulative fraction",
                        ylim=(0, 1.02), title=f"Mean {result['mean']:.2f}\n95% CI [{lo:.2f}, {hi:.2f}]")
            axes[3].legend(frameon=False, fontsize=8, loc="upper left")
        for index, ax in enumerate(axes):
            ax.text(-.23, 1.12, chr(97 + index), transform=ax.transAxes, fontweight="bold", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(w_pad=2.4, h_pad=2.4)
        finish(fig, args.out, "native-completion-evidence")
        write_csv(args.out / "native-completion-source.csv", native_pairs)
        write_csv(args.out / "native-completion-summary.csv", native_summary)

    if args.reference_sidecar or native_annotations:
        prefix = "reference_fresh_" if args.reference_sidecar else "reference_"
        grouped = defaultdict(list)
        for row in attempts:
            if row["role"] == "generation":
                grouped[(row["arm"], row["class_key"], row["condition"], row["primary_class"])].append(row)
        summary = []
        for (arm, cls, cond, primary), rows in sorted(grouped.items(), key=str):
            aligned = [r for r in rows if r.get(prefix + "search_status") == "aligned_query_coverage_available"]
            coverage_keys = (prefix + "coverage", prefix + "target_coverage") if args.reference_sidecar else (prefix + "coverage",)
            for key in coverage_keys:
                if any(r.get(key) is None or not 0 <= r[key] <= 1 for r in aligned):
                    raise ValueError(f"Aligned reference row lacks valid same-hit {key}")
            comparable = [r for r in rows if r.get("reference_identity") is not None
                          and r.get("reference_fresh_identity") is not None]
            summary.append({"arm": arm, "class_key": cls, "condition": cond, "primary_class": primary,
                            "n_attempts": len(rows), "n_aligned": len(aligned),
                            "n_no_hit": sum(r.get(prefix + "search_status") == "no_reported_alignment" for r in rows),
                            "n_legacy_fresh_identity_comparable": len(comparable),
                            "n_legacy_fresh_identity_changed": sum(abs(r["reference_identity"] - r["reference_fresh_identity"]) > 1e-9 for r in comparable),
                            **{key + "_median_among_aligned": float(np.median([r[key] for r in aligned])) if aligned else None
                               for key in (prefix + "identity",) + coverage_keys}})
        write_csv(args.out / "reference-accounting.csv", summary)
        generated = [r for r in pairs if r["role"] == "generation" and r["condition"] == condition and r["status"] == "ok"]
        fig, axes = plt.subplots(1, len(arms), figsize=(2.55 * len(arms), 3.1), squeeze=False)
        for index, (ax, arm) in enumerate(zip(axes[0], arms)):
            rows = [r for r in generated if r["arm"] == arm]
            title = {"zymctrl": "ZymCTRL", "prollama": "ProLLaMA", "progen3-3b": "ProGen3-3B"}.get(arm, arm)
            display = reference_panel(ax, rows, prefix, title)
            if len(arms) > 1:
                ax.text(-.55, 1.16, chr(97 + index), transform=ax.transAxes, fontweight="bold", fontsize=10)
            fig.colorbar(display, ax=ax, label="Query coverage")
        fig.tight_layout()
        finish(fig, args.out, "reference-supported-confidence")
        write_csv(args.out / "reference-structure-source.csv", generated)
    receipt = {"input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               "reference_channel": ("fresh same-hit identity and coverage; legacy retained separately" if args.reference_sidecar
                                     else "native first annotation; target coverage unavailable" if native_annotations else "not joined"),
               "primary_statistics_unchanged": True}
    (args.out / "structural-figure-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
