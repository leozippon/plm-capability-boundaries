#!/usr/bin/env python3
"""Render supplementary measurement checks from retained inputs and completed calibration."""
from __future__ import annotations

import argparse
from collections import Counter
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation.generation_biology_analysis import read_jsonl, supported

NAMES = {"zymctrl": "ZymCTRL", "prollama": "ProLLaMA"}


def save(fig, output, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table(output, name, rows):
    with (output / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--pilot-analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    attempts, selected = read_jsonl(args.attempts), read_jsonl(args.subset)
    pilot = json.loads(args.pilot_analysis.read_text())
    if pilot["phase"] != "pilot" or not pilot["terminal_predictions_complete"]:
        raise ValueError("Supplement requires completed control-only calibration")
    args.out.mkdir(parents=True, exist_ok=True)
    for p in font_manager.findSystemFonts():
        if Path(p).name.lower() in {"arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"}:
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.linewidth": 1,
                         "lines.linewidth": 1, "svg.fonttype": "none", "pdf.fonttype": 42})
    arms = ["zymctrl", "prollama"]
    fig, axes = plt.subplots(2, 2, figsize=(5.1, 5.8))
    rows = []
    for row_index, arm in enumerate(arms):
        classes = sorted({r["class_key"] for r in attempts if r["arm"] == arm and r["role"] == "generation"})
        labels = classes + ["No Pfam", "Other Pfam", "Multi-class"]
        for col, condition in enumerate(("requested", "mismatched")):
            matrix = []
            for cls in classes:
                cell = [r for r in attempts if r["arm"] == arm and r["role"] == "generation"
                        and r["class_key"] == cls and r["condition"] == condition]
                assert len(cell) == 200
                if any(not isinstance(r["any_profile_hit"], bool) for r in cell):
                    raise ValueError("Unknown profile result in retained census")
                counts = Counter(label for r in cell for label in r["profile_hit_classes"])
                counts.update({"No Pfam": sum(not r["any_profile_hit"] for r in cell),
                               "Other Pfam": sum(r["any_profile_hit"] and not r["profile_hit_classes"] for r in cell),
                               "Multi-class": sum(len(r["profile_hit_classes"]) > 1 for r in cell)})
                matrix.append([counts[label] / len(cell) for label in labels])
                rows.extend({"arm": arm, "condition": condition, "target_class": cls,
                             "recognized_class_or_status": label, "n_matches": counts[label],
                             "n_attempts": len(cell), "fraction": counts[label] / len(cell)} for label in labels)
            ax = axes[row_index, col]
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(labels)), [str(i + 1) for i in range(len(classes))] + ["N", "O", "M"], rotation=90, fontsize=7.5)
            ax.set_yticks(range(len(classes)), [f"{i + 1}: {cls}" for i, cls in enumerate(classes)], fontsize=8)
            ax.set_title(condition.capitalize() if row_index == 0 else "")
            ax.set_xlabel("Matched class / outcome")
            if col == 0:
                ax.set_ylabel(f"{NAMES[arm]} target class")
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.text(-.08, 1.035, chr(97 + 2 * row_index + col), transform=ax.transAxes,
                    fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=2, w_pad=.5, rect=(0, .1, 1, 1))
    color_axis = fig.add_axes([.32, .045, .52, .018])
    fig.colorbar(im, cax=color_axis, orientation="horizontal", label="Fraction of all 200 attempts")
    save(fig, args.out, "profile-confusion")
    table(args.out, "profile-confusion-source", rows)

    fig, axes = plt.subplots(1, 2, figsize=(5.1, 5.2))
    rows = []
    pilot_hashes = set(pilot["natural_control_sequence_sha256"])
    for ax, arm in zip(axes, arms):
        classes = sorted(pilot["arms"][arm]["class_values"])
        for index, cls in enumerate(classes):
            for role, offset, color in [("generation", -.15, "#0072B2"), ("natural_reference", .15, "#555555")]:
                population = [r for r in attempts if r["arm"] == arm and r["class_key"] == cls
                              and r["role"] == role and (role != "generation" or r["condition"] == "requested")
                              and (role != "natural_reference" or r["sequence_sha256"] not in pilot_hashes)]
                sample = [r for r in selected if r["arm"] == arm and r["class_key"] == cls
                          and r["role"] == role and (role != "generation" or r["condition"] == "requested")]
                q = np.quantile([r["length"] for r in population], [.25, .5, .75])
                sample_median = float(np.median([r["length"] for r in sample]))
                ax.plot(q[[0, 2]], [index + offset] * 2, color=color, linewidth=1.5)
                ax.scatter(q[1], index + offset, s=14, color=color)
                ax.scatter(sample_median, index + offset, s=20, facecolor="white", edgecolor=color, marker="s")
                rows.append({"arm": arm, "class_key": cls, "role": role, "n_population": len(population),
                             "n_supported": sum(supported(r) for r in population), "n_selected": len(sample),
                             "length_q25": q[0], "length_median": q[1], "length_q75": q[2],
                             "selected_length_median": sample_median})
        ax.set_yticks(range(len(classes)), classes)
        ax.set(xlabel="Residue length", title=NAMES[arm], xlim=(0, 1150))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot([], [], color="#0072B2", label="Requested generation")
    axes[0].plot([], [], color="#555555", label="Natural reference")
    axes[0].scatter([], [], facecolor="white", edgecolor="black", marker="s", label="Selected median")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False,
               loc="upper center", ncol=1)
    fig.tight_layout(w_pad=2, rect=(0, 0, 1, .88))
    save(fig, args.out, "length-support")
    table(args.out, "length-support-source", rows)

    fig, axes = plt.subplots(1, 2, figsize=(5.1, 3.8))
    rows = []
    for ax, arm in zip(axes, arms):
        result = pilot["arms"][arm]
        for index, (cls, value) in enumerate(result["class_values"].items()):
            ax.scatter(value, index, s=20, color="#0072B2")
            rows.append({"arm": arm, "class_key": cls, "natural_minus_shuffle_mean_ca_plddt": value,
                         "n_pairs": 2, "aggregate_mean": result["mean"],
                         "aggregate_ci95_low": result["ci95"][0], "aggregate_ci95_high": result["ci95"][1]})
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvspan(*result["ci95"], color="#0072B2", alpha=.12)
        ax.axvline(result["mean"], color="#0072B2", linewidth=1)
        ax.set_yticks(range(len(result["class_values"])), list(result["class_values"]))
        ax.set(xlabel="Natural − own shuffle\nCA-pLDDT", title=NAMES[arm], xlim=(-2, 80))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2)
    save(fig, args.out, "natural-calibration")
    table(args.out, "natural-calibration-source", rows)
    (args.out / "provenance.json").write_text(json.dumps({
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.attempts, args.subset, args.pilot_analysis)},
        "generated_structure_outcomes_used": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
