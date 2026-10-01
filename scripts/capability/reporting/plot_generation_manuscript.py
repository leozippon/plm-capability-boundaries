#!/usr/bin/env python3
"""Render the retained full-attempt class panel; never read partial structure output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.generation.generation_biology_analysis import profile_ledger, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = profile_ledger(read_jsonl(args.attempts))
    cells = [r for r in records if r["role"] == "generation"]
    if not cells or any(r["target_profile_rate"] is None for r in cells):
        raise ValueError("The conditional class panel requires observed target profiles")
    args.out.mkdir(parents=True, exist_ok=True)
    for font_path in font_manager.findSystemFonts():
        if Path(font_path).name.lower() in {"arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"}:
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                         "font.size": 8.5, "axes.linewidth": 1.0, "lines.linewidth": 1.0,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(5.1, 5.3), sharex=True)
    export = []
    for ax, arm, label, title in zip(axes, ("zymctrl", "prollama"), ("a", "b"),
                                    ("ZymCTRL · EC classes", "ProLLaMA · superfamilies")):
        requested = sorted((r for r in cells if r["arm"] == arm and r["condition"] == "requested"),
                           key=lambda r: r["class_key"])
        if len(requested) != 16:
            raise ValueError(f"Expected every original class for {arm}")
        for y, row in enumerate(requested):
            negative = next(r for r in cells if r["arm"] == arm
                            and r["class_key"] == row["class_key"] and r["condition"] == "mismatched")
            color = "#0072B2" if row["primary_class"] else "#909090"
            ax.plot([negative["target_profile_rate"], row["target_profile_rate"]], [y, y],
                    color=color, alpha=0.4, zorder=1)
            ax.scatter(row["target_profile_rate"], y, s=23, color=color, zorder=3,
                       label="Requested" if y == 0 else None)
            ax.scatter(negative["target_profile_rate"], y, s=21, facecolors="white", edgecolors=color,
                       zorder=3, label="Mismatched" if y == 0 else None)
            ax.scatter(row["distinct_target_groups_per_attempt"], y + .18, marker="s", s=16,
                       color="#D55E00" if row["primary_class"] else "#909090", zorder=4,
                       label="Recognized groups / attempts" if y == 0 else None)
            export.append({key: row[key] for key in ("arm", "class_key", "primary_class", "n_attempts",
                                                    "n_target_profile", "target_profile_rate",
                                                    "n_distinct_target_groups",
                                                    "distinct_target_groups_per_attempt")} |
                          {"mismatched_n_target_profile": negative["n_target_profile"],
                           "mismatched_n_attempts": negative["n_attempts"]})
        ax.set_yticks(range(len(requested)), [r["class_key"] + (" *" if not r["primary_class"] else "")
                                            for r in requested])
        ax.invert_yaxis()
        ax.set_xlim(-.035, 1.045)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlabel("Fraction of attempts")
        ax.set_title(title, loc="left", pad=15, fontsize=9)
        ax.text(-.19, 1.065, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=1)
        ax.grid(axis="x", color="#E8E8E8", linewidth=1)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=8.5,
               bbox_to_anchor=(.5, .015))
    fig.subplots_adjust(left=.13, right=.985, bottom=.20, top=.9, wspace=.85)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(args.out / f"class-yield.{suffix}", dpi=350, facecolor="white")
    plt.close(fig)
    with (args.out / "class-yield-source.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    receipt = {"source": str(args.attempts),
               "source_sha256": hashlib.sha256(args.attempts.read_bytes()).hexdigest(),
               "analysis_status": "posthoc_saved_R227_profile_census",
               "n_original_classes": len(export),
               "all_original_classes_visible": True,
               "source_tables": "class-yield-source.csv"}
    (args.out / "class-yield-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
