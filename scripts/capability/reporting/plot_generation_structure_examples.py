#!/usr/bin/env python3
"""Render fixed hash-selected ESMFold examples with an existing PyMOL installation.

The selected cartoons illustrate predictions; they do not estimate a rate or
establish folding/function. Selection never uses predictor scores or appearance.
The editable PML scene, exact source PDBs, hashes and residue confidence accompany
the figure. The scientific inference and cohort selection are never rerun.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

SELECTION_SALT = "main-structure-illustration-v1|"
LABELS = {"zymctrl": "ZymCTRL", "prollama": "ProLLaMA", "progen3-3b": "ProGen3-3B"}
COLORS = ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pymol", default="pymol", help="Existing PyMOL executable; no installation is attempted")
    parser.add_argument("--select-only", action="store_true", help="Freeze selection before raw coordinates arrive")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assets = args.out / "structure-example-assets"
    assets.mkdir(exist_ok=True)
    rows = {}
    sources = {}
    for path in args.predictions:
        data = [json.loads(line) for line in path.read_text().splitlines()]
        summary = json.loads(path.with_name("structure_evidence.json").read_text())
        if summary["index_sha256"] != digest(path) or summary["rows"] != len(data):
            raise ValueError("Prediction index and summary disagree")
        if any(r["structure"]["status"] == "pending" for r in data):
            raise ValueError("Illustrations require completed cohorts")
        for row in data:
            if row["id"] in rows:
                raise ValueError("Duplicate record ID across supplied phases")
            rows[row["id"]] = row
            sources[row["id"]] = path

    selections = []
    for arm in LABELS:
        eligible = [r for r in rows.values() if r["arm"] == arm and r["role"] == "generation"
                    and ((r["primary_class"] and r["condition"] == "requested")
                         or r["condition"] == "unconditioned")]
        if not eligible:
            raise ValueError(f"No eligible generation for {arm}")
        def rank(r):
            return hashlib.sha256((SELECTION_SALT + r["id"]).encode()).hexdigest()
        original = min(eligible, key=rank)
        partners = [r for r in rows.values() if r["role"] == "composition_shuffle" and r["paired_id"] == original["id"]]
        if len(partners) != 1:
            raise ValueError("Each selected original requires exactly one own shuffle")
        shuffle = partners[0]
        if sorted(original["sequence"]) != sorted(shuffle["sequence"]):
            raise ValueError("Pair does not preserve composition")
        selections.append({"arm": arm, "eligible_records": len(eligible), "selection_hash": rank(original),
                           "original_id": original["id"], "shuffle_id": shuffle["id"],
                           "length": original["length"], "class_key": original["class_key"],
                           "source_budget_censored": original.get("source_budget_censored"),
                           "official_compilation_valid": original.get("official_compilation_valid")})
    provenance = {
        "selection_policy": "For each arm, choose the smallest SHA256(salt + record ID) among original generations in the requested primary-class condition, or all unconditioned generations; show that record and its frozen composition shuffle. No predictor score, length filter, native compilation status or appearance is used.",
        "selection_salt": SELECTION_SALT,
        "selections": selections,
        "plot_script_sha256": digest(Path(__file__)),
        "input_sha256": {str(p): digest(p) for p in args.predictions},
        "interpretation": "Prediction illustrations, not rate estimates. No evidence of experimental folding or function. Views are independently oriented and scaled; structures are not aligned.",
        "confidence_colors": {"0–<50": COLORS[0], "50–<70": COLORS[1], "70–<90": COLORS[2], "90–100": COLORS[3]},
    }
    selection_path = assets / "selection.json"
    if selection_path.exists():
        old = json.loads(selection_path.read_text())
        if old["selections"] != selections or old["input_sha256"] != provenance["input_sha256"]:
            raise ValueError("Refusing to replace a frozen illustration selection")
    selection_path.write_text(json.dumps(provenance, indent=2) + "\n")
    if args.select_only:
        print(selection_path)
        return

    scene = ["reinitialize", "set max_threads, 4", "bg_color white", "set orthoscopic, on",
             "set antialias, 2", "set ray_shadows, off", "set specular, 0.15", "set ambient, 0.55",
             "set cartoon_fancy_helices, 1", "set cartoon_flat_sheets, 1", "set cartoon_loop_radius, 0.16",
             "set ray_trace_mode, 1", "set ray_trace_gain, 0.035", "set opaque_background, on",
             'print("PYMOL_VERSION", cmd.get_version(), flush=True)']
    for i, color in enumerate(COLORS):
        rgb = [int(color[j:j+2], 16) / 255 for j in (1, 3, 5)]
        scene.append(f"set_color confidence_{i}, {rgb}")
    object_sources = []
    residue_data = []
    for i, selection in enumerate(selections):
        for role in ("original", "shuffle"):
            row = rows[selection[f"{role}_id"]]
            result = row["structure"]
            if result["status"] != "ok":
                raise ValueError("Fixed selected record has no prediction; retain this fact instead of selecting a replacement")
            pdb = sources[row["id"]].parent / result["object_directory"] / "prediction.pdb"
            if digest(pdb) != result["files_sha256"]["prediction.pdb"]:
                raise ValueError("Source PDB digest differs from its scientific result receipt")
            ca_lines = [line for line in pdb.read_text().splitlines()
                        if line.startswith("ATOM") and line[12:16].strip() == "CA"]
            if [int(line[22:26]) for line in ca_lines] != list(range(1, row["length"] + 1)):
                raise ValueError("Expected one consecutively numbered CA per retained residue")
            if not np.allclose([float(line[60:66]) for line in ca_lines], result["ca_plddt"], atol=.0051, rtol=0):
                raise ValueError("PDB confidence differs from the retained CA confidence")
            filename = f"{i:02d}-{role}"
            shutil.copy2(pdb, assets / f"{filename}.pdb")
            object_sources.append({"panel_row": i, "role": role, "record_id": row["id"],
                                   "sequence_sha256": row["sequence_sha256"], "source_index": str(sources[row["id"]]),
                                   "pdb_sha256": digest(pdb), "pdb_file": f"{filename}.pdb",
                                   "mean_ca_plddt": result["mean_ca_plddt"], "ptm": result["ptm"]})
            for n, value in enumerate(result["ca_plddt"], 1):
                residue_data.append({"arm": selection["arm"], "role": role, "record_id": row["id"],
                                     "residue_index": n, "ca_plddt": value})
            scene.extend(["delete all", f"load {filename}.pdb, prediction", "hide everything", "show cartoon",
                          "color confidence_0, all"])
            # Use full-precision retained CA confidence, avoiding B-factor rounding
            # at bin boundaries and tool-specific numeric selection syntax.
            for color_index, (lower, upper) in enumerate([(50, 70), (70, 90), (90, 101)], 1):
                residues = [str(n) for n, value in enumerate(result["ca_plddt"], 1) if lower <= value < upper]
                if residues:
                    scene.append(f"color confidence_{color_index}, resi {'+'.join(residues)}")
            scene.extend(["orient prediction", "zoom prediction, 3", f"save {filename}.pse",
                          f"png {filename}.png, width=1200, height=1050, dpi=300, ray=1"])
    scene.append("quit")
    (assets / "render-structures.pml").write_text("\n".join(scene) + "\n")
    process = subprocess.run([str(Path(args.pymol).resolve()) if "/" in args.pymol else args.pymol,
                              "-cq", "render-structures.pml"], cwd=assets, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    if any("Error:" in line or "Traceback" in line for line in process.stdout.splitlines()):
        raise RuntimeError("PyMOL reported a rendering error:\n" + process.stdout)
    provenance["objects"] = object_sources
    version_lines = [line for line in process.stdout.splitlines() if line.startswith("PYMOL_VERSION")]
    if len(version_lines) != 1:
        raise ValueError("PyMOL did not provide one unambiguous renderer version receipt")
    provenance["pymol_version"] = version_lines[0].removeprefix("PYMOL_VERSION ")
    provenance["render_scene_sha256"] = digest(assets / "render-structures.pml")
    selection_path.write_text(json.dumps(provenance, indent=2) + "\n")
    with (assets / "residue-confidence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(residue_data[0]))
        writer.writeheader()
        writer.writerows(residue_data)

    for font in font_manager.findSystemFonts():
        if Path(font).name.lower() in {"arial.ttf", "arialbd.ttf", "arial_bold.ttf"}:
            font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(130.7 / 25.4, 5.8))
    grid = fig.add_gridspec(3, 3, width_ratios=(1, 1, 1.5), left=.025, right=.955, top=.88,
                           bottom=.17, hspace=.6, wspace=.42)
    for i, selection in enumerate(selections):
        original = rows[selection["original_id"]]
        shuffle = rows[selection["shuffle_id"]]
        for j, role in enumerate(("original", "shuffle")):
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(plt.imread(assets / f"{i:02d}-{role}.png"))
            ax.axis("off")
            row = original if role == "original" else shuffle
            ax.set_title(f"{'Generated' if role == 'original' else 'Own shuffle'}\nCA-pLDDT: {row['structure']['mean_ca_plddt']:.1f}", fontsize=8, pad=2)
            if j == 0:
                fig.text(.025, grid[i, 0].get_position(fig).y1 + .045,
                         f"{chr(97+i)}  {LABELS[selection['arm']]}",
                         fontsize=9, fontweight="bold", va="bottom")
        ax = fig.add_subplot(grid[i, 2])
        x = np.arange(1, selection["length"] + 1)
        ax.plot(x, original["structure"]["ca_plddt"], color="#0072B2", lw=.85, label="Generated")
        ax.plot(x, shuffle["structure"]["ca_plddt"], color="#D55E00", lw=.85, alpha=.8, label="Own shuffle")
        ax.set(xlim=(1, len(x)), ylim=(0, 100), xlabel="Residue position" if i == 2 else "", ylabel="CA-pLDDT")
        ax.set_yticks([0, 50, 100])
        ax.spines[["top", "right"]].set_visible(False)
        title = f"{selection['length']} residues"
        if selection["source_budget_censored"]:
            title += "\nBudget-censored prefix"
        elif selection["official_compilation_valid"]:
            title += "\nNative compiled"
        ax.set_title(title, fontsize=8)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper center")
    fig.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(COLORS, ["<50", "50–<70", "70–<90", "≥90"])],
               title="Cartoon residue pLDDT", loc="lower center", bbox_to_anchor=(.5, .025),
               ncol=4, frameon=False, fontsize=8, title_fontsize=8)
    fig.text(.5, .991, "Predicted structures and residue confidence", ha="center", va="top", fontsize=10, fontweight="bold")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.out / f"structure-examples.{ext}", dpi=350, facecolor="white")
    plt.close(fig)
    print(json.dumps({"figure": str(args.out / "structure-examples.pdf"), "selection": str(selection_path)}))


if __name__ == "__main__":
    main()
