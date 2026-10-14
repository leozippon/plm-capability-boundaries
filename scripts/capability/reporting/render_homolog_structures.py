#!/usr/bin/env python3
"""CPU-only, sequence-selected PF00595 WT illustrations; no prediction or scoring.

Global affine sequence alignment fixes selection; PyMOL CE supplies structural
correspondences for proper rigid Kabsch fitting of unchanged source coordinates.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from matplotlib.colors import to_rgb

if __package__:
    from . import palette
    from .render_framework_structures import af_residues, camera_axes, digest, map_af
else:
    import palette
    from render_framework_structures import af_residues, camera_axes, digest, map_af

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "manuscript/assets/homologs"
RUNTIME = ROOT / ".pi/tasks/structure-panels-20261004"
CE_PARAMETERS = {"target_state": 1, "mobile_state": 1, "quiet": 1, "guide": 1,
                 "d0": 3.0, "d1": 4.0, "window": 8, "gap_max": 30, "transform": 0}
COHORT = ROOT / "results/R1/external_confirmation_20260924/cohort.json"
PFAM = ROOT / "data/interpro/pfam_residue.tsv"
QUERY = "O14640_PF00595_246"
VERSION = "homolog-illustration-v2-ce"
CAPTION = "illustrative pairwise sequence identity to DVL1; not phylogenetic distances or the corpus-defined remote-cohort test"
# Standard BLOSUM62, canonical amino acids, order ARNDCQEGHILKMFPSTWYV.
LETTERS = "ARNDCQEGHILKMFPSTWYV"
BLOSUM62 = np.array([
    [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
    [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
    [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
    [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
    [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
    [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
    [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
    [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
    [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
    [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
    [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
])
POLICY = {
    "universe": "All retained PF00595 cohort WT records with unique exact full-domain mapping to cached human AF chain A, plus independently annotated Q92997 PF00595 core 250–333. No other candidates.",
    "query": "Whole assayed DVL1 WT domain, 93 residues, chain A246–338, not only annotated core 252–335.",
    "alignment": "Global affine BLOSUM62; gap-open -10, gap-extension -0.5, including terminal gaps; length L gap costs -10 - 0.5*(L-1).",
    "alignment_ties": "First maximum in state order paired, gap in candidate, gap in query; traceback from global endpoint.",
    "identity": "Identical paired residues / all non-gap residue pairs.",
    "coverage": "Non-gap pairs / query length and non-gap pairs / candidate length; both >=0.80.",
    "selection": "Exclude query and exact sequence duplicates; retain smallest SHA256(domain ID) representative per sequence. Required close >=70% nearest 85%; required lower <30% nearest 20%; optional intermediate [30%,70%) nearest 50%. Ties SHA256(domain ID).",
    "empty_band": "Stop with eligibility statistics if required close or lower band is empty; do not relax rules.",
    "independence": "No confidence, structure appearance, experimental fitness or quality rank used in selection. Confidence summaries computed only after selection.",
    "timing": "Explicit reproducible rule for this illustration, not a claim of preregistration before previous confidence inspection.",
    "structure": "PyMOL cmd.cealign directly to query, fixed defaults (guide=1, d0=3, d1=4, window=8, gap_max=30, state=1), transform=0; get_raw_alignment atom indices map through atom IDs to source residues. Proper Kabsch uses all CE CA pairs and original coordinates, with no reflections or additional outlier rejection. All WT atoms preserved up to this rigid transform and PDB rounding. Global-sequence-pair Kabsch RMSD retained separately as a diagnostic; sequence selection is unchanged.",
    "camera": "One query-only PCA orientation and query CA centroid; one union all-atom radius sets shared orthographic scale/clipping. Never independently orient homologs.",
}


def align_sequences(query: str, candidate: str) -> dict:
    """Exact global affine DP; deterministic ties, 1-based paired positions."""
    if not query or not candidate or (set(query + candidate) - set(LETTERS)):
        raise ValueError("Expected nonempty canonical amino-acid sequences")
    n, m = len(query), len(candidate)
    scores = np.full((3, n + 1, m + 1), -np.inf)
    previous = np.zeros((3, n + 1, m + 1), dtype=np.int8)
    scores[0, 0, 0] = 0
    for i in range(1, n + 1):
        scores[1, i, 0] = -10 - .5 * (i - 1)
        previous[1, i, 0] = 0 if i == 1 else 1
    for j in range(1, m + 1):
        scores[2, 0, j] = -10 - .5 * (j - 1)
        previous[2, 0, j] = 0 if j == 1 else 2
    qi, ci = [LETTERS.index(a) for a in query], [LETTERS.index(a) for a in candidate]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            values = (scores[:, i-1, j-1], scores[:, i-1, j] + [-10, -.5, -10],
                      scores[:, i, j-1] + [-10, -10, -.5])
            for state, value in enumerate(values):
                parent = int(np.argmax(value))
                previous[state, i, j] = parent
                scores[state, i, j] = value[parent] + (BLOSUM62[qi[i-1], ci[j-1]] if state == 0 else 0)
    state = int(np.argmax(scores[:, n, m]))
    score = float(scores[state, n, m])
    a, b, pairs = [], [], []
    i, j = n, m
    while i or j:
        parent = int(previous[state, i, j])
        if state == 0:
            a.append(query[i-1]); b.append(candidate[j-1]); pairs.append([i, j])
            i -= 1; j -= 1
        elif state == 1:
            a.append(query[i-1]); b.append("-"); i -= 1
        else:
            a.append("-"); b.append(candidate[j-1]); j -= 1
        state = parent
    pairs.reverse()
    identical = sum(query[i-1] == candidate[j-1] for i, j in pairs)
    return {"aligned_query": "".join(reversed(a)), "aligned_candidate": "".join(reversed(b)),
            "score": score, "pairs": pairs, "pair_count": len(pairs), "identical_pairs": identical,
            "identity": identical / len(pairs) if pairs else 0.,
            "query_coverage": len(pairs) / n, "candidate_coverage": len(pairs) / m}


def id_hash(identifier: str) -> str:
    return hashlib.sha256(identifier.encode()).hexdigest()


def select_candidates(rows: list[dict], query_sequence: str) -> tuple[list[dict], dict]:
    """Sequence metrics alone determine selection, regardless of other row fields."""
    seen = {query_sequence}
    eligible = []
    for row in sorted(rows, key=lambda r: id_hash(r["id"])):
        if row["sequence"] in seen:
            row["exclusion"] = "query or exact sequence duplicate"
            continue
        seen.add(row["sequence"])
        if min(row["query_coverage"], row["candidate_coverage"]) < .8:
            row["exclusion"] = "bidirectional coverage below 0.80"
            continue
        row["exclusion"] = None
        eligible.append(row)
    bands = {"close": [r for r in eligible if r["identity"] >= .7],
             "intermediate": [r for r in eligible if .3 <= r["identity"] < .7],
             "lower": [r for r in eligible if r["identity"] < .3]}
    stats = {"verified_candidates_including_query": len(rows), "eligible_nonquery_unique": len(eligible),
             "bands": {k: len(v) for k, v in bands.items()}}
    if not bands["close"] or not bands["lower"]:
        raise ValueError("Required identity band absent: " + json.dumps(stats, sort_keys=True))
    selected = []
    for band, target in (("close", .85), ("intermediate", .5), ("lower", .2)):
        if bands[band]:
            chosen = min(bands[band], key=lambda r: (abs(r["identity"] - target), id_hash(r["id"])))
            chosen["selection_band"] = band
            selected.append(chosen)
    return selected, stats


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Row-vector convention: aligned = original @ rotation + translation."""
    mobile, reference = np.asarray(mobile, float), np.asarray(reference, float)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3 or len(mobile) < 3:
        raise ValueError("Need >=3 paired three-dimensional coordinates")
    if not np.isfinite(mobile).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite coordinates")
    x, y = mobile - mobile.mean(0), reference - reference.mean(0)
    if min(np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)) < 2:
        raise ValueError("Degenerate CA pairs")
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    translation = reference.mean(0) - mobile.mean(0) @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((mobile @ rotation + translation - reference)**2, axis=1))))
    return rotation, translation, rmsd


def source_receipt(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def collect_candidates() -> tuple[list[dict], dict, dict]:
    domains = json.loads(COHORT.read_text())["domains"]
    rows = [{"id": d["name"], "accession": d["accession"], "sequence": d["wildtype"],
             "sequence_source": str(COHORT.relative_to(ROOT))} for d in domains if d["pfam"] == "PF00595"]
    with PFAM.open() as handle:
        annotation = [line.strip().split("\t") for line in handle if line.startswith("Q92997\t") and line.rstrip().endswith("\tPF00595")]
    if annotation != [["Q92997", "250", "333", "PF00595"]]:
        raise ValueError("Independent DVL3 PF00595 annotation differs")
    rows.append({"id": "Q92997_PF00595_250", "accession": "Q92997", "sequence_source": str(PFAM.relative_to(ROOT)),
                 "annotated_core": [250, 333]})
    maps, verified, excluded = {}, [], []
    for row in rows:
        path = ROOT / f"data/alphafold/AF-{row['accession']}-F1-model_v6.pdb.gz"
        if not path.is_file():
            excluded.append({"id": row["id"], "reason": "cached AFDB file absent", "expected_path": str(path.relative_to(ROOT))})
            continue
        residues, headers = af_residues(path, row["accession"])
        if "annotated_core" in row:
            start, end = row["annotated_core"]
            core = [r for r in residues if start <= r["coordinate_id"] <= end]
            if [r["coordinate_id"] for r in core] != list(range(start, end + 1)):
                raise ValueError("DVL3 annotation is not completely covered by cached coordinates")
            row["sequence"] = "".join(r["residue"] for r in core)
        mapping, _ = map_af(row["sequence"], residues)
        ids = [r["coordinate_id"] for r in mapping.values()]
        if ids != list(range(ids[0], ids[0] + len(ids))) or ids[0] != int(row["id"].rsplit("_", 1)[1]):
            raise ValueError("Exact mapping does not agree with declared domain coordinates")
        row.update({"structure_source": source_receipt(path), "source_headers": headers, "chain": "A",
                    "coordinate_start": ids[0], "coordinate_end": ids[-1], "length": len(ids),
                    "sequence_sha256": hashlib.sha256(row["sequence"].encode()).hexdigest()})
        maps[row["id"]] = mapping
        verified.append(row)
    query = next(r for r in verified if r["id"] == QUERY)
    if query["length"] != 93 or query["coordinate_end"] != 338:
        raise ValueError("Whole assayed DVL1 query must be A246–338")
    for row in verified:
        row.update(align_sequences(query["sequence"], row["sequence"]))
    return verified, maps, {"cohort": source_receipt(COHORT), "pfam_annotation": source_receipt(PFAM),
                            "retained_pfam_records": len(rows)-1, "unverified": excluded}


def aligned_pdb(row: dict, mapping: dict, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Keep every domain ATOM record, atom name, residue number and B factor."""
    ids = {r["coordinate_id"] for r in mapping.values()}
    lines, coordinates = [], []
    with gzip.open(ROOT / row["structure_source"]["path"], "rt") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[21] != "A" or int(line[22:26]) not in ids:
                continue
            xyz = np.array([float(line[a:b]) for a, b in ((30,38), (38,46), (46,54))]) @ rotation + translation
            coordinates.append(xyz)
            lines.append(line[:30] + "".join(f"{v:8.3f}" for v in xyz) + line[54:].rstrip())
    (OUT / f"{row['id']}.pdb").write_text("\n".join(lines) + "\nTER\nEND\n")
    return np.array(coordinates)


def ce_correspondences(selected: list[dict], maps: dict, pymol: str) -> dict:
    """Ask CPU PyMOL only for CE correspondences; do not use its transformed atoms."""
    for row in selected:
        aligned_pdb(row, maps[row["id"]], np.eye(3), np.zeros(3))
    script = RUNTIME / "homolog-ce.py"
    script.write_text('''from pymol import cmd
import json
from pathlib import Path
''' + f"root = Path({str(OUT)!r})\nquery = {QUERY!r}\nnames = {[r['id'] for r in selected[1:]]!r}\nparameters = {CE_PARAMETERS!r}\n" + '''
cmd.reinitialize()
cmd.set("max_threads", 4)
cmd.load(str(root / (query + ".pdb")), "query")
receipt = {"pymol_version": cmd.get_version(), "parameters": parameters, "records": {}}
for name in names:
    cmd.load(str(root / (name + ".pdb")), "mobile")
    result = cmd.cealign("query", "mobile", object="correspondence", **parameters)
    raw = cmd.get_raw_alignment("correspondence")
    atoms = {obj: {a.index: a for a in cmd.get_model(obj).atom} for obj in ("query", "mobile")}
    pairs = []
    for column in raw:
        indices = dict(column)
        assert len(column) == 2 and set(indices) == {"query", "mobile"}
        pair = {}
        for obj, index in indices.items():
            atom = atoms[obj][index]
            assert atom.name == "CA" and atom.chain == "A"
            pair[obj] = {"pymol_index": index, "pdb_atom_id": atom.id, "chain": atom.chain,
                         "coordinate_id": int(atom.resi), "residue_name": atom.resn, "atom_name": atom.name}
        pairs.append(pair)
    assert len(pairs) == result["alignment_length"]
    receipt["records"][name] = {"source_atom_pairs": pairs, "ce_reported_rmsd_angstrom": result["RMSD"]}
    cmd.delete("correspondence")
    cmd.delete("mobile")
print("HOMOLOG_CE_RECEIPT", json.dumps(receipt), flush=True)
cmd.quit()
''')
    process = subprocess.run([pymol, "-cq", str(script)], cwd=OUT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
    (RUNTIME / "homolog-ce.log").write_text(process.stdout)
    lines = [line for line in process.stdout.splitlines() if line.startswith("HOMOLOG_CE_RECEIPT ")]
    if process.returncode or len(lines) != 1 or re.search(r"Error|Traceback|Exception|AssertionError", process.stdout):
        raise RuntimeError("CPU PyMOL CE correspondence failed; see runtime homolog-ce.log")
    return json.loads(lines[0].split(" ", 1)[1])


def render(selected: list[dict], maps: dict, manifest: dict, pymol: str) -> None:
    query_map = maps[QUERY]
    ce = ce_correspondences(selected, maps, pymol)
    manifest["ce_method"] = {"command": "cmd.cealign(query, mobile, object='correspondence', **parameters)",
                             "parameters": ce["parameters"], "pymol_version": ce["pymol_version"],
                             "correspondence": "cmd.get_raw_alignment; object-local atom index -> PDB atom ID and source chain/residue"}
    atoms = []
    for row in selected:
        mapping = maps[row["id"]]
        sequence_pairs = row["pairs"]
        _, _, sequence_rmsd = kabsch(np.array([mapping[j]["ca"] for i, j in sequence_pairs]),
                                     np.array([query_map[i]["ca"] for i, j in sequence_pairs]))
        row["sequence_pair_kabsch_diagnostic"] = {"pair_count": len(sequence_pairs), "rmsd_angstrom": sequence_rmsd,
                                                   "purpose": "Global sequence correspondence diagnostic, not rendered structural fit"}
        correspondence = ce["records"].get(row["id"], {})
        if row["id"] == QUERY:
            pairs = [[i, i] for i in query_map]
            rotation, translation, rmsd = np.eye(3), np.zeros(3), 0.
        else:
            query_positions = {r["coordinate_id"]: i for i, r in query_map.items()}
            mobile_positions = {r["coordinate_id"]: i for i, r in mapping.items()}
            pairs = [[query_positions[p["query"]["coordinate_id"]], mobile_positions[p["mobile"]["coordinate_id"]]]
                     for p in correspondence["source_atom_pairs"]]
            if len({i for i, j in pairs}) != len(pairs) or len({j for i, j in pairs}) != len(pairs):
                raise ValueError("CE correspondence contains repeated residues")
            rotation, translation, rmsd = kabsch(np.array([mapping[j]["ca"] for i, j in pairs]),
                                                 np.array([query_map[i]["ca"] for i, j in pairs]))
        row["rigid_alignment"] = {"method": "query identity transform" if row["id"] == QUERY else "proper Kabsch, all PyMOL CE CA pairs, directly to query",
                                  "pairs": pairs, "pair_count": len(pairs), "rmsd_angstrom": rmsd,
                                  "query_coverage": len(pairs) / len(query_map), "candidate_coverage": len(pairs) / len(mapping),
                                  **correspondence, "rotation": rotation.tolist(), "translation": translation.tolist(),
                                  "convention": "aligned_xyz = original_xyz @ rotation + translation"}
        confidence = [r["ca_plddt"] for r in mapping.values()]
        row["confidence_after_selection"] = {"mean_ca_plddt": float(np.mean(confidence)),
                                              "median_ca_plddt": float(np.median(confidence)),
                                              "min_ca_plddt": min(confidence),
                                              "fraction_ca_plddt_below70": sum(v < 70 for v in confidence) / len(confidence)}
        atoms.append(aligned_pdb(row, mapping, rotation, translation))
    marker = query_map[31]
    if marker["coordinate_id"] != 276 or marker["residue"] != "G" or marker["ca_plddt"] != 70.75:
        raise ValueError("DVL1 G31/G276 confidence provenance differs")
    manifest["dvl1_reference_note"] = {"domain_residue": "G31", "chain_A_residue": "G276", "ca_plddt": 70.75,
                                       "interpretation": "Unchanged predicted WT coordinates; no side-chain mechanism implied."}
    center = np.array([r["ca"] for r in query_map.values()]).mean(0)
    radius = float(np.linalg.norm(np.concatenate(atoms) - center, axis=1).max()) + 5.
    distance = radius / np.tan(np.deg2rad(10))
    view = camera_axes(query_map, []) + [0., 0., -distance] + center.tolist() + [distance-radius, distance+radius, -20.]
    manifest["camera"] = {"pymol_view": view, "union_radius_angstrom_with_5A_margin": radius,
                           "query_center": center.tolist(), "image_size": [1200, 1200]}
    scene = ["reinitialize", "set max_threads, 4", "set orthoscopic, on", "set field_of_view, 20",
             "set antialias, 2", "set ray_shadows, off", "set specular, 0", "set ambient, 0.6",
             "set cartoon_fancy_helices, 1", "set cartoon_flat_sheets, 1", "set cartoon_loop_radius, 0.2",
             "set opaque_background, off", "set ray_opaque_background, off", "bg_color white",
             f"set_color backbone_blue, {list(to_rgb(palette.BACKBONE))}"]
    for row in selected:
        name = row["id"]
        scene += [f"load {name}.pdb, {name}", f'assert cmd.count_atoms("{name} and name CA") == {row["length"]}']
    scene += ["hide everything", "show cartoon, all", "color backbone_blue, all", "disable all",
              f"cmd.set_view({view!r})", 'print("HOMOLOG_PYMOL_VERSION", repr(cmd.get_version()), flush=True)',
              'print("HOMOLOG_VIEW", repr(cmd.get_view()), flush=True)']
    for row in selected:
        name = row["id"]
        scene += ["disable all", f"enable {name}", f"scene {name}, store",
                  f"png {name}.png, width=1200, height=1200, dpi=300, ray=1"]
    scene += ["disable all", f"enable {QUERY}", "save aligned-homologs.pse", "quit"]
    path = OUT / "render.pml"
    path.write_text("\n".join(scene) + "\n")
    process = subprocess.run([pymol, "-cq", path.name], cwd=OUT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
    (RUNTIME / "homolog-render.log").write_text(process.stdout)
    if process.returncode or re.search(r"Error|Traceback|Exception|AssertionError", process.stdout):
        raise RuntimeError("CPU PyMOL rendering failed; see runtime homolog-render.log")
    import ast
    versions = [line for line in process.stdout.splitlines() if line.startswith("HOMOLOG_PYMOL_VERSION ")]
    views = [line for line in process.stdout.splitlines() if line.startswith("HOMOLOG_VIEW ")]
    if len(versions) != 1 or len(views) != 1:
        raise RuntimeError("Missing PyMOL camera/version receipt")
    manifest["pymol_version"] = ast.literal_eval(versions[0].split(" ", 1)[1])
    manifest["camera"]["actual_pymol_view"] = ast.literal_eval(views[0].split(" ", 1)[1])
    if not np.allclose(manifest["camera"]["actual_pymol_view"], view, atol=1e-4):
        raise RuntimeError("PyMOL camera differs from shared specified camera")
    from PIL import Image
    for row in selected:
        with Image.open(OUT / f"{row['id']}.png") as image:
            if image.size != (1200, 1200) or image.mode != "RGBA":
                raise RuntimeError("Unexpected image dimensions or missing alpha")
            alpha = np.array(image.getchannel("A"))
            if alpha.min() != 0 or alpha.max() != 255 or np.any(alpha[[0,-1], :]) or np.any(alpha[:, [0,-1]]):
                raise RuntimeError("Empty, opaque or cropped illustration")


def contact_sheet(selected: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "svg.hashsalt": VERSION})
    fig, axes = plt.subplots(1, len(selected), figsize=(3.4 * len(selected), 4.6))
    for ax, row in zip(axes, selected):
        ax.imshow(plt.imread(OUT / f"{row['id']}.png")); ax.axis("off")
        ax.set_title(row["id"] + "\n" + ("DVL1 query WT" if row["id"] == QUERY else f"{100*row['identity']:.1f}% identity to DVL1"), fontsize=9)
        fit_label = "Query reference" if row["id"] == QUERY else "CE structural fit"
        ax.text(.5, -.02, f"Sequence coverage Q / H: {100*row['query_coverage']:.1f}% / {100*row['candidate_coverage']:.1f}%\n"
                f"{fit_label}: {row['rigid_alignment']['pair_count']} CA; RMSD {row['rigid_alignment']['rmsd_angstrom']:.2f} Å\n"
                f"Mean pLDDT {row['confidence_after_selection']['mean_ca_plddt']:.1f} (post-selection)\n"
                f"AFDB file v6; predicted WT, chain A{row['coordinate_start']}–{row['coordinate_end']}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8, linespacing=1.5)
    fig.text(.5, .05, CAPTION, ha="center", fontsize=9)
    fig.text(.5, .012, "Same rigidly aligned WT backbones, shared camera and scale; no mutant conformations or side-chain mechanisms.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.01, right=.99, top=.87, bottom=.31, wspace=.02)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"contact-sheet.{ext}", dpi=200, facecolor="white", metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pymol", default="/home/lzp/miniconda3/envs/pymolviz/bin/pymol")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # Runtime receipts are not manuscript handoff assets, including previous runs.
    for name in ("render.log", "tests.log", "resource-checks.json", "selection-failure.json"):
        old = OUT / name
        if old.exists():
            shutil.move(str(old), str(RUNTIME / ("homolog-previous-" + name)))
    resources = {"time_utc": datetime.now(timezone.utc).isoformat(), "cpu_count": os.cpu_count(),
                 "free_disk_bytes": shutil.disk_usage(OUT).free, "render_threads": 4,
                 "execution": "CPU only; no network, GPU, prediction, pLM scoring or predictive fitting"}
    (RUNTIME / "homolog-resource-checks.json").write_text(json.dumps(resources, indent=2) + "\n")
    rows, maps, inputs = collect_candidates()
    query = next(r for r in rows if r["id"] == QUERY)
    try:
        homologs, stats = select_candidates(rows, query["sequence"])
    except ValueError as error:
        (RUNTIME / "homolog-selection-failure.json").write_text(json.dumps({"policy": POLICY, "error": str(error)}, indent=2) + "\n")
        raise
    selected = [query, *homologs]
    query["selection_band"] = "query"
    query["exclusion"] = None
    manifest = {"schema": VERSION, "policy": POLICY, "caption": CAPTION, "inputs": inputs,
                "implementation": {"script": source_receipt(Path(__file__)), "framework_helpers": source_receipt(Path(__file__).with_name("render_framework_structures.py")),
                                   "palette": source_receipt(palette.PATH), "python": sys.version, "numpy": np.__version__,
                                   "alignment_version": "homolog-illustration-v1:global-affine-dp", "rigid_version": "homolog-illustration-v1:proper-kabsch"},
                "sequence_metric_fields": "Top-level identity, pairs, pair_count and query/candidate_coverage are GLOBAL SEQUENCE metrics. rigid_alignment contains separate CE structural-fit metrics (query uses identity transform).",
                "method_correction": "Selection unchanged. CE structural correspondence replaces global-sequence correspondence for rendering; original all-sequence-pair Kabsch RMSDs remain in sequence_pair_kabsch_diagnostic.",
                "eligibility": stats, "selected_ids": [r["id"] for r in selected],
                "limitations": ["Small fixed available human PF00595 universe, not a representative evolutionary sampling.",
                                "Identity depends on alignment and domain boundaries; DVL3 is its independently annotated 84-residue core, not a 93-residue assayed domain.",
                                "All source structures are predicted WT AFDB files labelled v6; file version is not a claimed prediction-software revision.",
                                "CE chooses geometric CA correspondences, independently of sequence identity. Kabsch fits all those pairs without further trimming; unmatched residues and every WT atom are retained. This is not mechanistic inference.",
                                "PDB export rounds rigidly transformed coordinates to 0.001 Å; pLDDT is prediction confidence, not measured fitness."]}
    render(selected, maps, manifest, args.pymol)
    contact_sheet(selected)
    (OUT / "sequences.fasta").write_text("".join(f">{r['id']} chain=A:{r['coordinate_start']}-{r['coordinate_end']}\n{r['sequence']}\n" for r in rows))
    manifest["candidates"] = [{k: v for k, v in r.items() if k != "sequence"} for r in rows]
    fields = ["id", "selected", "selection_band", "identity", "query_coverage", "candidate_coverage", "pair_count", "exclusion", "structural_fit_pair_count", "structural_fit_query_coverage", "structural_fit_candidate_coverage", "structural_fit_rmsd_angstrom", "sequence_pair_kabsch_rmsd_angstrom", "mean_ca_plddt", "source", "source_sha256"]
    with (OUT / "selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k, "") for k in fields[:8] if k != "selected"},
                             "selected": row["id"] in manifest["selected_ids"], "exclusion": "" if row["id"] == QUERY else row.get("exclusion", ""),
                             **{"structural_fit_" + k: row.get("rigid_alignment", {}).get(k, "")
                                for k in ("pair_count", "query_coverage", "candidate_coverage", "rmsd_angstrom")},
                             "sequence_pair_kabsch_rmsd_angstrom": row.get("sequence_pair_kabsch_diagnostic", {}).get("rmsd_angstrom", ""),
                             "mean_ca_plddt": row.get("confidence_after_selection", {}).get("mean_ca_plddt", ""),
                             "source": row["structure_source"]["path"], "source_sha256": row["structure_source"]["sha256"]})
    outputs = ["sequences.fasta", "selection.csv", "render.pml", "aligned-homologs.pse", "contact-sheet.svg", "contact-sheet.png"]
    outputs += [f"{r['id']}.{ext}" for r in selected for ext in ("pdb", "png")]
    manifest["outputs"] = {name: source_receipt(OUT / name) for name in outputs}
    (OUT / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"eligibility": stats, "selected": [{k: r[k] for k in ("id", "identity", "query_coverage", "candidate_coverage", "rigid_alignment")} for r in selected]}, indent=2))


if __name__ == "__main__":
    main()
