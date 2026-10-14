#!/usr/bin/env python3
"""Render only pinned UBC9, DVL1 PDZ and mouse 1UFM WT structure illustrations.

Use an existing CPU PyMOL executable (--pymol or PYMOL_EXECUTABLE). No prediction,
sequence design, downloads, refitting, or figure assembly is performed. All output
is restricted to manuscript assets and a minimal public marked-image bundle.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from matplotlib.colors import to_rgb

if __package__:
    from . import palette, publication_paths
else:
    import palette
    import publication_paths

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.interactions.contact_enrichment import (  # noqa: E402
    THREE_TO_ONE, load_structure, read_cif,
)

ASSETS = ROOT / publication_paths.ASSETS
SOURCES = {
    "ubc9": "data/proteingym/DMS_ProteinGym_substitutions/UBC9_HUMAN_Weile_2017.csv",
    "ubc9_metadata": "data/proteingym_raw/DMS_substitutions.csv",
    "ubc9_af": "data/alphafold/AF-P63279-F1-model_v6.pdb.gz",
    "abundance": "results/R1/external_confirmation_20260924/cohort.json",
    "dvl1_af": "data/alphafold/AF-O14640-F1-model_v6.pdb.gz",
    "pairwise": "archive/logs/R3/pairwise_cohort_20260924/cohort.json",
    "annotation": "archive/logs/shared/gate_structure_contact_20260924/contact_annotation.json",
    "cif": "archive/logs/shared/gate_structure_contact_20260924/structures/1UFM.cif.gz",
    "stability": "results/R4/gate_stability_20260924/cohort/cohort.json",
    "clean": "results/R3/pairwise_indel_exclusion_20260928/pairwise_indel_exclusion.json",
}
PAIR_SHA = "8133463ec30293013864685a426a99b69c51c1bac6774b37ebd036915670040f"
CIF_SHA = "fd3793407066da7a58bffc2284a2925883e4f931063084c057f01bf1378b34fd"
SALT = "framework-pinned-median-v1|"
AA = set("ACDEFGHIKLMNPQRSTVWY")
POLICY = {
    "rule": "Nearest cohort median measured scalar (median absolute epsilon for cycles); SHA256(salt + record ID) breaks ties. Never extrema or predicted outcomes.",
    "salt": SALT,
    "universe": "Only retained measured records for the three pinned backgrounds, not their pre-cap eligible pools.",
    "admission": "Exact substitution and sequence identity, finite labels, exact coordinate correspondence and one CA at each marker; cycles require all four supported original/clean states.",
    "geometry_thresholds": None,
    "confidence_thresholds": None,
    "camera": "Coordinate-only PCA; signs fixed by largest axis component and marker centroid, pair direction determines horizontal axis. No measured labels enter camera construction.",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(rows: list[dict], key: str, value: str | int) -> dict:
    matches = [r for r in rows if r.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {key}={value}")
    return matches[0]


def substitution(wt: str, sequence: str, position: int, mutant: str) -> str:
    if not 1 <= position <= len(wt) or mutant not in AA or mutant == wt[position - 1]:
        raise ValueError("Invalid or missing substitution")
    expected = wt[:position - 1] + mutant + wt[position:]
    if sequence != expected:
        raise ValueError("Variant sequence is not the exact declared single substitution")
    return f"{wt[position - 1]}{position}{mutant}"


def reconstruct_wt(rows: list[dict]) -> str:
    references = set()
    for row in rows:
        match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])", row["mutant"])
        if match is None:
            raise ValueError("UBC9 record is not a single substitution")
        old, index, new = match.groups()
        position = int(index)
        sequence = row["mutated_sequence"]
        if not 1 <= position <= len(sequence) or sequence[position - 1] != new:
            raise ValueError("UBC9 mutant and sequence disagree")
        wt = sequence[:position - 1] + old + sequence[position:]
        substitution(wt, sequence, position, new)
        references.add(wt)
    if len(references) != 1:
        raise ValueError("Single-substitution records do not reconstruct one reference")
    return references.pop()


def exact_subsequence(wt: str, entity: str) -> dict[int, int]:
    starts = [i for i in range(len(entity) - len(wt) + 1) if entity.startswith(wt, i)]
    if len(starts) != 1:
        raise ValueError("WT must match the coordinate/entity sequence uniquely and exactly")
    return {i: starts[0] + i for i in range(1, len(wt) + 1)}


def af_residues(path: Path, title: str) -> tuple[list[dict], dict]:
    with gzip.open(path, "rt") as handle:
        lines = handle.read().splitlines()
    headers = {tag: " ".join(line[10:].strip() for line in lines if line.startswith(tag))
               for tag in ("TITLE", "SOURCE")}
    if title not in headers["TITLE"] or "HOMO SAPIENS" not in headers["SOURCE"]:
        raise ValueError("Cached AF structure title/species does not match pinned source")
    residues = []
    for line in lines:
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            if line[16] not in (" ", "A") or line[21] != "A" or line[26] != " ":
                raise ValueError("Unexpected AF chain/alternate/insertion numbering")
            residues.append({"coordinate_id": int(line[22:26]), "author_id": line[22:26].strip(),
                             "residue": THREE_TO_ONE[line[17:20]],
                             "ca": [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))],
                             "ca_plddt": float(line[60:66])})
    ids = [r["coordinate_id"] for r in residues]
    if len(ids) != len(set(ids)) or not residues:
        raise ValueError("Missing or duplicate CA residues in AF")
    return residues, headers


def map_af(wt: str, residues: list[dict], *, allow_absent_terminal: bool = False) -> tuple[dict, list[int]]:
    observed = "".join(r["residue"] for r in residues)
    if allow_absent_terminal:
        # This path is exclusive to UBC9. Numbered coordinates must agree with
        # the independently reconstructed full reference, not a cropped query.
        mapping = {}
        for residue in residues:
            position = residue["coordinate_id"]
            if not 1 <= position <= len(wt) or wt[position - 1] != residue["residue"]:
                raise ValueError("Numbered AF residue disagrees with full experimental reference")
            mapping[position] = residue
        missing = sorted(set(range(1, len(wt) + 1)) - set(mapping))
        if missing != [159] or len(wt) != 159 or len(residues) != 158 or observed != wt[:158]:
            raise ValueError("UBC9 is not the verified exact 1–158 reference prefix with absent terminal 159")
        return mapping, missing
    positions = exact_subsequence(wt, observed)
    mapping = {i: residues[j - 1] for i, j in positions.items()}
    return mapping, []


def median_select(rows: list[dict], key: str, *, absolute: bool = False) -> tuple[dict, dict]:
    eligible = [r for r in rows if math.isfinite(float(r[key]))]
    if not eligible:
        raise ValueError("No exact eligible mapped measured records")
    ids = [r["id"] for r in eligible]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate selected-universe IDs")
    def value(row: dict) -> float:
        return abs(float(row[key])) if absolute else float(row[key])

    median = float(np.median([value(r) for r in eligible]))
    chosen = min(eligible, key=lambda r: (abs(value(r) - median),
                                        hashlib.sha256((SALT + r["id"]).encode()).hexdigest()))
    return chosen, {"qualified_eligible": len(eligible), "median_selection_scalar": median,
                    "absolute_scalar": absolute,
                    "tie_break_sha256": hashlib.sha256((SALT + chosen["id"]).encode()).hexdigest()}


def single_rows(wt: str, variants: list[dict], mapping: dict, scalar: str, prefix: str) -> list[dict]:
    rows = []
    for index, row in enumerate(variants):
        position = int(row["position"])
        identifier = substitution(wt, row["sequence"], position, row["mutant"])
        if position in mapping and math.isfinite(float(row[scalar])):
            rows.append({**row, "id": prefix + ":" + identifier, "substitution": identifier,
                         "source_row_index_zero_based": index, "positions": [position]})
    return rows


def cycle_rows(background: dict, wt: str, mapping: dict, clean: dict) -> list[dict]:
    if clean["cohort_sha256"] != PAIR_SHA:
        raise ValueError("Clean receipt is not bound to the frozen original cohort")
    # Refuse rather than improvise corrected cycle selection for a touched record.
    if background["name"] in clean["accounting"]["backgrounds_touched"] or any(
            r["background"] == background["name"] for r in clean["states"]):
        raise ValueError("1UFM clean-state identity is not supported by this receipt")
    rows = []
    for index, cycle in enumerate(background["cycles"]):
        states = cycle["sequences"]
        if len(states) != 4 or len(set(states)) != 4 or states[0] != wt:
            raise ValueError("Cycle must have four distinct states with the exact WT")
        diffs = [[i for i in range(len(wt)) if s[i] != wt[i]] if len(s) == len(wt) else [] for s in states]
        if not (len(diffs[1]) == len(diffs[2]) == 1 and len(diffs[3]) == 2
                and diffs[1][0] != diffs[2][0] and sorted(diffs[1] + diffs[2]) == diffs[3]):
            raise ValueError("Cycle is not a complete exact substitution rectangle")
        mutations = [substitution(wt, states[k], diffs[k][0] + 1, states[k][diffs[k][0]]) for k in (1, 2)]
        expected = list(wt)
        for k in (1, 2):
            expected[diffs[k][0]] = states[k][diffs[k][0]]
        if "".join(expected) != states[3] or cycle["positions"] != [i + 1 for i in diffs[3]]:
            raise ValueError("Double state/positions disagree with the single states")
        measurements = []
        for sequence in states:
            if sequence not in background["sequences"] or sequence not in background["measurements"]:
                raise ValueError("Missing measured cycle state")
            measurement = background["measurements"][sequence]
            if not measurement["rows"] or not math.isfinite(measurement["value"]):
                raise ValueError("Missing/nonfinite supporting state measurements")
            if not all(math.isfinite(r["value"]) for r in measurement["rows"]) or not math.isclose(
                    measurement["value"], float(np.median([r["value"] for r in measurement["rows"]])), abs_tol=1e-10):
                raise ValueError("State value differs from original supporting-row median")
            measurements.append(measurement)
        values = [m["value"] for m in measurements]
        if not math.isfinite(cycle["epsilon"]) or not math.isclose(
                cycle["epsilon"], values[3] - values[1] - values[2] + values[0], abs_tol=1e-10):
            raise ValueError("Stored epsilon disagrees with its four measured states")
        if all(i in mapping for i in cycle["positions"]):
            rows.append({**cycle, "id": "1UFM.pdb:" + ":".join(mutations), "substitutions": mutations,
                         "source_row_index_zero_based": index, "state_measurements": measurements,
                         "clean_state_identity": "All four states untouched in frozen clean exclusion receipt"})
    return rows


def public_mapping(mapping: dict) -> list[dict]:
    return [{"wt_position": i, **{k: v for k, v in r.items() if k != "ca"}} for i, r in mapping.items()]


def select_sources() -> tuple[dict, dict, list[dict]]:
    paths = {k: ROOT / v for k, v in SOURCES.items()}
    hashes = {k: digest(p) for k, p in paths.items()}
    if hashes["pairwise"] != PAIR_SHA or hashes["cif"] != CIF_SHA:
        raise ValueError("Pinned pairwise or experimental CIF digest differs")
    data = {k: json.loads(paths[k].read_text()) for k in ("abundance", "pairwise", "annotation", "stability", "clean")}
    with paths["ubc9"].open() as handle:
        ubc9 = list(csv.DictReader(handle))
    wt = reconstruct_wt(ubc9)
    with paths["ubc9_metadata"].open() as handle:
        metadata = one(list(csv.DictReader(handle)), "DMS_id", "UBC9_HUMAN_Weile_2017")
    if wt != metadata["target_seq"] or metadata["region_mutated"] != "1-158":
        raise ValueError("Reconstructed UBC9 WT differs from pinned metadata")
    if metadata["raw_DMS_directionality"] != "1" or metadata["selection_assay"] != "Yeast growth":
        raise ValueError("Pinned UBC9 assay directionality differs from its declared endpoint")
    residues, headers = af_residues(paths["ubc9_af"], "SUMO-CONJUGATING ENZYME UBC9")
    umap, absent = map_af(wt, residues, allow_absent_terminal=True)
    urows = []
    for row in ubc9:
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", row["mutant"])
        assert match is not None  # reconstruct_wt already verified every row
        urows.append({**row, "position": int(match[2]), "mutant": match[3], "sequence": row["mutated_sequence"]})
    ur, uc = median_select(single_rows(wt, urows, umap, "DMS_score", "UBC9_HUMAN_Weile_2017"), "DMS_score")
    position = ur["position"]
    start, end = max(1, position - 6), min(len(wt), position + 6)
    segment = {"start": start, "end": end, "wt": wt[start - 1:end], "variant": ur["sequence"][start - 1:end]}

    domain = one(data["abundance"]["domains"], "name", "O14640_PF00595_246")
    if domain["accession"] != "O14640" or domain["pfam"] != "PF00595" or domain["length"] != 93 or len(domain["variants"]) != 256:
        raise ValueError("Pinned DVL1 domain identity/counts changed")
    drs, dh = af_residues(paths["dvl1_af"], "DISHEVELLED HOMOLOG DVL-1")
    dmap, dm = map_af(domain["wildtype"], drs)
    dr, dc = median_select(single_rows(domain["wildtype"], domain["variants"], dmap, "target", domain["name"]), "target")
    if not math.isfinite(dr["uncertainty"]):
        raise ValueError("Selected abundance uncertainty is not finite")

    pair = one(data["pairwise"]["backgrounds"], "name", "1UFM.pdb")
    stability = one(data["stability"]["backgrounds"], "name", "1UFM.pdb")
    annotation = one(data["annotation"]["backgrounds"], "name", "1UFM.pdb")["structure"]
    pwt = pair["cycles"][0]["sequences"][0]
    if pair["length"] != 72 or len(pair["cycles"]) != 128 or len(stability["variants"]) != 256 or stability["sequences"][0] != pwt:
        raise ValueError("Pinned 1UFM WT/counts changed")
    structure = load_structure(paths["cif"])
    with gzip.open(paths["cif"], "rt") as handle:
        cif = read_cif(handle.read())
    title = cif["struct"]["title"][0]
    species = cif["entity_src_gen"]["pdbx_gene_src_scientific_name"]
    if "Solution structure of the PCI domain" not in title or "Mus musculus" not in species:
        raise ValueError("1UFM is not the pinned mouse PCI-domain structure")
    positions = exact_subsequence(pwt, structure["entities"]["1"])
    if structure["entry_id"] != "1UFM" or structure["method"] != "SOLUTION NMR" or len(structure["models"]) != 20:
        raise ValueError("Experimental structure identity/model count differs")
    chain = structure["chains"][(structure["models"][0], "A")]
    pmap = {}
    for i, j in positions.items():
        residue = chain.residues.get(j)
        if residue is None or residue.residue != pwt[i - 1] or residue.named("CA") is None:
            raise ValueError("Missing/mismatched mapped 1UFM CA residue")
        pmap[i] = {"coordinate_id": j, "author_id": residue.auth_seq_id,
                   "residue": residue.residue, "ca": residue.named("CA").tolist()}
    checks = {"entity_id": chain.entity_id, "entity_length": len(structure["entities"]["1"]),
              "models": len(structure["models"]), "observed_residues": len(chain.residues), "auth_asym_id": chain.auth_asym_id,
              "label_asym_id": chain.label_asym_id, "wildtype_to_entity_offset": positions[1] - 1,
              "identity_over_wildtype": 1.0, "identical_columns": len(pmap), "aligned_columns": len(pmap),
              "file_sha256": hashes["cif"]}
    if any(annotation[k] != v for k, v in checks.items()) or checks["wildtype_to_entity_offset"] != 4:
        raise ValueError("Independently parsed 1UFM mapping disagrees with contact annotation")
    sr, sc = median_select(single_rows(pwt, stability["variants"], pmap, "ddg", "1UFM.pdb"), "ddg")
    pr, pc = median_select(cycle_rows(pair, pwt, pmap, data["clean"]), "epsilon", absolute=True)

    def record(row, counts, total, prior, source, scalar, endpoint):
        return {"selected": row, "selection": {**counts, "retained_records": total, "pre_cap_eligible": prior},
                "source": source, "scalar_field": scalar, "endpoint": endpoint}

    records = {
        "ubc9": record(ur, uc, len(ubc9), len(ubc9), SOURCES["ubc9"], "DMS_score",
                       "ProteinGym DMS_score, dimensionless yeast-growth complementation assay score; raw screenscore directionality +1 (higher is greater growth), not an enzyme-activity unit."),
        "dvl1": record(dr, dc, len(domain["variants"]), domain["eligible_variants"], SOURCES["abundance"], "target", data["abundance"]["endpoint"]),
        "stability": record(sr, sc, len(stability["variants"]), stability["eligible_variants"], SOURCES["stability"], "ddg", data["stability"]["endpoint"]),
        "pair": record(pr, pc, len(pair["cycles"]), pair["eligible_cycles"], SOURCES["pairwise"], "epsilon",
                       "epsilon = y_AB - y_A - y_B + y_WT, kcal/mol on combined MegaScale unfolding dG_ML; positive is more stabilising than additive. Absolute epsilon used only for median illustration selection."),
    }
    structures = {
        "ubc9": {"file": SOURCES["ubc9_af"], "accession": "P63279", "chain": "A", "model": 1,
                  "provenance": "Cached AlphaFold WT prediction, not a measured mutant structure", "headers": headers,
                  "residue_map": public_mapping(umap), "absent_wt_positions": absent,
                  "absence_note": "Experimental reference has terminal Y159; AF cached coordinates cover exactly 1–158; Y159 is absent, not assumed unresolved experimentally.",
                  "reference_proof": "All 2563 single-substitution rows reconstruct one full 159-residue WT, identical to ProteinGym target_seq; all 158 numbered coordinate residues agree.",
                  "local_sequence_segment": segment},
        "dvl1": {"file": SOURCES["dvl1_af"], "accession": "O14640", "domain": domain["name"], "chain": "A", "model": 1,
                  "provenance": "Cached AlphaFold WT/domain illustration; abundance is the separate measured PCA scalar, not inferred from fold",
                  "headers": dh, "residue_map": public_mapping(dmap), "absent_wt_positions": dm,
                  "reference_proof": "Unique exact 93-residue domain WT subsequence of all 695 cached coordinate residues; no numeric suffix assumption",
                  "omitted_coordinate_residues": [r["coordinate_id"] for r in drs if r["coordinate_id"] not in {r["coordinate_id"] for r in dmap.values()}]},
        "pci": {"file": SOURCES["cif"], "title": title, "species": species, **checks, "model": chain.model,
                "provenance": "Experimental WT solution NMR, first deposited model only; mouse PCI domain, not UFM1",
                "residue_map": public_mapping(pmap), "absent_wt_positions": [],
                "omitted_entity_positions": sorted(set(chain.residues) - set(positions.values())),
                "reference_proof": "Unique exact 72-residue WT subsequence of entity 1 (84 residues), verified against author numbering and cached annotation"},
    }
    panels = [
        {"name": "ubc9-wt", "structure": "ubc9", "record": "ubc9", "positions": [], "title": "UBC9 reference backbone", "subtitle": "Cached AlphaFold WT; no mutant conformation"},
        {"name": "ubc9-site", "structure": "ubc9", "record": "ubc9", "positions": ur["positions"], "title": "UBC9 measured substitution site", "subtitle": f"{ur['substitution']} | DMS score {float(ur['DMS_score']):.4f}; same WT backbone"},
        {"name": "dvl1-abundance", "structure": "dvl1", "record": "dvl1", "positions": dr["positions"], "title": "DVL1 PDZ abundance illustration", "subtitle": f"{dr['substitution']} | f = {dr['target']:.4f} ± {dr['uncertainty']:.4f} source SE"},
        {"name": "1ufm-stability", "structure": "pci", "record": "stability", "positions": sr["positions"], "title": "Mouse PCI domain: stability site", "subtitle": f"{sr['substitution']} | ΔΔG = {sr['ddg']:.4f} kcal/mol"},
        {"name": "1ufm-pair", "structure": "pci", "record": "pair", "positions": pr["positions"], "title": "Mouse PCI domain: measured pair", "subtitle": f"{' / '.join(pr['substitutions'])} | ε = {pr['epsilon']:.4f} kcal/mol"},
    ]
    manifest = {"schema": "framework-pinned-structures-v1", "policy_declared_before_selection": POLICY,
                "inputs": {k: {"path": SOURCES[k], "sha256": hashes[k]} for k in SOURCES},
                "script_sha256": digest(Path(__file__)),
                "palette_dependency": {"path": str(palette.PATH.relative_to(ROOT)), "sha256": digest(palette.PATH)},
                "records": records, "structures": structures,
                "panels": panels, "clean_exclusion": {"selected_background_touched": False, "receipt": SOURCES["clean"], "cohort_sha256": PAIR_SHA},
                "limitations": ["Illustrative points, not population estimates or new scientific analyses.",
                                "All marker overlays use unchanged WT coordinates; no mutant folding, activity or abundance is depicted structurally.",
                                "Separate assay endpoints are not interchangeable. No geometry/confidence filtering or outcome-dependent camera tuning.",
                                "AF full-length UBC9 lacks reference terminal Y159. The two PCI panels show only the experimentally matched 72-residue WT segment from model 1 of 20."]}
    return manifest, {"ubc9": umap, "dvl1": dmap, "pci": pmap}, panels


def camera_axes(mapping: dict, positions: list[int]) -> list[float]:
    coordinates = np.array([r["ca"] for r in mapping.values()])
    center = coordinates.mean(axis=0)
    _, axes = np.linalg.eigh((coordinates - center).T @ (coordinates - center))
    x, z = axes[:, -1], axes[:, 0]
    if len(positions) == 2:
        x = np.array(mapping[positions[1]]["ca"]) - mapping[positions[0]]["ca"]
        x /= np.linalg.norm(x)
        z = z - x * np.dot(z, x)
        z /= np.linalg.norm(z)
    if x[np.argmax(abs(x))] < 0:
        x = -x
    if z[np.argmax(abs(z))] < 0:
        z = -z
    if positions:
        centroid = np.mean([mapping[p]["ca"] for p in positions], axis=0) - center
        if np.dot(centroid, z) < 0:
            z = -z
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.array([x, y, z]).ravel().tolist()


def render(manifest: dict, maps: dict, panels: list[dict], pymol: str) -> None:
    scenes = ["reinitialize", "set max_threads, 4", "bg_color white", "set orthoscopic, on",
              "set antialias, 2", "set ray_shadows, off", "set specular, 0", "set shininess, 0",
              "set ambient, 0.6", "set cartoon_fancy_helices, 1", "set cartoon_flat_sheets, 1",
              "set cartoon_loop_radius, 0.2", "set opaque_background, off", "set ray_opaque_background, off",
              "set sphere_scale, 1",
              f"set_color framework_blue, {list(to_rgb(palette.BACKBONE))}",
              f"set_color framework_red, {list(to_rgb(palette.SITE))}",
              f"set_color framework_orange, {list(to_rgb(palette.SECOND_SITE))}",
              'print("PYMOL_VERSION", cmd.get_version(), flush=True)']
    marker_unions = {k: sorted({p for panel in panels if panel["structure"] == k for p in panel["positions"]}) for k in maps}
    # Match the PCI camera across overlays: its pair, rather than scalar or best-looking variant, fixes horizontal direction.
    marker_unions["pci"] = manifest["records"]["pair"]["selected"]["positions"]
    for panel in panels:
        key = panel["structure"]
        structure = manifest["structures"][key]
        mapping = maps[key]
        ids = [r["author_id"] for r in mapping.values()]
        if any(not re.fullmatch(r"-?\d+", i) for i in ids):
            raise ValueError("Pinned scene expects unambiguous integer author residue IDs")
        path = (ROOT / structure["file"]).resolve()
        scenes.extend(["delete all", f'load "{path}", source',
                       f"create wt_structure, source and chain A and resi {'+'.join(ids)}, {structure['model']}, 1",
                       "delete source", "hide everything", "show cartoon, wt_structure", "color framework_blue, wt_structure",
                       f'assert cmd.count_atoms("wt_structure and name CA") == {len(mapping)}, "Wrong cropped backbone length"',
                       'assert cmd.count_states("wt_structure") == 1, "Expected only one structural model"'])
        for j, position in enumerate(panel["positions"]):
            rid = mapping[position]["author_id"]
            scenes.extend([f"select marker{j}, wt_structure and chain A and resi {rid} and name CA",
                           f'assert cmd.count_atoms("marker{j}") == 1, "Missing or duplicate marker CA"',
                           f"show spheres, marker{j}", f"set sphere_scale, 1.05, marker{j}",
                           f"color framework_{'red' if j == 0 else 'orange'}, marker{j}"])
        rotation = camera_axes(mapping, marker_unions[key])
        panel["camera_rotation"] = rotation
        scenes.extend(["zoom wt_structure, 5", "python", "view = list(cmd.get_view())",
                       f"view[:9] = {rotation!r}", "cmd.set_view(view)",
                       f'print("FRAMEWORK_VIEW {panel["name"]}", repr(cmd.get_view()), flush=True)', "python end",
                       f"save {panel['name']}.pse",
                       f"png {panel['name']}.png, width=1600, height=1600, dpi=450, ray=1",
                       "hide spheres, wt_structure", "color framework_blue, wt_structure",
                       f"save bare/{panel['name']}.pse",
                       f"png bare/{panel['name']}.png, width=1600, height=1600, dpi=450, ray=1"])
    scenes.append("quit")
    (ASSETS / 'bare').mkdir(exist_ok=True)
    scene = ASSETS / "render-framework.pml"
    scene.write_text("\n".join(scenes) + "\n")
    process = subprocess.run([pymol, "-cq", scene.name], cwd=ASSETS, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (ASSETS / "render.log").write_text(process.stdout)
    if process.returncode or re.search(r"Error|Traceback|Exception|AssertionError", process.stdout):
        raise RuntimeError("PyMOL failed; inspect framework-assets/render.log")
    versions = [line for line in process.stdout.splitlines() if line.startswith("PYMOL_VERSION")]
    if len(versions) != 1:
        raise RuntimeError("Missing unambiguous PyMOL version receipt")
    manifest["pymol_version"] = versions[0].removeprefix("PYMOL_VERSION ")
    manifest["scene_sha256"] = digest(scene)
    import ast
    from PIL import Image
    for panel in panels:
        views = [line for line in process.stdout.splitlines() if line.startswith(f"FRAMEWORK_VIEW {panel['name']} ")]
        if len(views) != 1:
            raise RuntimeError("Missing scene camera receipt")
        panel["pymol_view"] = list(ast.literal_eval(views[0].split(" ", 2)[2]))
        image = ASSETS / f"{panel['name']}.png"
        with Image.open(image) as png:
            png.load()
            if png.size != (1600, 1600) or png.mode != "RGBA" or png.getbbox() is None:
                raise RuntimeError("Empty or incorrectly sized/opaque illustration")
            alpha = np.array(png.getchannel("A"))
            if alpha.min() != 0 or alpha.max() != 255:
                raise RuntimeError("Illustration lacks transparent background or opaque backbone")
        panel["image_sha256"] = digest(image)
        panel['bare_image_sha256'] = digest(ASSETS / 'bare' / f"{panel['name']}.png")
    # Identical view is an invariant of matched overlay pairs, not a visual judgement.
    if panels[0]["pymol_view"] != panels[1]["pymol_view"] or panels[3]["pymol_view"] != panels[4]["pymol_view"]:
        raise RuntimeError("Matched overlays have different camera views")


def annotated_exports(manifest: dict) -> None:
    """Raster WT cartoon with editable SVG mutation/color keys, never mutant folds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans',
                         'svg.hashsalt': 'framework-annotations-v1'})
    output = ASSETS / 'annotated'
    output.mkdir(exist_ok=True)
    for panel in manifest['panels']:
        selected = manifest['records'][panel['record']]['selected']
        mutations = selected.get('substitutions', [selected.get('substitution', '')])
        fig = plt.figure(figsize=(4, 4.8))
        ax = fig.add_axes((0, .16, 1, .80))
        ax.imshow(plt.imread(ASSETS / f"{panel['name']}.png"))
        ax.set_axis_off()
        fig.text(.04, .975, panel['title'], fontsize=9, va='top')
        for index, mutation in enumerate(mutations if panel['positions'] else []):
            x = .06 + .46 * index
            fig.text(x, .12, '\u25cf', color=palette.SITE if index == 0 else palette.SECOND_SITE, fontsize=12)
            fig.text(x + .055, .12, mutation + ' (WT CA site)', fontsize=8)
        fig.text(.04, .065, 'Unchanged WT coordinates; illustrative only', fontsize=8)
        fig.text(.04, .025, 'Cartoon is raster; SVG labels/color keys are editable', fontsize=7)
        panel['annotated_outputs'] = {}
        for extension in ('svg', 'png'):
            path = output / f"{panel['name']}.{extension}"
            fig.savefig(path, dpi=450, facecolor='white', metadata={'Date': None} if extension == 'svg' else None)
            if extension == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
            panel['annotated_outputs'][extension] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
        plt.close(fig)


def bundle_insets(manifest: dict) -> None:
    """Retain only public derived images and minimal provenance, not measurements."""
    import shutil
    output = ROOT / 'scripts/capability/reporting/figure_data/structure-insets'
    output.mkdir(parents=True, exist_ok=True)
    images = {}
    for name in ('dvl1-abundance', '1ufm-stability', '1ufm-pair'):
        panel = one(manifest['panels'], 'name', name)
        source = ASSETS / f'{name}.png'
        if digest(source) != panel['image_sha256']:
            raise ValueError('Marked image differs from final selection manifest')
        shutil.copyfile(source, output / source.name)
        images[name] = {'file': source.name, 'sha256': digest(source),
                        'record_id': manifest['records'][panel['record']]['selected']['id']}
    coordinates = {key: {'accession': structure.get('accession', '1UFM'),
                         'path': structure['file'],
                         'sha256': next(item['sha256'] for item in manifest['inputs'].values()
                                        if item['path'] == structure['file'])}
                   for key, structure in manifest['structures'].items() if key in ('dvl1', 'pci')}
    payload = {'schema': 'manuscript-structure-insets-v1', 'images': images,
               'source_selection': {'path': str((ASSETS / 'selection.json').relative_to(ROOT)),
                                    'sha256': digest(ASSETS / 'selection.json')},
               'coordinate_sources': coordinates, 'provenance': 'unchanged WT; illustrative; marked/no-text images'}
    (output / 'manifest.json').write_text(json.dumps(payload, indent=2) + '\n')


def contact_sheet(manifest: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans", "text.color": "black"})
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    for ax, panel in zip(axes.flat, manifest["panels"]):
        ax.imshow(plt.imread(ASSETS / f"{panel['name']}.png"))
        ax.axis("off")
        ax.set_title(panel["title"] + "\n" + panel["subtitle"], fontsize=9, color="black")
        notes = {
            "ubc9": "Cached AlphaFold WT prediction\nSame backbone; no measured mutant conformation",
            "dvl1": "Cached AlphaFold WT/domain illustration\nAbundance scalar is measured, not inferred from fold",
            "pci": "Experimental WT solution NMR: model 1 of 20\nMouse PCI domain, not UFM1; unchanged WT coordinates",
        }
        ax.text(.5, -.025, notes[panel["structure"]],
                transform=ax.transAxes, ha="center", va="top", fontsize=8)
    axes.flat[-1].axis("off")
    segment = manifest["structures"]["ubc9"]["local_sequence_segment"]
    axes.flat[-1].text(.03, .92, "UBC9 exact local segment (separate asset)\n"
                         f"Reference positions {segment['start']}–{segment['end']}\n"
                         f"WT:      {segment['wt']}\nVariant: {segment['variant']}\n\n"
                         "Red/orange spheres: original WT CA sites.\n"
                         "Cartoons: unchanged reference coordinates.\n"
                         "PCI: NMR model 1 of 20, chain A.\n"
                         "AF: cached WT prediction, not a measurement.\n\n"
                         "Selection: nearest retained-cohort median;\nabsolute epsilon for pair, SHA256 ID tie-break.\n"
                         "Illustrative points only; no population inference.",
                         transform=axes.flat[-1].transAxes, va="top", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(left=.02, right=.98, top=.95, bottom=.07, wspace=.25, hspace=.45)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(ASSETS / f"contact-sheet.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def write_sources(manifest: dict) -> None:
    fields = ["panel", "record_id", "source", "source_sha256", "scalar_field", "scalar", "wt_position", "coordinate_id", "author_id", "wt_residue", "structure_source", "structure_sha256"]
    with (ASSETS / "selected-source.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for panel in manifest["panels"]:
            record = manifest["records"][panel["record"]]
            row = record["selected"]
            structure = manifest["structures"][panel["structure"]]
            for position in panel["positions"]:
                residue = one(structure["residue_map"], "wt_position", position)
                writer.writerow({"panel": panel["name"], "record_id": row["id"], "source": record["source"],
                                 "source_sha256": next(v["sha256"] for v in manifest["inputs"].values() if v["path"] == record["source"]),
                                 "scalar_field": record["scalar_field"], "scalar": row[record["scalar_field"]],
                                 "wt_position": position, "coordinate_id": residue["coordinate_id"], "author_id": residue["author_id"],
                                 "wt_residue": residue["residue"], "structure_source": structure["file"],
                                 "structure_sha256": next(v["sha256"] for v in manifest["inputs"].values() if v["path"] == structure["file"])})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pymol", default=os.environ.get("PYMOL_EXECUTABLE", "pymol"), help="Existing headless PyMOL executable; no installation")
    parser.add_argument("--select-only", action="store_true", help="Validate and freeze pinned records without rendering")
    args = parser.parse_args()
    import shutil
    from datetime import datetime, timezone

    immutable = ROOT / publication_paths.FRAMEWORK_SOURCE
    immutable_sha = digest(immutable) if immutable.exists() else None
    manifest, maps, panels = select_sources()
    ASSETS.mkdir(parents=True, exist_ok=True)
    resources = {"time_utc": datetime.now(timezone.utc).isoformat(), "cpu_count": os.cpu_count(),
                 "free_disk_bytes": shutil.disk_usage(ASSETS).free,
                 "render_threads": 4, "execution": "CPU-only; no GPU operations requested"}
    (ASSETS / "resource-checks.json").write_text(json.dumps(resources, indent=2) + "\n")
    selection = ASSETS / "selection.json"
    if selection.exists():
        old = json.loads(selection.read_text())
        if old["inputs"] != manifest["inputs"] or old["records"] != manifest["records"]:
            raise ValueError("Refusing to replace frozen source identities/selection")
    manifest["resource_policy"] = "CPU-only PyMOL, four threads; no GPU inference, network, installs, bootstrap or refits. Runtime log stored locally in manuscript/assets."
    write_sources(manifest)
    selection.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.select_only:
        render(manifest, maps, panels, args.pymol)
        annotated_exports(manifest)
        contact_sheet(manifest)
        selection.write_text(json.dumps(manifest, indent=2) + "\n")
        bundle_insets(manifest)
    if immutable_sha is not None and digest(immutable) != immutable_sha:
        raise RuntimeError("Immutable framework-original.png changed during execution")
    resources["immutable_Fig1_sha256_before_and_after"] = immutable_sha
    (ASSETS / "resource-checks.json").write_text(json.dumps(resources, indent=2) + "\n")
    print(json.dumps({k: {"id": r["selected"]["id"], "scalar": r["selected"][r["scalar_field"]],
                          "qualified": r["selection"]["qualified_eligible"]} for k, r in manifest["records"].items()}))


if __name__ == "__main__":
    main()
