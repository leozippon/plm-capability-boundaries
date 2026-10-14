#!/usr/bin/env python3
"""CPU-only editing gallery of the six frozen, isolated-sequence ESMFold2 folds.

Run only after the prediction worker finishes. Validate the entire evidence tree
before staging assets; never select, refold, align, crop or modify coordinates.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.capability.core.io import sha256_file
from src.capability.generation import structure_evidence as se
from scripts.capability.reporting import palette
from scripts.capability.reporting.render_framework_structures import camera_axes

ASSETS = ROOT / "manuscript/assets/generated-fragments"
PREDICTIONS = ROOT / "results/R6/manuscript_structures_20261004_l20/predictions"
RUNTIME = ROOT / ".pi/tasks/structure-panels-20261004"
FROZEN_HASHES = {
    "cohort.jsonl": "154c7f80b7e540cbed2d8c9829ac106be2d80ac8962d0c06a61e0f94d11af3fa",
    "selection.json": "3eac60eb08c12b34741f1273c739b3d1651cdbbc65f46e98fed34cc064e1388e",
}
SETTINGS = {"min_length": 16, "max_length": 1024, "num_loops": 3,
            "num_sampling_steps": 14, "num_diffusion_samples": 32,
            "seed": 20260905, "msa": None, "batch_size": 1}
PRECISION = "bfloat16; no autocast; TF32 disabled"
AA3 = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
               "ARNDCQEGHILKMFPSTWYV"))
VERSION = "generated-fragment-gallery-v1"
IMAGE_SIZE = 1200
LIMITATIONS = [
    "Editing reference, not a newly certified submission figure or population-level benchmark.",
    "Prediction confidence is not experimental folding or function; real isolated fragments need not form autonomous domains.",
    "All six preselected sequences and every predicted residue are retained, including the lowest confidence; no replacements.",
    "Independent deterministic PCA cameras, no structural alignment: generated sequences and matched fragments are not homologs. Scale is shared within each pair only.",
    "PDB B factors contain raw 0–1 per-residue/token pLDDT (output.plddt), repeated across residue atoms. Reported CA-pLDDT is the distinct output.plddt_ca on a 0–100 scale from the selected-sample NPZ; the two quantities are not equated.",
    "Adaptive OOM splitting retains 32 samples but changes realized random streams; unequal within-pair splitting is explicitly reported.",
    "Generator upstream revisions/runtime versions absent from retained source provenance are unknown, not inferred from model names.",
]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def receipt(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": sha256_file(path)}


def load_frozen(assets: Path) -> tuple[list[dict], dict, dict, dict]:
    for name, expected in FROZEN_HASHES.items():
        if sha256_file(assets / name) != expected:
            raise ValueError(f"Changed frozen {name}; no reselection or hash override is allowed")
    rows = se.load_cohort(assets / "cohort.jsonl")
    selection = json.loads((assets / "selection.json").read_text())
    provenance = json.loads((assets / "source-provenance.json").read_text())
    if sha256_file(assets / "source-provenance.json") != selection["files_sha256"]["source-provenance.json"]:
        raise ValueError("Changed source provenance")
    plan = json.loads((assets / "prediction-plan.json").read_text())
    frozen = plan["frozen_input"]
    if (frozen["sha256"] != FROZEN_HASHES["cohort.jsonl"] or
            frozen["selection_sha256"] != FROZEN_HASHES["selection.json"] or frozen["sequence_records"] != 6):
        raise ValueError("Prediction plan does not bind the frozen six-row cohort/selection")
    expected_ids = [s[key] for s in selection["selections"] for key in ("generated_id", "real_fragment_id")]
    if len(rows) != 6 or len(selection["selections"]) != 3 or [r["id"] for r in rows] != expected_ids:
        raise ValueError("Exactly the frozen six rows in pair order are required; no row dropping")
    return rows, selection, provenance, plan


def check_contract(worker: dict, plan: dict, cohort_hash: str) -> dict:
    """Narrow gallery-to-plan check, after se.verify_evidence verifies signatures."""
    contract = worker["contract"]
    if (worker.get("cohort_sha256") != cohort_hash or worker.get("shard_index") != 0 or
            worker.get("num_shards") != 1 or worker.get("unique_sequences_assigned") != 6 or
            not worker.get("finished_utc") or "L20" not in worker.get("gpu_name", "")):
        raise ValueError("Expected completed single-shard six-sequence L20 worker bound to frozen cohort")
    expected = {**SETTINGS, "schema_version": se.SCHEMA_VERSION, "model_id": se.PREDICTOR_ID,
                "predictor_kind": se.PREDICTOR_KIND, "never_truncate": True,
                "precision": PRECISION + "; single sequence; no MSA",
                "sample_selection": "highest pTM, matching infer_protein_as_pdb",
                "model_files_sha256": plan["local_checkpoint"]["model_files_sha256"]}
    if plan["local_checkpoint"]["model_id"] != se.PREDICTOR_ID:
        raise ValueError("Prediction plan has a different predictor")
    for key, value in SETTINGS.items():
        if plan["settings"].get(key) != value:
            raise ValueError(f"Prediction plan changed setting {key}")
    if plan["settings"]["precision"] != PRECISION:
        raise ValueError("Prediction plan changed precision")
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Incompatible prediction contract: {key}")
    for key in ("num_loops", "num_sampling_steps", "num_diffusion_samples"):
        if plan["local_checkpoint"][key] != SETTINGS[key]:
            raise ValueError(f"Prediction plan checkpoint setting differs: {key}")
    source = plan["source_sha256"]
    if contract.get("code_sha256") != {
        "module": source["src/capability/generation/structure_evidence.py"],
        "runner": source["scripts/capability/generation/run_structure_evidence.py"],
    }:
        raise ValueError("Predictor code differs from frozen prediction plan")
    versions = plan["local_runtime_import_verified"]
    if contract.get("python") != versions["python"] or any(
            contract.get("versions", {}).get(k) != versions[k] for k in ("torch", "transformers")):
        raise ValueError("Predictor runtime differs from prediction plan")
    return contract


def parse_prediction_pdb(path: Path, sequence: str) -> dict:
    """Read ESMFold2 ATOM records, without AF headers or AF confidence scaling."""
    residues: dict[tuple, dict] = {}
    atoms = []
    model_count = 0
    for line in path.read_text().splitlines():
        if line.startswith("MODEL "):
            model_count += 1
            if model_count > 1:
                raise ValueError("Multiple PDB models are not a selected prediction")
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line.startswith("HETATM") or line[16] != " " or line[26] != " ":
            raise ValueError("Unexpected heteroatom, alternate location or insertion code")
        xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
        confidence = float(line[60:66])
        if not np.isfinite(xyz).all() or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Bad coordinates or raw PDB confidence scale (expected 0–1)")
        key = (line[21], int(line[22:26]))
        name = line[17:20]
        if name not in AA3:
            raise ValueError(f"Noncanonical predicted residue {name}")
        residue = residues.setdefault(key, {"residue": AA3[name], "atoms": set(), "raw_b": confidence})
        if residue["raw_b"] != confidence:
            raise ValueError("Inconsistent per-residue confidence across atoms; output_to_pdb repeats output.plddt[token]")
        atom = line[12:16].strip()
        if atom in residue["atoms"] or residue["residue"] != AA3[name]:
            raise ValueError("Duplicate atom or inconsistent PDB residue")
        residue["atoms"].add(atom)
        if atom == "CA":
            residue.update(ca=xyz)
        atoms.append(xyz)
    if ("".join(r["residue"] for r in residues.values()) != sequence or
            any("ca" not in r for r in residues.values()) or
            len({key[0] for key in residues}) != 1):
        raise ValueError("Predicted CA sequence/length differs from exact isolated sequence")
    numbers = [key[1] for key in residues]
    if numbers != list(range(numbers[0], numbers[0] + len(sequence))):
        raise ValueError("Noncontiguous predicted residue numbering")
    coordinates = np.asarray(atoms, dtype=float)
    if not len(atoms) or np.ptp(coordinates, axis=0).max() <= 0:
        raise ValueError("Empty or degenerate prediction coordinates")
    return {"ca": np.asarray([r["ca"] for r in residues.values()]), "atoms": coordinates,
            "raw_b": np.asarray([r["raw_b"] for r in residues.values()]),
            "residue_numbers": numbers, "chain": next(iter(residues))[0]}


def validate_prediction(row: dict, directory: Path, result: dict) -> tuple[dict, dict]:
    if result.get("status") != "ok":
        raise ValueError(f"Selected prediction {row['id']} is not ok; never replace or drop it")
    if (result.get("length") != len(row["sequence"]) or result.get("evaluator_truncated") is not False or
            result.get("predictor") != se.PREDICTOR_ID):
        raise ValueError("Prediction summary length/truncation/predictor differs")
    if set(result.get("files_sha256", {})) != {"prediction.pdb", "prediction.npz"}:
        raise ValueError("Selected prediction must bind both PDB and NPZ")
    parsed = parse_prediction_pdb(directory / "prediction.pdb", row["sequence"])
    with np.load(directory / "prediction.npz", allow_pickle=False) as arrays:
        ca = np.asarray(arrays["ca_plddt_0_100"], dtype=float)
        pae = np.asarray(arrays["predicted_aligned_error_angstrom"], dtype=float)
        ptm = np.asarray(arrays["ptm"], dtype=float)
        sample = np.asarray(arrays["diffusion_sample_index"])
        length = np.asarray(arrays["length"])
    n = len(row["sequence"])
    if (ca.shape != (n,) or not np.isfinite(ca).all() or not ((0 <= ca) & (ca <= 100)).all() or
            pae.shape != (n, n) or not np.isfinite(pae).all() or (pae < 0).any() or
            ptm.shape != (1,) or not np.isfinite(ptm).all() or not 0 <= ptm[0] <= 1 or
            length.tolist() != [n] or sample.shape != (1,) or not np.isfinite(sample).all() or
            sample[0] != int(sample[0]) or not 0 <= sample[0] < 32):
        raise ValueError("Bad NPZ coordinates/confidence dimensions, units or sample metadata")
    # Transformers output_to_pdb writes output.plddt[token], not output.plddt_ca.
    # PDB integrity, scale and within-residue constancy are checked independently;
    # CA confidence is checked against its own NPZ and summary, never against PDB B.
    mean, fraction = float(ca.mean()), float(np.mean(ca >= 70))
    expected = {"mean_ca_plddt": mean, "fraction_ca_plddt_ge70": fraction,
                "mean_pae_angstrom": float(pae.mean()), "ptm": float(ptm[0]),
                "diffusion_sample_index": int(sample[0])}
    for key, value in expected.items():
        if not np.isclose(result[key], value, rtol=0, atol=1e-5):
            raise ValueError(f"Prediction summary disagrees with NPZ: {key}")
    recorded_ca = np.asarray(result["ca_plddt"], dtype=float)
    if recorded_ca.shape != ca.shape or not np.allclose(recorded_ca, ca, rtol=0, atol=1e-5):
        raise ValueError("Per-residue summary confidence disagrees with NPZ")
    if result["predicted_confidence_event"] != (mean >= 70 and fraction >= .8):
        raise ValueError("Confidence event disagrees with NPZ")
    splits = result.get("diffusion_sample_splits")
    if (not isinstance(splits, list) or not splits or
            any(type(v) is not int or v < 1 for v in splits) or sum(splits) != 32):
        raise ValueError("Missing/invalid realized 32-sample splitting")
    return parsed, {**expected, "min_ca_plddt": float(ca.min()), "median_ca_plddt": float(np.median(ca)),
                    "max_ca_plddt": float(ca.max()), "ca_plddt_0_100": ca.tolist(),
                    "pdb_residue_plddt_raw_0_1": parsed["raw_b"].tolist(), "diffusion_sample_splits": splits,
                    "chunk_stream_seeds": [SETTINGS["seed"] + i for i in range(len(splits))],
                    "adaptive_oom_split": len(splits) > 1}


def pair_views(parsed_pair: list[dict]) -> list[list[float]]:
    centers = [p["ca"].mean(0) for p in parsed_pair]
    radius = max(float(np.linalg.norm(p["atoms"] - center, axis=1).max())
                 for p, center in zip(parsed_pair, centers)) + 5.
    distance = radius / np.tan(np.deg2rad(10))
    return [camera_axes({i: {"ca": xyz} for i, xyz in enumerate(p["ca"])}, []) +
            [0., 0., -distance] + center.tolist() + [distance - radius, distance + radius, -20.]
            for p, center in zip(parsed_pair, centers)]


def collect(assets: Path, predictions: Path) -> tuple[dict, list[dict]]:
    rows, selection, provenance, plan = load_frozen(assets)
    # The runner owns signature, rowset and object-file integrity verification.
    verified = se.verify_evidence(rows, predictions, shard_index=0, num_shards=1)
    workers = sorted(predictions.glob("worker-*.json"))
    if len(workers) != 1 or not (predictions / "structure_evidence.json").is_file():
        raise ValueError("Require one finished worker and final completion receipt")
    worker = json.loads(workers[0].read_text())
    contract = check_contract(worker, plan, FROZEN_HASHES["cohort.jsonl"])
    if verified["rows"] != 6 or verified["status_counts"] != {"ok": 6}:
        raise ValueError("All six selected predictions must be ok; no row dropping")
    entries, geometry = [], []
    for index, row in enumerate(rows):
        directory = predictions / "objects" / row["sequence_sha256"]
        result = se.load_result(directory, verified["evaluation_signature"], row["sequence_sha256"])
        if result is None:
            raise ValueError(f"Missing selected result: {row['id']}")
        parsed, confidence = validate_prediction(row, directory, result)
        stem = f"pair-{index // 2 + 1}-" + ("generated" if index % 2 == 0 else "real-fragment")
        entries.append({**row, "stem": stem, "confidence": confidence,
                        "sequence_source": row.get("source"),
                        "source": {name: receipt(directory / name) for name in
                                   ("result.json", "prediction.pdb", "prediction.npz")},
                        "chain": parsed["chain"], "residue_numbers": parsed["residue_numbers"],
                        "atom_count": len(parsed["atoms"])})
        geometry.append(parsed)
    pairs = []
    for start in range(0, 6, 2):
        for entry, view in zip(entries[start:start+2], pair_views(geometry[start:start+2])):
            entry["pymol_view"] = view
        splits = [e["confidence"]["diffusion_sample_splits"] for e in entries[start:start+2]]
        pairs.append({"pair": start // 2 + 1, "ids": [e["id"] for e in entries[start:start+2]],
                      "diffusion_sample_splits": splits, "unequal_within_pair_splits": splits[0] != splits[1]})
    arms = {r["arm"] for r in rows}
    metadata = {"schema": VERSION, "limitations": LIMITATIONS,
                "inputs": {name: receipt(assets / name) for name in
                           (*FROZEN_HASHES, "source-provenance.json", "prediction-plan.json")},
                "evidence_verification": verified, "worker_receipt": receipt(workers[0]),
                "predictor_contract": contract, "prediction_gpu": worker["gpu_name"],
                "generator_provenance": [g for g in provenance["generators"] if g["cell"]["arm"] in arms],
                "selection_protocol": selection["protocol"], "pairs": pairs,
                "camera_policy": "Independent CA PCA orientation/centroid; shared pair all-atom sphere radius plus 5 Å; orthographic, 20 degree field of view; no alignment or atom transforms.",
                "image_size": [IMAGE_SIZE, IMAGE_SIZE], "backbone_color": palette.BACKBONE,
                "implementation": {"script": receipt(Path(__file__)), "palette": receipt(palette.PATH),
                                   "camera_helper": receipt(Path(__file__).with_name("render_framework_structures.py")),
                                   "python": sys.version, "numpy": np.__version__},
                "records": entries}
    return metadata, entries


def stage_pdbs(entries: list[dict], stage: Path) -> None:
    for entry in entries:
        source = entry["source"]["prediction.pdb"]
        path = Path(source["path"])
        path = path if path.is_absolute() else ROOT / path
        target = stage / (entry["stem"] + ".pdb")
        shutil.copyfile(path, target)
        if sha256_file(target) != source["sha256"]:
            raise ValueError("Prediction changed while staging; original PDB bytes must be preserved")


def render_pml(entries: list[dict]) -> str:
    lines = ["reinitialize", "set max_threads, 4", "set orthoscopic, on", "set field_of_view, 20",
             "set antialias, 2", "set ray_shadows, off", "set specular, 0", "set ambient, 0.6",
             "set cartoon_fancy_helices, 1", "set cartoon_flat_sheets, 1", "set cartoon_loop_radius, 0.2",
             "set opaque_background, off", "set ray_opaque_background, off", "bg_color white",
             f"set_color backbone_blue, {list(to_rgb(palette.BACKBONE))}",
             'print("GENERATED_PYMOL_VERSION", repr(cmd.get_version()), flush=True)']
    for entry in entries:
        name = entry["stem"].replace("-", "_")
        lines += [f"load {entry['stem']}.pdb, {name}",
                  f'assert cmd.count_atoms("{name}") == {entry["atom_count"]}',
                  f'assert cmd.count_atoms("{name} and name CA") == {entry["length"]}']
    lines += ["hide everything", "show cartoon, all", "color backbone_blue, all", "disable all"]
    for entry in entries:
        name = entry["stem"].replace("-", "_")
        lines += ["disable all", f"enable {name}", f"cmd.set_view({entry['pymol_view']!r})",
                  f'print("GENERATED_VIEW {entry["stem"]}", repr(cmd.get_view()), flush=True)',
                  f"scene {name}, store",
                  f"png {entry['stem']}.png, width={IMAGE_SIZE}, height={IMAGE_SIZE}, dpi=300, ray=1"]
    lines += [f"scene {entries[0]['stem'].replace('-', '_')}, recall", "save generated-fragments.pse", "quit"]
    return "\n".join(lines) + "\n"


def render(entries: list[dict], stage: Path, runtime: Path, pymol: str) -> dict:
    from PIL import Image
    (stage / "render.pml").write_text(render_pml(entries))
    process = subprocess.run([pymol, "-cq", "render.pml"], cwd=stage, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LIBGL_ALWAYS_SOFTWARE": "1"})
    (runtime / "generated-render.log").write_text(process.stdout)
    if process.returncode or re.search(r"Error|Traceback|Exception|AssertionError", process.stdout):
        raise RuntimeError("CPU PyMOL failed; see generated-render.log")
    versions = [s.split(" ", 1)[1] for s in process.stdout.splitlines() if s.startswith("GENERATED_PYMOL_VERSION ")]
    views = {s.split(" ", 2)[1]: ast.literal_eval(s.split(" ", 2)[2]) for s in
             process.stdout.splitlines() if s.startswith("GENERATED_VIEW ")}
    if len(versions) != 1 or set(views) != {e["stem"] for e in entries}:
        raise RuntimeError("Missing PyMOL version/camera receipts")
    for entry in entries:
        if not np.allclose(views[entry["stem"]], entry["pymol_view"], atol=1e-4, rtol=0):
            raise RuntimeError("PyMOL camera differs from specified pair scale/view")
        with Image.open(stage / (entry["stem"] + ".png")) as image:
            if image.size != (IMAGE_SIZE, IMAGE_SIZE) or image.mode != "RGBA":
                raise RuntimeError("Unexpected image dimensions or missing transparency")
            alpha = np.asarray(image.getchannel("A"))
            # Thin antialiased backbones can have no fully opaque pixel.
            if alpha.min() != 0 or alpha.max() == 0 or np.any(alpha[[0, -1], :]) or np.any(alpha[:, [0, -1]]):
                raise RuntimeError("Empty, opaque or clipped illustration")
    if not (stage / "generated-fragments.pse").stat().st_size:
        raise RuntimeError("Empty editable PyMOL session")
    return {"pymol_version": ast.literal_eval(versions[0]), "actual_views": views}


def contact_sheet(entries: list[dict], stage: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans", "svg.hashsalt": VERSION}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 11))
        for index, entry in enumerate(entries):
            ax = axes[index % 2, index // 2]
            ax.imshow(plt.imread(stage / (entry["stem"] + ".png")))
            ax.axis("off")
            role = "Generated" if index % 2 == 0 else "Real isolated fragment"
            ax.set_title(f"Pair {index // 2 + 1} · {entry['arm']} · {role}\n{entry['id']}", fontsize=8)
            c = entry["confidence"]
            ax.text(.5, -.02, f"L={entry['length']} · Mean CA-pLDDT {c['mean_ca_plddt']:.1f}/100 · pTM {c['ptm']:.3f}\n"
                    f"biohub/ESMFold2-hf · L20 · samples {c['diffusion_sample_splits']}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8)
        unequal = [str(i // 2 + 1) for i in range(0, 6, 2) if
                   entries[i]["confidence"]["diffusion_sample_splits"] != entries[i+1]["confidence"]["diffusion_sample_splits"]]
        split_note = "Unequal within-pair splitting: " + (", ".join(unequal) if unequal else "none")
        fig.text(.5, .065, "Editing reference only; independent orientations, same scale within each pair; every residue retained.", ha="center", fontsize=9)
        fig.text(.5, .045, "Confidence is not experimental folding/function. Real isolated fragments need not form autonomous domains.", ha="center", fontsize=9)
        fig.text(.5, .025, "OOM splits retain 32 samples but change random streams. " + split_note + ".", ha="center", fontsize=9)
        fig.subplots_adjust(left=.025, right=.975, top=.94, bottom=.14, wspace=.10, hspace=.40)
        for ext in ("svg", "png"):
            fig.savefig(stage / f"contact-sheet.{ext}", dpi=200, facecolor="white",
                        metadata={"Date": None} if ext == "svg" else None)
        plt.close(fig)


def confidence_csv(entries: list[dict], stage: Path) -> None:
    fields = ["id", "arm", "stem", "length", "mean_ca_plddt", "min_ca_plddt", "median_ca_plddt",
              "max_ca_plddt", "fraction_ca_plddt_ge70", "ptm", "mean_pae_angstrom",
              "diffusion_sample_index", "diffusion_sample_splits", "adaptive_oom_split"]
    with (stage / "confidence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for entry in entries:
            values = {**entry, **entry["confidence"]}
            writer.writerow({key: values[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ASSETS, help="Existing directory with byte-identical frozen inputs; also output directory")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--pymol", default="/home/lzp/miniconda3/envs/pymolviz/bin/pymol")
    parser.add_argument("--verify-only", action="store_true", help="Validate completed evidence only; publish nothing")
    args = parser.parse_args()
    args.runtime = args.runtime.resolve()
    args.assets = args.assets.resolve()
    if args.runtime.is_relative_to(args.assets):
        parser.error("Runtime logs/staging must stay outside manuscript assets")
    args.runtime.mkdir(parents=True, exist_ok=True)
    write_json(args.runtime / "generated-resource-checks.json", {
        "time_utc": datetime.now(timezone.utc).isoformat(), "cpu_count": os.cpu_count(),
        "free_disk_bytes": shutil.disk_usage(args.runtime).free, "render_threads": 4,
        "memory": Path("/proc/meminfo").read_text().splitlines()[:3],
        "execution": "CPU only; no inference, network or GPU; original PDB bytes unchanged",
    })
    metadata, entries = collect(args.assets, args.predictions.resolve())
    if args.verify_only:
        print(json.dumps({"verified": 6, "pairs": metadata["pairs"]}, indent=2))
        return
    # Stage outside manuscript: a validation/render failure publishes no new assets.
    with tempfile.TemporaryDirectory(prefix="generated-render-", dir=args.runtime) as directory:
        stage = Path(directory)
        stage_pdbs(entries, stage)
        metadata["render"] = render(entries, stage, args.runtime, args.pymol)
        contact_sheet(entries, stage)
        confidence_csv(entries, stage)
        names = [f"{e['stem']}.{ext}" for e in entries for ext in ("pdb", "png")]
        names += ["generated-fragments.pse", "render.pml", "contact-sheet.svg", "contact-sheet.png", "confidence.csv"]
        metadata["outputs_sha256"] = {name: sha256_file(stage / name) for name in names}
        write_json(stage / "prediction-results.json", metadata)
        # Recheck source/frozen integrity immediately before publishing; metadata last.
        again, _ = collect(args.assets, args.predictions.resolve())
        if again != {k: v for k, v in metadata.items() if k not in {"render", "outputs_sha256"}}:
            raise ValueError("Evidence changed while rendering; refuse to publish")
        for name in [*names, "prediction-results.json"]:
            os.replace(stage / name, args.assets / name)
    print(json.dumps({"published_records": 6, "assets": str(args.assets), "pairs": metadata["pairs"]}, indent=2))


if __name__ == "__main__":
    main()
