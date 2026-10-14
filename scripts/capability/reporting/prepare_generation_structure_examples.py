#!/usr/bin/env python3
"""Freeze existing generated/real-fragment pairs; never generate or predict sequences."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_PATH = Path("results/R6/generation_replication_20260927")
POOL_PATH = Path("archive/logs/R6/gate_generative_control/pool")
OUTPUT_PATH = Path("manuscript/assets/generated-fragments")
CAMPAIGN = "replicate_1"
SALT = "manuscript-panel-f-20261004-v1"
ARMS = (
    "galactica-1.3b", "galactica-125m", "galactica-30b", "galactica-6.7b",
    "instructprotein", "progen2-base", "progen2-large", "progen2-medium",
    "progen2-small", "progen2-xlarge", "progen3-112m", "progen3-3b",
    "prollama", "prollama-stage-1", "proteinglm-7b-clm", "protgpt2",
    "protgpt3-1.3b", "rita-xl",
)
STRATA = {"16_128": (16, 128), "129_256": (129, 256), "257_512": (257, 512)}
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_digest(value) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Invalid SHA256 provenance")


def read_jsonl(path: Path):
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"Blank source line: {path}:{line_number}")
            row = json.loads(line)
            require(isinstance(row, dict), f"Non-object source line: {path}:{line_number}")
            yield row, {"path": str(path), "line": line_number, "line_sha256": sha256(line)}


def all_occurrences(donor: str, fragment: str) -> list[int]:
    """All overlapping, zero-based possible offsets, not recovered sampling offsets."""
    starts = []
    start = donor.find(fragment)
    while start >= 0:
        starts.append(start)
        start = donor.find(fragment, start + 1)
    return starts


def protocol() -> dict:
    return {
        "schema": "manuscript-panel-f-inputs-v1", "campaign_path": CAMPAIGN_PATH.as_posix(),
        "campaign": CAMPAIGN, "condition": "unconditioned", "arms": sorted(ARMS),
        "strata_inclusive": STRATA, "salt": SALT,
        "selection": "Minimum SHA256 of UTF-8 salt + NUL + campaign + NUL + cell + NUL + attempt_id in each length stratum; cell is run_manifest.cell.cell (arm__unconditioned).",
        "eligibility": "Existing exact equal-length canonical AA20 pairs with verified source attempt and contiguous donor fragment; no sequence edits. Empty, noncanonical and outside-16-512 pairs are excluded, in that precedence order.",
        "integrity": "Missing cells/strata, unequal lengths, duplicate IDs, or source/donor provenance inconsistencies abort the entire freeze, never trigger substitution.",
        "ignored": "Confidence, profile annotations, appearance and existing prediction results are neither loaded nor selection criteria.",
        "failure_policy": "Keep selected IDs after prediction failure; no replacements.",
        "candidate_universe_hash": "SHA256 of canonical JSON plus LF of eligible candidate identities, sequence hashes and donor provenance, sorted by (campaign,cell,attempt_id); excludes input order, line locations and unused fields.",
    }


def prepare(root: Path) -> dict[str, bytes]:
    """Validate the whole fixed universe and build an in-memory, prediction-free handoff."""
    root = root.resolve()
    campaign_dir = root / CAMPAIGN_PATH
    found = {p.name for p in (campaign_dir / "profiles").glob("replicate_1__*__unconditioned")}
    expected = {f"{CAMPAIGN}__{arm}__unconditioned" for arm in ARMS}
    require(found == expected, "Fixed campaign must contain exactly the 18 declared unconditioned profiles")
    receipt_path = root / POOL_PATH / "pool_receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    reservoir_path = root / POOL_PATH / "reservoir.json"
    require_digest(receipt["reservoir_sha256"])
    require(file_sha256(reservoir_path) == receipt["reservoir_sha256"], "Reservoir digest mismatch")
    reservoir = json.loads(reservoir_path.read_bytes())
    # Index by exact sequence digest and retain stratum membership (duplicates are possible).
    donors = {}
    for stratum, sequences in reservoir["strata"].items():
        for sequence in sequences:
            require(isinstance(sequence, str) and bool(sequence) and set(sequence) <= AA,
                    "Noncanonical reservoir donor")
            key = sha256(sequence.encode())
            if key not in donors:
                donors[key] = (sequence, set())
            require(donors[key][0] == sequence, "Donor digest collision")
            donors[key][1].add(stratum)
    del reservoir

    candidates = []
    exclusions = Counter({"empty": 0, "noncanonical": 0, "outside_length_strata": 0})
    manifests = []
    seen_ids = set()
    campaign_seed = None
    for arm in sorted(ARMS):
        cell = f"{arm}__unconditioned"
        directory = f"{CAMPAIGN}__{cell}"
        manifest_path = campaign_dir / "cells" / directory / "run_manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        require(manifest["campaign"]["id"] == CAMPAIGN, "Manifest campaign mismatch")
        config = manifest["cell"]
        require(config["arm"] == arm and config["condition"] == "unconditioned" and config["cell"] == cell,
                "Manifest cell mismatch")
        seed = manifest["campaign"]["seed"]
        require(type(seed) is int, "Invalid campaign seed")
        if campaign_seed is None:
            campaign_seed = seed
        require(seed == campaign_seed, "Campaign seed mismatch")
        versions = manifest["checkpoint"]["files_sha256"]
        require(isinstance(versions, dict) and bool(versions), "Missing immutable generator version")
        for value in [manifest["code_sha256"], *versions.values()]:
            require_digest(value)
        # Copy only generator version fields, never host paths, environment or arbitrary metadata.
        metadata = {"campaign": {"id": CAMPAIGN, "seed": seed},
                    "cell": {k: config[k] for k in ("arm", "condition", "cell", "attempts")},
                    "manifest_path": manifest_path.relative_to(root).as_posix(),
                    "manifest_bytes_sha256": sha256(manifest_bytes),
                    "checkpoint_files_sha256": versions, "code_sha256": manifest["code_sha256"],
                    "snapshot": manifest["snapshot"], "torch": manifest["torch"]}
        for key in ("upstream_source_sha256", "upstream_tokenizer_sha256"):
            if key in manifest:
                metadata[key] = manifest[key]
        attempts_path = manifest_path.with_name("attempts.jsonl")
        attempts = {}
        for attempt, location in read_jsonl(attempts_path):
            aid = attempt["id"]
            require(isinstance(aid, str) and re.fullmatch(r"ge_[a-zA-Z0-9_-]+", aid) is not None,
                    "Invalid source attempt ID")
            require(aid not in seen_ids, f"Duplicate source ID: {aid}")
            seen_ids.add(aid)
            sequence = attempt["sequence"]
            require(isinstance(sequence, str) and attempt["length"] == len(sequence), "Source length mismatch")
            require(attempt["sequence_sha256"] == sha256(sequence.encode()), "Source sequence digest mismatch")
            require(attempt["campaign"] == CAMPAIGN and attempt["campaign_seed"] == seed
                    and attempt["arm"] == arm and attempt["condition"] == "unconditioned",
                    "Source campaign/cell mismatch")
            location["path"] = attempts_path.relative_to(root).as_posix()
            attempts[aid] = (sequence, location)
        require(len(attempts) == config["attempts"], "Source attempt count mismatch")
        pair_path = campaign_dir / "profiles" / directory / "paired_fragments.jsonl"
        pair_ids = set()
        cell_exclusions = Counter({key: 0 for key in exclusions})
        eligible_count = 0
        for pair, location in read_jsonl(pair_path):
            aid = pair["attempt_id"]
            require(aid not in pair_ids, f"Duplicate source pair ID: {aid}")
            pair_ids.add(aid)
            require(aid in attempts, f"Pair has no source attempt: {aid}")
            generated, fragment = pair["generated"], pair["fragment"]
            require(isinstance(generated, str) and isinstance(fragment, str), "Pair sequences must be strings")
            require(len(generated) == len(fragment), f"Pair length mismatch: {aid}")
            require(generated == attempts[aid][0], f"Pair/source sequence mismatch: {aid}")
            provenance = pair["fragment_provenance"]
            if not generated:
                require(provenance == {"reason": "empty_attempt_retained"}, "Empty pair provenance mismatch")
                cell_exclusions["empty"] += 1
                continue
            require_digest(provenance["donor_sha256"])
            require(type(provenance["seed"]) is int, "Invalid fragment seed")
            donor = donors.get(provenance["donor_sha256"])
            if donor is None:
                raise ValueError(f"Donor SHA missing from reservoir: {aid}")
            donor_sequence, donor_strata = donor
            require(provenance["donor_length"] == len(donor_sequence), f"Donor length mismatch: {aid}")
            require(provenance["donor_stratum"] in donor_strata, f"Donor stratum mismatch: {aid}")
            positions = all_occurrences(donor_sequence, fragment)
            require(bool(positions), f"Fragment is not contiguous in donor: {aid}")
            if not (set(generated) <= AA and set(fragment) <= AA):
                cell_exclusions["noncanonical"] += 1
                continue
            stratum = next((name for name, (lo, hi) in STRATA.items() if lo <= len(generated) <= hi), None)
            if stratum is None:
                cell_exclusions["outside_length_strata"] += 1
                continue
            location["path"] = pair_path.relative_to(root).as_posix()
            identity = {"campaign": CAMPAIGN, "cell": cell, "arm": arm, "attempt_id": aid,
                        "length": len(generated), "stratum": stratum,
                        "generated_sha256": sha256(generated.encode()), "fragment_sha256": sha256(fragment.encode()),
                        "fragment_provenance": {k: provenance[k] for k in ("donor_length", "donor_sha256", "donor_stratum", "seed")}}
            candidates.append({**identity, "generated": generated, "fragment": fragment,
                               "identity": identity, "pair_source": location, "attempt_source": attempts[aid][1],
                               "donor_fragment_starts_0based": positions,
                               "selection_hash": sha256("\0".join((SALT, CAMPAIGN, cell, aid)).encode())})
            eligible_count += 1
        require(pair_ids == set(attempts), "Pair/source ID universe mismatch")
        exclusions.update(cell_exclusions)
        metadata.update({"attempts_path": attempts_path.relative_to(root).as_posix(),
                         "attempts_sha256": file_sha256(attempts_path),
                         "pairs_path": pair_path.relative_to(root).as_posix(), "pairs_sha256": file_sha256(pair_path),
                         "pair_count": len(pair_ids), "eligible_count": eligible_count, "exclusions": dict(cell_exclusions)})
        manifests.append(metadata)

    universe = [r["identity"] for r in sorted(candidates, key=lambda r: (r["campaign"], r["cell"], r["attempt_id"]))]
    csv_buffer = io.StringIO(newline="")
    fields = ["stratum", "rank", "selected", "cell", "attempt_id", "length", "selection_hash", "generated_sha256", "fragment_sha256", "donor_sha256"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    selected = []
    counts = {}
    for stratum in STRATA:
        ranked = sorted((r for r in candidates if r["stratum"] == stratum), key=lambda r: r["selection_hash"])
        require(bool(ranked), f"Empty eligible stratum: {stratum}")
        require(len({r["selection_hash"] for r in ranked}) == len(ranked), "Selection hash collision")
        selected.append(ranked[0])
        counts[stratum] = len(ranked)
        for rank, row in enumerate(ranked, 1):
            writer.writerow({**{k: row[k] for k in fields if k in row}, "rank": rank,
                             "selected": int(rank == 1), "donor_sha256": row["fragment_provenance"]["donor_sha256"]})

    cohort = []
    selections = []
    for row in selected:
        aid = row["attempt_id"]
        source = {"pair": row["pair_source"], "attempt": row["attempt_source"],
                  "manifest_bytes_sha256": next(m["manifest_bytes_sha256"] for m in manifests if m["cell"]["cell"] == row["cell"])}
        donor = {**row["fragment_provenance"], "donor_id": "sha256:" + row["fragment_provenance"]["donor_sha256"],
                 "fragment_starts_0based": row["donor_fragment_starts_0based"],
                 "position_ambiguous": len(row["donor_fragment_starts_0based"]) > 1,
                 "position_semantics": "All exact contiguous occurrences; original sampling offset was not retained.",
                 "accession": None, "accession_status": "not retained in reservoir; sequence-SHA identifier only"}
        selections.append({**row["identity"], "selection_hash": row["selection_hash"], "rank": 1,
                           "generated_id": aid + "__generated", "real_fragment_id": aid + "__real-fragment",
                           "source": source, "donor": donor})
        for role, key, suffix in (("generation", "generated", "generated"), ("real_fragment", "fragment", "real-fragment")):
            cohort.append({"id": aid + "__" + suffix, "pair_id": aid, "attempt_id": aid,
                           "role": role, "arm": row["arm"], "campaign": CAMPAIGN, "cell": row["cell"],
                           "condition": "unconditioned", "length": row["length"], "stratum": row["stratum"],
                           "sequence": row[key], "sequence_sha256": row[key + "_sha256"],
                           "source": source, "fragment_provenance": donor})
    metadata = {"schema": "generation-example-source-provenance-v1", "generators": manifests,
                "pool": {"reservoir_path": reservoir_path.relative_to(root).as_posix(),
                         "reservoir_sha256": receipt["reservoir_sha256"],
                         "receipt_path": receipt_path.relative_to(root).as_posix(),
                         "receipt_bytes_sha256": sha256(receipt_bytes), "receipt": receipt}}
    files = {"candidate-ranks.csv": csv_buffer.getvalue().encode(),
             "cohort.jsonl": b"".join(json_bytes(r) for r in cohort),
             "sequences.fasta": "".join(f">{r['id']}\n{r['sequence']}\n" for r in cohort).encode(),
             "source-provenance.json": json_bytes(metadata)}
    selection = {"protocol": protocol(), "protocol_sha256": sha256(json_bytes(protocol())),
                 "candidate_universe_sha256": sha256(json_bytes(universe)),
                 "source_universe_sha256": sha256(files["source-provenance.json"]),
                 "script_sha256": file_sha256(Path(__file__)), "prediction_status": "pending",
                 "source_pair_count": len(seen_ids), "eligible_pair_count": len(candidates),
                 "eligible_by_stratum": counts, "exclusions": dict(exclusions), "selections": selections,
                 "files_sha256": {name: sha256(payload) for name, payload in files.items()}}
    files["selection.json"] = json_bytes(selection)
    files["README.md"] = (
        "# Frozen generated and real-fragment inputs\n\n"
        "Prediction pending. These six exact existing sequences form three length-matched pairs, one per 16–128, 129–256 and 257–512 residue stratum. No predictions, coordinates or confidence scores are supplied, and no folding or functional conclusions can be drawn.\n\n"
        "Selection uses the fixed replicate_1 campaign and all 18 unconditioned arms, not requested conditions or other replicates. The minimum salted SHA256 in each eligible stratum fixes the pair without consulting confidence, profile annotations, appearance or existing predictions. Failed predictions must remain failures: do not select replacements.\n\n"
        "`selection.json` records the exact hash protocol, selected IDs, exclusion counts and content hashes. `candidate-ranks.csv` lists every eligible pair and within-stratum rank without unselected sequences. `source-provenance.json` retains source-byte hashes, checkpoint file hashes, generator code/snapshot versions and the donor-pool receipt; no model weights or donor corpus are included. The candidate-universe hash is order- and confidence-independent; the separate source-universe hash intentionally detects any raw source changes.\n\n"
        "`cohort.jsonl` is the six-row structure-runner input; `sequences.fasta` contains the same exact sequences. IDs preserve original attempt IDs with generated/real-fragment suffixes. Selected source line numbers are one-based; line SHA256 includes the original line terminator. Donor positions are all zero-based exact substring starts, with ambiguity explicitly recorded, not claimed original sampling offsets. The reservoir retained no accession headers: donor IDs are sequence SHA256 values, not invented accessions. HuggingFace revisions are not inferred; immutable checkpoint file hashes identify generator versions.\n\n"
        "Reproduce from the repository root with `python scripts/capability/reporting/prepare_generation_structure_examples.py` in the validated ct environment. The command performs local CPU input verification only and refuses to replace differing frozen output files.\n"
    ).encode()
    return files


def freeze(files: dict[str, bytes], out: Path) -> None:
    """Never overwrite a differing freeze, including after downstream prediction failure."""
    for name, payload in files.items():
        path = out / name
        require(not path.exists() or path.read_bytes() == payload, f"Refusing to replace frozen input: {path}")
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        path = out / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / OUTPUT_PATH)
    args = parser.parse_args()
    files = prepare(ROOT)
    freeze(files, args.out)
    selection = json.loads(files["selection.json"])
    print(json.dumps({"output": str(args.out), "candidate_universe_sha256": selection["candidate_universe_sha256"],
                      "selected_ids": [r["attempt_id"] for r in selection["selections"]],
                      "prediction_status": "pending"}, indent=2))


if __name__ == "__main__":
    main()
