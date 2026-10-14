"""Small local fixtures for the prediction-free manuscript input freeze."""
import json

import pytest

from src.capability.generation import structure_evidence as se
import scripts.capability.reporting.prepare_generation_structure_examples as examples


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(examples.json_bytes(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(examples.json_bytes(r) for r in rows))


def fixture_campaign(root, lengths=(16, 129, 257)):
    donors = {"16_128": ["A" * 32], "129_256": ["A" * 180], "257_512": ["A" * 320]}
    pool = root / examples.POOL_PATH
    write_json(pool / "reservoir.json", {"strata": donors, "population": {}})
    write_json(pool / "pool_receipt.json", {"reservoir_sha256": examples.file_sha256(pool / "reservoir.json"),
                                           "corpus": "uniref50", "seed": 7})
    for i, arm in enumerate(examples.ARMS):
        cell = arm + "__unconditioned"
        directory = f"replicate_1__{cell}"
        base = root / examples.CAMPAIGN_PATH
        manifest = {"campaign": {"id": "replicate_1", "seed": 123},
                    "cell": {"cell": cell, "arm": arm, "condition": "unconditioned", "attempts": len(lengths)},
                    "checkpoint": {"files_sha256": {"model.safetensors": "a" * 64}, "path": "/private/model"},
                    "code_sha256": "b" * 64, "snapshot": "fixture", "torch": "fixture"}
        write_json(base / "cells" / directory / "run_manifest.json", manifest)
        pairs, attempts = [], []
        for j, length in enumerate(lengths):
            aid = f"ge_{i}_{j}"
            stratum = next(k for k, (lo, hi) in examples.STRATA.items() if lo <= length <= hi)
            donor = donors[stratum][0]
            generated = "C" * length
            pairs.append({"attempt_id": aid, "generated": generated, "fragment": "A" * length,
                          "fragment_provenance": {"donor_length": len(donor), "donor_sha256": examples.sha256(donor.encode()),
                                                  "donor_stratum": stratum, "seed": j}})
            attempts.append({"id": aid, "sequence": generated, "length": length,
                             "sequence_sha256": examples.sha256(generated.encode()), "campaign": "replicate_1",
                             "campaign_seed": 123, "arm": arm, "condition": "unconditioned"})
        write_rows(base / "profiles" / directory / "paired_fragments.jsonl", pairs)
        write_rows(base / "cells" / directory / "attempts.jsonl", attempts)
    return root


def first_path(root, kind, filename):
    return root / examples.CAMPAIGN_PATH / kind / f"replicate_1__{examples.ARMS[0]}__unconditioned" / filename


def test_order_and_confidence_independent_hash_selection(tmp_path):
    fixture_campaign(tmp_path)
    before = examples.prepare(tmp_path)
    for path in (tmp_path / examples.CAMPAIGN_PATH).glob("*/*/*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for i, row in enumerate(rows):
            row.update({"confidence": 100 - i, "profile": {"complete_domain": i % 2 == 0},
                        "structure": {"status": "failed"}, "selected_for_structure": False})
        write_rows(path, list(reversed(rows)))
    after = examples.prepare(tmp_path)
    left, right = (json.loads(bundle["selection.json"]) for bundle in (before, after))
    for field in ("candidate_universe_sha256", "protocol_sha256", "eligible_by_stratum"):
        assert left[field] == right[field]
    assert before["candidate-ranks.csv"] == after["candidate-ranks.csv"]
    assert [r["attempt_id"] for r in left["selections"]] == [r["attempt_id"] for r in right["selections"]]
    assert left["source_universe_sha256"] != right["source_universe_sha256"]
    # Independently compute the declared rank using the manifest campaign/cell values.
    for j, selected in enumerate(left["selections"]):
        expected = min((examples.sha256(f"{examples.SALT}\0replicate_1\0{arm}__unconditioned\0ge_{i}_{j}".encode()),
                        f"ge_{i}_{j}") for i, arm in enumerate(examples.ARMS))
        assert (selected["selection_hash"], selected["attempt_id"]) == expected


@pytest.mark.parametrize("change,error", [
    ("length", "Pair length mismatch"),
    ("source", "Pair/source sequence mismatch"),
    ("donor_sha", "Donor SHA missing"),
    ("donor_length", "Donor length mismatch"),
    ("donor_stratum", "Donor stratum mismatch"),
    ("fragment", "not contiguous"),
    ("duplicate", "Duplicate source pair ID"),
])
def test_pair_integrity_aborts_instead_of_substitution(tmp_path, change, error):
    fixture_campaign(tmp_path)
    path = first_path(tmp_path, "profiles", "paired_fragments.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[0]
    if change == "length":
        row["fragment"] += "A"
    elif change == "source":
        row["generated"] = "D" * len(row["generated"])
    elif change == "donor_sha":
        row["fragment_provenance"]["donor_sha256"] = "f" * 64
    elif change == "donor_length":
        row["fragment_provenance"]["donor_length"] += 1
    elif change == "donor_stratum":
        row["fragment_provenance"]["donor_stratum"] = "257_512"
    elif change == "fragment":
        row["fragment"] = "D" * len(row["fragment"])
    else:
        rows.append(row)
    write_rows(path, rows)
    with pytest.raises(ValueError, match=error):
        examples.prepare(tmp_path)


def test_digest_and_duplicate_source_rejection(tmp_path):
    fixture_campaign(tmp_path)
    path = first_path(tmp_path, "cells", "attempts.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    write_rows(path, rows + rows[:1])
    with pytest.raises(ValueError, match="Duplicate source ID"):
        examples.prepare(tmp_path)
    rows[0]["sequence_sha256"] = "f" * 64
    write_rows(path, rows)
    with pytest.raises(ValueError, match="Source sequence digest mismatch"):
        examples.prepare(tmp_path)
    pool = tmp_path / examples.POOL_PATH / "reservoir.json"
    pool.write_bytes(pool.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Reservoir digest mismatch"):
        examples.prepare(tmp_path)


def test_empty_stratum_fails(tmp_path):
    fixture_campaign(tmp_path, lengths=(16, 129))
    with pytest.raises(ValueError, match="Empty eligible stratum: 257_512"):
        examples.prepare(tmp_path)


def test_exact_six_row_runner_handoff_and_no_invented_accessions(tmp_path):
    fixture_campaign(tmp_path)
    files = examples.prepare(tmp_path)
    out = tmp_path / "out"
    examples.freeze(files, out)
    examples.freeze(files, out)  # An identical rerun is safe.
    cohort = se.load_cohort(out / "cohort.jsonl")
    assert len(cohort) == len({r["id"] for r in cohort}) == 6
    assert {r["role"] for r in cohort} == {"generation", "real_fragment"}
    for a, b in zip(cohort[::2], cohort[1::2]):
        assert a["pair_id"] == b["pair_id"]
        assert a["length"] == b["length"] == len(a["sequence"]) == len(b["sequence"])
        assert a["id"] == a["attempt_id"] + "__generated"
        assert b["id"] == b["attempt_id"] + "__real-fragment"
    for row in cohort:
        donor = row["fragment_provenance"]
        assert donor["accession"] is None
        assert donor["donor_id"] == "sha256:" + donor["donor_sha256"]
        assert donor["position_ambiguous"] is True
        assert donor["fragment_starts_0based"] == list(range(donor["donor_length"] - row["length"] + 1))
        assert row["sequence_sha256"] == examples.sha256(row["sequence"].encode())
        for source in (row["source"]["pair"], row["source"]["attempt"]):
            line = (tmp_path / source["path"]).read_bytes().splitlines(keepends=True)[source["line"] - 1]
            assert examples.sha256(line) == source["line_sha256"]
    assert files["sequences.fasta"] == "".join(f">{r['id']}\n{r['sequence']}\n" for r in cohort).encode()
    provenance = json.loads(files["source-provenance.json"])
    assert "/private/model" not in files["source-provenance.json"].decode()
    for manifest in provenance["generators"]:
        assert manifest["manifest_bytes_sha256"] == examples.file_sha256(tmp_path / manifest["manifest_path"])
        assert manifest["checkpoint_files_sha256"] == {"model.safetensors": "a" * 64}
    selection = json.loads(files["selection.json"])
    assert selection["prediction_status"] == "pending"
    assert selection["eligible_pair_count"] == 54
    for name, digest in selection["files_sha256"].items():
        assert examples.sha256(files[name]) == digest
    with pytest.raises(ValueError, match="Refusing to replace"):
        examples.freeze({**files, "cohort.jsonl": b"different\n"}, out)


def test_exclusion_accounting_and_manifest_mismatch(tmp_path):
    fixture_campaign(tmp_path)
    pair_path = first_path(tmp_path, "profiles", "paired_fragments.jsonl")
    attempts_path = first_path(tmp_path, "cells", "attempts.jsonl")
    pairs = [json.loads(line) for line in pair_path.read_text().splitlines()]
    attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    for i, sequence in enumerate(("", "X" * 129, "C" * 15)):
        pairs[i]["generated"] = attempts[i]["sequence"] = sequence
        attempts[i]["length"] = len(sequence)
        attempts[i]["sequence_sha256"] = examples.sha256(sequence.encode())
        pairs[i]["fragment"] = "A" * len(sequence)
        if not sequence:
            pairs[i]["fragment_provenance"] = {"reason": "empty_attempt_retained"}
    write_rows(pair_path, pairs)
    write_rows(attempts_path, attempts)
    selection = json.loads(examples.prepare(tmp_path)["selection.json"])
    assert selection["exclusions"] == {"empty": 1, "noncanonical": 1, "outside_length_strata": 1}
    assert selection["source_pair_count"] == selection["eligible_pair_count"] + sum(selection["exclusions"].values())
    manifest_path = first_path(tmp_path, "cells", "run_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["cell"]["condition"] = "requested"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Manifest cell mismatch"):
        examples.prepare(tmp_path)
