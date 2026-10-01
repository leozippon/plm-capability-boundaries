"""Resumable ESMFold2 measurements for the generation cohort.

Every source row survives evaluation. Exact duplicate sequences share a fold,
not a sampling unit. Predictor confidence is indirect structural evidence and
is never labelled an experimental fold, activity, or biological success.
facebook/esmfold_v1 is a retired instrument and is refused.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.io import _atomic_write, sha256_file, write_json

SCHEMA_VERSION = "generation_structure_evidence_v2"
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
PREDICTOR_ID = "biohub/ESMFold2-hf"
PREDICTOR_KIND = "esmfold2"
# Single-sequence, no MSA. The runner keeps the staged checkpoint's own sampling
# settings (biohub/ESMFold2-hf: num_loops=3, inference_num_steps=14,
# num_diffusion_samples=32) unless a flag overrides them.
#
# The evaluation signature digests the measurement contract, which is the
# recorded contract without CODE_PROVENANCE_KEYS. Code-file digests stay in the
# contract as provenance but do not enter the signature: editing a comment in
# this module must not invalidate rows whose measured quantity is unchanged, and
# an operator override of the signature is never an acceptable substitute.
CODE_PROVENANCE_KEYS = ("code_sha256",)
INTERPRETATION = (
    "ESMFold2 confidence is indirect predicted structural feasibility, not "
    "experimentally verified folding or function. The 0-100 CA-pLDDT numbers "
    "and the mean>=70 / 80%-residues event are the same arithmetic as the "
    "retired v1 instrument, not a transferred calibration. Low confidence may "
    "reflect disorder, predictor limitations, or sequence inadequacy. This "
    "evaluator does not identify learned rules or mechanisms."
)


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def measurement_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The contract fields that decide the measured quantity."""

    return {key: value for key, value in contract.items() if key not in CODE_PROVENANCE_KEYS}


def contract_signature(contract: Mapping[str, Any]) -> str:
    return digest_json(measurement_contract(contract))


def manifest_signature(path: Path) -> str:
    """A worker receipt must carry the digest of the contract printed beside it."""

    manifest = json.loads(Path(path).read_text())
    contract, recorded = manifest.get("contract"), manifest.get("evaluation_signature")
    if not isinstance(contract, dict) or not isinstance(recorded, str):
        raise ValueError(f"worker receipt without a contract and signature: {path}")
    expected = contract_signature(contract)
    if recorded != expected:
        raise ValueError(
            f"worker receipt {path} claims evaluation signature {recorded} while its own "
            f"contract digests to {expected}; the receipt does not bind the contract it prints"
        )
    return recorded


def tree_signature(out: Path) -> str:
    """One receipt-bound evaluation signature per evidence tree."""

    manifests = sorted(Path(out).glob("worker-*.json"))
    if not manifests:
        raise FileNotFoundError(f"no worker receipt under {out}")
    signatures = {manifest_signature(path) for path in manifests}
    if len(signatures) != 1:
        raise ValueError(
            f"{out} mixes {len(signatures)} evaluation signatures across {len(manifests)} receipts"
        )
    return signatures.pop()


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def load_cohort(path: Path) -> list[dict[str, Any]]:
    rows = []
    ids = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"line {line_number}: nonempty string id required")
        if row["id"] in ids:
            raise ValueError(f"duplicate source id: {row['id']}")
        if not isinstance(row.get("sequence"), str):
            raise ValueError(f"line {line_number}: sequence must be a string, including empty failures")
        if row.get("sequence_sha256", sequence_digest(row["sequence"])) != sequence_digest(row["sequence"]):
            raise ValueError(f"sequence digest mismatch for {row['id']}")
        ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("cohort has no records")
    return rows


def eligibility(sequence: str, *, min_length: int, max_length: int) -> str | None:
    if not sequence:
        return "empty_sequence"
    if set(sequence) - set(AA_ORDER):
        return "noncanonical_residues"
    if len(sequence) < min_length:
        return "below_minimum_length"
    if len(sequence) > max_length:
        return "above_maximum_length"
    return None


def shard_for(sequence: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return int(sequence_digest(sequence), 16) % num_shards


def diffusion_sample_splits(n_samples: int, n_chunks: int) -> list[int]:
    """Split the diffusion replica axis. Counts still sum to n_samples."""

    if n_samples < 1 or n_chunks < 1:
        raise ValueError("n_samples and n_chunks must be positive")
    n_chunks = min(n_chunks, n_samples)
    quotient, remainder = divmod(n_samples, n_chunks)
    return [quotient + 1] * remainder + [quotient] * (n_chunks - remainder)


def require_esmfold2_checkpoint(model: Path) -> dict[str, Any]:
    """Refuse the retired ESMFold v1 files and any other folding architecture."""

    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"ESMFold2 checkpoint needs {config_path}")
    config = json.loads(config_path.read_text())
    model_type = str(config.get("model_type") or "")
    architectures = [str(item) for item in config.get("architectures") or []]
    retired = (
        "esmfold_config" in config
        or model_type in {"esmfold", "esm"}
        or any(name in {"EsmForProteinFolding", "EsmModel"} for name in architectures)
        or "esmfold_v1" in Path(model).name
    )
    accepted = model_type == "esmfold2" or "EsmFold2Model" in architectures
    if retired or not accepted:
        raise ValueError(
            "facebook/esmfold_v1 is retired; this runner loads only biohub/ESMFold2-hf "
            f"(model_type=esmfold2). Found model_type={model_type!r} architectures={architectures}"
        )
    weights = sorted(Path(model).glob("*.safetensors")) + sorted(Path(model).glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"ESMFold2 checkpoint under {model} has no weight files")
    return config


def checkpoint_file_digests(model: Path) -> dict[str, str]:
    files = [Path(model) / "config.json", *sorted(Path(model).glob("*.safetensors")), *sorted(Path(model).glob("pytorch_model*.bin"))]
    return {path.name: sha256_file(path) for path in files if path.is_file()}


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def summarize_esmfold2(output: Any, *, sequence: str) -> dict[str, Any]:
    """Read CA confidence from ESMFold2 ``plddt_ca`` on the declared 0-1 scale.

    ``infer_protein_as_pdb`` keeps the highest-pTM diffusion sample; this
    function uses the same rule so the PDB and the metrics describe one sample.
    """

    if not sequence:
        raise ValueError("ESMFold2 summary needs the folded sequence")
    ptm = _as_numpy(output.ptm).reshape(-1)
    if ptm.size < 1:
        raise ValueError("ESMFold2 output has no pTM samples")
    sample = int(np.argmax(ptm))
    plddt_ca = _as_numpy(output.plddt_ca)
    if plddt_ca.ndim == 1:
        plddt_ca = plddt_ca[None, :]
    if plddt_ca.ndim != 2:
        raise ValueError("ESMFold2 plddt_ca must be (samples, residues)")
    ca = plddt_ca[sample]
    if ca.shape != (len(sequence),):
        raise ValueError(
            f"ESMFold2 CA pLDDT length {ca.shape[0]} does not match sequence {len(sequence)}"
        )
    if not ((ca >= 0) & (ca <= 1)).all():
        raise ValueError("ESMFold2 pLDDT API changed from [0, 1]")
    ca_100 = ca * 100.0
    pae = _as_numpy(output.pae)
    if pae.ndim == 2:
        pae = pae[None, ...]
    pae = pae[sample]
    if pae.shape != (len(sequence), len(sequence)):
        raise ValueError("ESMFold2 PAE shape does not match sequence length")
    if not np.isfinite(ca_100).all() or not np.isfinite(pae).all() or not np.isfinite(ptm).all():
        raise ValueError("nonfinite ESMFold2 prediction")
    ptm_value = float(ptm[sample])
    if not 0 <= ptm_value <= 1:
        raise ValueError("ESMFold2 pTM outside [0, 1]")
    mean_ca = float(ca_100.mean())
    fraction = float(np.mean(ca_100 >= 70.0))
    return {
        "mean_ca_plddt": mean_ca,
        "fraction_ca_plddt_ge70": fraction,
        "ca_plddt": ca_100.tolist(),
        "predicted_confidence_event": mean_ca >= 70.0 and fraction >= 0.8,
        "ptm": ptm_value,
        "mean_pae_angstrom": float(pae.mean()),
        "diffusion_sample_index": sample,
        "predictor": PREDICTOR_ID,
    }


def esmfold2_arrays(output: Any, *, sequence: str, sample: int) -> dict[str, np.ndarray]:
    """Persist the selected sample's CA pLDDT, PAE and pTM."""

    plddt_ca = _as_numpy(output.plddt_ca)
    if plddt_ca.ndim == 1:
        plddt_ca = plddt_ca[None, :]
    pae = _as_numpy(output.pae)
    if pae.ndim == 2:
        pae = pae[None, ...]
    return {
        "ca_plddt_0_100": plddt_ca[sample] * 100.0,
        "predicted_aligned_error_angstrom": pae[sample],
        "ptm": np.asarray([float(_as_numpy(output.ptm).reshape(-1)[sample])]),
        "diffusion_sample_index": np.asarray([sample]),
        "length": np.asarray([len(sequence)]),
    }


def save_prediction(directory: Path, arrays: Mapping[str, np.ndarray], pdb: str) -> dict[str, str]:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _atomic_write(directory / "prediction.npz", stream.getvalue())
    _atomic_write(directory / "prediction.pdb", pdb.encode())
    return {name: sha256_file(directory / name) for name in ("prediction.npz", "prediction.pdb")}


def load_result(directory: Path, signature: str, sequence_sha256: str) -> dict[str, Any] | None:
    path = directory / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["evaluation_signature"] != signature or result["sequence_sha256"] != sequence_sha256:
        raise ValueError(f"incompatible resumable result: {path}")
    if result["status"] == "ok":
        for name, digest in result["files_sha256"].items():
            if not (directory / name).is_file() or sha256_file(directory / name) != digest:
                raise ValueError(f"incomplete/corrupted prediction: {directory / name}")
    return result


def index_payload(
    rows: Sequence[Mapping[str, Any]], out: Path, signature: str, *,
    shard_index: int, num_shards: int,
) -> tuple[list[dict[str, Any]], bytes]:
    """All rows, including duplicates and errors, remain in the denominator."""
    indexed = []
    for row in rows:
        if shard_for(row["sequence"], num_shards) != shard_index:
            continue
        sha = sequence_digest(row["sequence"])
        directory = out / "objects" / sha
        result = load_result(directory, signature, sha)
        item = dict(row)
        item["structure"] = result or {
            "status": "pending", "sequence_sha256": sha,
            "evaluation_signature": signature,
        }
        item["structure"]["object_directory"] = f"objects/{sha}"
        indexed.append(item)
    payload = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for item in indexed
    ).encode()
    return indexed, payload


def index_summary(indexed: Sequence[Mapping[str, Any]], payload: bytes, signature: str) -> dict[str, Any]:
    """The realized replica chunking is reported, not left inside the rows.

    Chunked and undivided folds draw from different sampling streams, so a
    cohort that mixes them must say so in its own summary.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_signature": signature,
        "rows": len(indexed),
        "status_counts": dict(Counter(item["structure"]["status"] for item in indexed)),
        "unique_sequences": len({item["structure"]["sequence_sha256"] for item in indexed}),
        "diffusion_sample_splits_counts": dict(Counter(
            json.dumps(item["structure"].get("diffusion_sample_splits")) for item in indexed)),
        "interpretation": INTERPRETATION,
        "confidence_event": "mean_CA_pLDDT >= 70 and fraction_CA_pLDDT_ge_70 >= 0.8; operational prediction only",
        "index_sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_index(
    rows: Sequence[Mapping[str, Any]], out: Path, signature: str, *,
    shard_index: int, num_shards: int, filename: str | None = None,
) -> dict[str, Any]:
    indexed, payload = index_payload(rows, out, signature, shard_index=shard_index, num_shards=num_shards)
    name = filename or f"index-{shard_index:03d}-of-{num_shards:03d}.jsonl"
    _atomic_write(out / name, payload)
    summary = index_summary(indexed, payload, signature)
    write_json(out / name.replace(".jsonl", ".summary.json"), summary)
    return summary


RECEIPT_KEYS = ("rows", "unique_sequences", "status_counts", "evaluation_signature", "index_sha256")


def verify_evidence(
    rows: Sequence[Mapping[str, Any]], out: Path, *, shard_index: int, num_shards: int,
) -> dict[str, Any]:
    """Replay an evidence tree against its own receipts, writing nothing.

    Raises on the first disagreement rather than returning a partial pass: a
    signature no receipt binds, a tree that mixes signatures, a stored
    prediction whose bytes do not match the digest its row claims, a row set
    that differs from the cohort's, a row without a terminal result, or an
    index receipt whose digest does not match the replayed index.
    """
    out = Path(out)
    signature = tree_signature(out)
    assigned = {sequence_digest(row["sequence"]) for row in rows
                if shard_for(row["sequence"], num_shards) == shard_index}
    objects = out / "objects"
    stored = {path.name for path in objects.iterdir() if path.is_dir()} if objects.is_dir() else set()
    stray = stored - assigned
    if stray:
        raise ValueError(
            f"{out} holds {len(stray)} of {len(stored)} prediction object directories "
            f"outside this shard's cohort"
        )
    indexed, payload = index_payload(rows, out, signature, shard_index=shard_index, num_shards=num_shards)
    summary = index_summary(indexed, payload, signature)
    pending = summary["status_counts"].get("pending", 0)
    if pending:
        raise ValueError(f"{out} leaves {pending} of {summary['rows']} rows without a terminal result")
    receipts = sorted(out.glob("*.summary.json"))
    if (out / "structure_evidence.json").is_file():
        receipts.append(out / "structure_evidence.json")
    if not receipts:
        raise FileNotFoundError(f"{out} keeps no index receipt to verify the replayed index against")
    for path in receipts:
        recorded = json.loads(path.read_text())
        for key in RECEIPT_KEYS:
            if recorded.get(key) != summary[key]:
                raise ValueError(
                    f"{path} disagrees on {key}: recorded {recorded.get(key)!r}, replayed {summary[key]!r}"
                )
    return summary | {"verified_receipts": [path.name for path in receipts]}
