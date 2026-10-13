"""External family statistics for one wild type: a Potts (DCA) model and its frozen profile.

Every quantity here is a function of the wild type's homolog alignment only; no
phenotype, no generated sequence and no language-model output is read.

Convention (plmc OutputParametersFull, focus mode, gap-ignoring). The model is
``P(a) proportional to exp(E(a))`` with ``E(a) = sum_i h[i, a_i] + sum_{i<j} J[i, j, a_i, a_j]``,
so a larger value is more favourable. ``J[i, j]`` is the ``(20, 20)`` block with the
first axis on residue ``i``; ``J[j, i] = J[i, j].T`` and the diagonal blocks are zero.
The gauge is plmc's own (L2-regularised ``h`` and ``J``, no gauge fixing afterwards): a
bias vector is only meaningful up to an additive constant per position, and the
coupling-only vector is not invariant to moving mass between ``h`` and ``J``. Column
``j`` of every array is residue ``j`` (0-based) of the wild type.

Left-to-right use. Conditioned on a prefix, the next residue is scored by
``h[t] + sum_{s<t} J[t, s, :, a_s]``. This ignores couplings to later positions, so the
ancestral sampler draws from the prefix-conditional chain, which is not the Potts
model's own marginal distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

from ..core.amino_acids import AA20

#: Directory of the exported references. Overridable per call for tests.
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[3] / "data" / "pairwise_assets" / "family_references_20261013"
SCHEMA = "family_potts_reference_v1"
#: Independent-site pseudocount, the value the archived independent-site baseline declared.
PROFILE_PSEUDOCOUNT = 0.5
SIGN_CONVENTION = ("Larger is more favourable: P(a) is proportional to exp(sum_i h[i,a_i] + sum_{i<j} J[i,j,a_i,a_j]); "
                   "J[i,j] has the first axis on residue i and J[j,i] is its transpose.")
GAUGE_CONVENTION = ("plmc L2-regularised gauge, unmodified; natural-log units, unit inverse temperature; "
                    "per-position biases are meaningful up to an additive constant per position.")
_ENCODE = np.full(256, -1, dtype=np.int64)
for _code, _residue in enumerate(AA20):
    _ENCODE[ord(_residue)] = _code


@dataclass(frozen=True)
class FamilyModel:
    domain: str
    wildtype: str
    alphabet: str
    h: np.ndarray  # (L, 20)
    J: np.ndarray  # (L, L, 20, 20)
    profile_freq: np.ndarray  # (L, 20) weighted column frequencies of the frozen profile
    profile_column_weight: np.ndarray  # (L,)
    log_profile: np.ndarray  # (L, 20) log f_t(a) with the pseudocount applied
    column_to_wt: np.ndarray  # (L,) wild-type residue index of each parameter column
    qualified: bool
    qualification_failures: tuple[str, ...]
    sidecar: dict

    @property
    def length(self) -> int:
        return len(self.wildtype)


def encode(sequence: str) -> np.ndarray:
    """Integer codes of a canonical-residue string; raises on any other symbol."""
    codes = _ENCODE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    if (codes < 0).any():
        raise ValueError("sequence contains a symbol outside the 20 canonical residues")
    return codes


def log_profile(freq: np.ndarray, column_weight: np.ndarray) -> np.ndarray:
    """``log f_t(a)`` with ``f = (weighted count + 0.5) / (column weight + 20 * 0.5)``, renormalised."""
    p = freq.astype(np.float64) * column_weight.astype(np.float64)[:, None] + PROFILE_PSEUDOCOUNT
    return np.log(p / p.sum(axis=1, keepdims=True))


def dense_couplings(pairs: np.ndarray, length: int) -> np.ndarray:
    """plmc upper-triangle blocks ``(L(L-1)/2, 20, 20)`` as a symmetric ``(L, L, 20, 20)`` tensor."""
    q = pairs.shape[1]
    if pairs.shape != (length * (length - 1) // 2, q, q):
        raise ValueError("coupling block count does not match the focus length")
    J = np.zeros((length, length, q, q), dtype=np.float64)
    i, j = np.triu_indices(length, k=1)  # row-major, the order plmc writes
    J[i, j] = pairs
    J[j, i] = pairs.transpose(0, 2, 1)
    return J


def potts_score(model: FamilyModel, sequence: str) -> float:
    """``E(a)``: positive unnormalized log weight of a full-length sequence."""
    codes = encode(sequence)
    if len(codes) != model.length:
        raise ValueError("sequence length differs from the model length")
    idx = np.arange(model.length)
    pair = model.J[idx[:, None], idx[None, :], codes[:, None], codes[None, :]]
    return float(model.h[idx, codes].sum() + np.triu(pair, k=1).sum())


def guidance_biases(model: FamilyModel, prefix: str) -> dict[str, np.ndarray]:
    """The three per-residue bias vectors for position ``t = len(prefix)``, in AA20 order.

    ``marginal`` is ``log f_t(a)`` of the frozen profile, ``coupling`` is
    ``sum_{s<t} J[t, s, a, a_s]`` and ``full`` is ``h[t, a]`` plus that. All are in
    natural-log units; ``coupling`` is exactly zero at ``t = 0``.
    """
    t = len(prefix)
    if t >= model.length:
        raise ValueError("prefix already covers the whole sequence")
    codes = encode(prefix)
    rows = model.J[t][np.arange(t), :, codes]  # (t, 20): J[t, s, a, a_s]
    coupling = rows.sum(axis=0)
    full = np.vstack([model.h[t][None], rows]).sum(axis=0)
    if not np.allclose(full, model.h[t] + coupling, rtol=0.0, atol=1e-9):
        raise RuntimeError("full bias differs from fields plus coupling-only bias")
    return {"marginal": model.log_profile[t].copy(), "coupling": coupling, "full": full}


def sample_potts(model: FamilyModel, rng: np.random.Generator) -> str:
    """Draw a length-L sequence left to right from the Potts prefix conditional alone."""
    chosen = ""
    for _ in range(model.length):
        logits = guidance_biases(model, chosen)["full"]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        chosen += model.alphabet[int(rng.choice(len(p), p=p))]
    return chosen


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_family_model(directory: Path, domain: str, *, wildtype: str, h: np.ndarray, pairs: np.ndarray,
                      profile_freq: np.ndarray, profile_column_weight: np.ndarray,
                      qualified: bool, qualification_failures: list[str], extra: dict) -> dict[str, Path]:
    """Write ``<domain>.npz`` and ``<domain>.json``; parameters are stored as the float32 plmc wrote them."""
    length = len(wildtype)
    for name, array, shape in (("h", h, (length, 20)), ("profile_freq", profile_freq, (length, 20)),
                               ("profile_column_weight", profile_column_weight, (length,))):
        if array.shape != shape:
            raise ValueError(f"{domain}: {name} has shape {array.shape}, wild type length requires {shape}")
    J = dense_couplings(pairs.astype(np.float64), length)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npz, side = directory / f"{domain}.npz", directory / f"{domain}.json"
    np.savez_compressed(npz, h=h.astype(np.float32), J=J.astype(np.float32),
                        profile_freq=profile_freq.astype(np.float32),
                        profile_column_weight=profile_column_weight.astype(np.float32))
    sidecar = {"schema": SCHEMA, "domain": domain, "wildtype": wildtype, "length": length,
               "alphabet": AA20, "column_to_wt_index": list(range(length)),
               "column_to_wt_index_base": 0,
               "sign_convention": SIGN_CONVENTION, "gauge_convention": GAUGE_CONVENTION,
               "profile": f"frozen independent-site profile, pseudocount {PROFILE_PSEUDOCOUNT} per residue",
               "qualified": bool(qualified), "qualification_failures": list(qualification_failures),
               "npz_sha256": sha256_file(npz), **extra}
    side.write_text(json.dumps(sidecar, indent=1) + "\n")
    return {"npz": npz, "json": side}


def load_family_model(domain: str, directory: Path = DEFAULT_DIRECTORY, *, allow_unqualified: bool = False) -> FamilyModel:
    """Load and validate one exported reference; any inconsistency raises ``ValueError``/``FileNotFoundError``.

    An unqualified reference is refused unless the caller opts in with ``allow_unqualified=True``.
    """
    directory = Path(directory)
    npz, side = directory / f"{domain}.npz", directory / f"{domain}.json"
    for path in (npz, side):
        if not path.is_file():
            raise FileNotFoundError(f"family reference artifact missing: {path.name}")
    sidecar = json.loads(side.read_text())
    if sidecar.get("schema") != SCHEMA or sidecar.get("domain") != domain:
        raise ValueError(f"{domain}: sidecar schema or domain mismatch")
    if sha256_file(npz) != sidecar.get("npz_sha256"):
        raise ValueError(f"{domain}: npz digest differs from the sidecar")
    wildtype, alphabet = sidecar["wildtype"], sidecar["alphabet"]
    length = len(wildtype)
    if alphabet != AA20 or sidecar["length"] != length:
        raise ValueError(f"{domain}: alphabet or length declaration inconsistent")
    if sidecar["column_to_wt_index"] != list(range(length)):
        raise ValueError(f"{domain}: columns do not map one-to-one onto wild-type residues")
    with np.load(npz) as data:
        missing = {"h", "J", "profile_freq", "profile_column_weight"} - set(data.files)
        if missing:
            raise ValueError(f"{domain}: npz lacks {sorted(missing)}")
        h, J = data["h"].astype(np.float64), data["J"].astype(np.float64)
        freq, weight = data["profile_freq"], data["profile_column_weight"]
    shapes = {"h": (h, (length, 20)), "J": (J, (length, length, 20, 20)),
              "profile_freq": (freq, (length, 20)), "profile_column_weight": (weight, (length,))}
    for name, (array, shape) in shapes.items():
        if array.shape != shape:
            raise ValueError(f"{domain}: {name} has shape {array.shape}, expected {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{domain}: {name} is not finite")
    if not np.array_equal(J, J.transpose(1, 0, 3, 2)) or any(J[i, i].any() for i in range(length)):
        raise ValueError(f"{domain}: couplings are not symmetric with zero diagonal blocks")
    encode(wildtype)
    if not sidecar["qualified"] and not allow_unqualified:
        raise ValueError(f"{domain}: reference is not qualified ({'; '.join(sidecar['qualification_failures'])}); "
                         "pass allow_unqualified=True to load it anyway")
    return FamilyModel(domain, wildtype, alphabet, h, J, freq, weight, log_profile(freq, weight),
                       np.arange(length), bool(sidecar["qualified"]),
                       tuple(sidecar["qualification_failures"]), sidecar)
