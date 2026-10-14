"""External family statistics for one wild type: a Potts (DCA) model and its frozen profile.

Every quantity here is a function of the wild type's homolog alignment only; no
phenotype, no generated sequence and no language-model output is read.

Convention (plmc OutputParametersFull, focus mode, gap-ignoring). The model is
``P(a) proportional to exp(E(a))`` with ``E(a) = sum_i h[i, a_i] + sum_{i<j} J[i, j, a_i, a_j]``,
so a larger value is more favourable. ``J[i, j]`` is the ``(20, 20)`` block with the
first axis on residue ``i``; ``J[j, i] = J[i, j].T`` and the diagonal blocks are zero.
The stored parameters are in plmc's own gauge (L2-regularised ``h`` and ``J``, no gauge
fixing afterwards, as ``GAUGE_CONVENTION`` states), and at most 0.68% of the coupling
norm is gauge-removable, so they sit that close to the zero-sum gauge (see the gauge
note below). A bias vector is only meaningful up to an additive constant per position,
and the coupling-only vector is not invariant to moving mass between ``h`` and ``J``.
Column
``j`` of every array is residue ``j`` (0-based) of the wild type.

Left-to-right use. Conditioned on a prefix, the next residue is scored by
``h[t] + sum_{s<t} J[t, s, :, a_s]``. With ``P_t(x_1..x_t) proportional to
exp(sum_{i<=t} h_i + sum_{i<j<=t} J_ij)`` this is ``log P_t(x_t = a | x_<t)`` up to a
constant in ``a`` (exact to 1e-15 by enumeration): the *prefix-truncated* Potts
conditional. It is not ``log P_L(x_t = a | x_<t)``, which carries an extra
residue-dependent future free-energy term. :func:`sample_prefix_chain` therefore does not
sample the Potts model; :func:`sample_potts_joint` does.

Gauge. Every per-step bias is gauge-dependent. Under the zero-sum change
(:func:`..potts_gauge.to_zero_sum`) the step-``t`` coupling-only bias changes, up to a
constant in the candidate residue ``a``, by ``-sum_{s<t} mean_b J[t, s, a, b]``: a
first-order term in ``a`` that depends on the prefix length and not on its content. The
full bias changes by ``+sum_{j>t} mean_b J[t, j, a, b]``, the future half. Hence the
accumulated full-sequence score ``E`` is gauge-invariant up to a constant (and so is the
accumulated full-bias score), while the accumulated coupling-only score is not; "coupling
component" and "field component" always name the decomposition of the declared gauge, not
an intrinsic split of information. Measured magnitude in plmc's gauge (see
``results/R6/e20_potts_gauge_20261013/gauge_diagnostic.json``): at most 0.68% of the
coupling Frobenius norm is gauge-removable, and the gauge-carried first-order share of the
coupling bias is 3.4-7.9% (median).

Provenance note. This file is hashed into the guided-decoding cell records (``code_sha256``). The ``MixingError``
class, :func:`validate_gibbs_mixing` and the optional mixing check were added after the 232 development cells were
generated, so new cells record a different hash for it; no guided-decoding path calls any of them and
:func:`sample_potts_joint` draws are bit-identical at a fixed seed.
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
    natural-log units; ``coupling`` is exactly zero at ``t = 0``. That zero is an
    arithmetic fact about the empty prefix sum, not a gauge-free "no information"
    baseline: the coupling-only bias is gauge-dependent at every later position, so
    what the coupling component means is fixed by the declared gauge (see the gauge
    note in this module).
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


def sample_prefix_chain(model: FamilyModel, rng: np.random.Generator) -> str:
    """Draw a length-L sequence left to right from the prefix-truncated Potts conditionals.

    This samples ``Q(x) = exp(E(x)) / prod_t Z_t(x_<t)`` with ``Z_t`` the local normaliser of
    step ``t``, **not** the Potts model ``P_L(x) proportional to exp(E(x))``; use
    :func:`sample_potts_joint` for that. Measured against a Gibbs sampler of ``P_L`` (2,000
    draws each), the chain's mean Potts energy is lower: 144.06 vs 155.76 (1K1V), 159.29 vs
    175.96 (2KWH), 206.05 vs 255.66 (3L1X), i.e. 54, 63 and 155 standard errors; the maximum
    per-column single-site total-variation distance is 0.33 / 0.28 / 0.68 against a
    Monte-Carlo floor of 0.03-0.07, concentrated in early columns (prefix truncation).
    """
    chosen = ""
    for _ in range(model.length):
        logits = guidance_biases(model, chosen)["full"]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        chosen += model.alphabet[int(rng.choice(len(p), p=p))]
    return chosen


@dataclass(frozen=True)
class GibbsDraws:
    """``n`` sequences from :func:`sample_potts_joint` with the schedule that produced them."""

    sequences: list[str]
    sweeps: int  # sweeps each chain ran before its final state was taken (burn-in = sweeps)
    check_sweeps: int  # the earlier sweep count whose marginals were compared with the final ones
    mixing_max_tv: float | None  # max over columns of TV(single-site marginals at check_sweeps vs sweeps)
    mixing_tolerance: float | None


class MixingError(RuntimeError):
    """A Gibbs mixing check failed; ``numbers`` holds the measured quantities for the cell record."""

    def __init__(self, message: str, **numbers):
        super().__init__(message)
        self.numbers = numbers


def _site_marginals(codes: np.ndarray) -> np.ndarray:
    q = len(AA20)
    return np.stack([np.bincount(codes[:, i], minlength=q) for i in range(codes.shape[1])]) / codes.shape[0]


def _column_tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum(axis=1).max())


def _gibbs(model: FamilyModel, X: np.ndarray, sweeps: int, rng: np.random.Generator, snapshot_at: int | None = None):
    """Run ``sweeps`` random-scan sweeps in place on ``X`` (chains by columns); return the marginals at ``snapshot_at``."""
    cols, early = np.arange(model.length), None
    for sweep in range(1, sweeps + 1):
        for i in rng.permutation(model.length):
            logits = model.h[i] + model.J[i][cols, :, X].sum(axis=1)  # (n, 20); J[i, i] = 0
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            cdf = p.cumsum(axis=1)
            X[:, i] = (rng.random((len(X), 1)) * cdf[:, -1:] > cdf).sum(axis=1)
        if sweep == snapshot_at:
            early = _site_marginals(X)
    return early


def sample_potts_joint(model: FamilyModel, n: int, rng: np.random.Generator, *, sweeps: int,
                       mixing_tolerance: float | None) -> GibbsDraws:
    """Draw ``n`` sequences from ``P_L(x) proportional to exp(E(x))`` by random-scan Gibbs sampling.

    ``n`` independent chains start from uniform random sequences and each runs ``sweeps`` sweeps
    (one sweep resamples every column once, in a random order, from its full conditional given
    all other columns); the final state is the draw, so the burn-in is the whole ``sweeps``.
    With a ``mixing_tolerance`` the single-site marginals after ``sweeps // 2`` sweeps are compared with
    the final ones and the call raises if any column's total-variation distance exceeds it; that
    statistic is dominated by Monte-Carlo noise unless ``n`` is large (about 1/sqrt(n)), so the tolerance
    must sit above that noise. ``None`` skips the check (``mixing_max_tv`` is then ``None``); use
    :func:`validate_gibbs_mixing` to validate the sampler for a domain.
    """
    if n < 1 or sweeps < 2 or (mixing_tolerance is not None and mixing_tolerance <= 0):
        raise ValueError("need n >= 1, sweeps >= 2 and a positive mixing tolerance")
    X = rng.integers(0, len(AA20), size=(n, model.length))
    check_sweeps = sweeps // 2
    early = _gibbs(model, X, sweeps, rng, check_sweeps if mixing_tolerance is not None else None)
    tv = None
    if mixing_tolerance is not None:
        tv = _column_tv(early, _site_marginals(X))
        if tv > mixing_tolerance:
            raise MixingError(f"Gibbs chains have not mixed: max single-site TV {tv:.4f} between "
                              f"{check_sweeps} and {sweeps} sweeps exceeds {mixing_tolerance}",
                              mixing_max_tv=tv, check_sweeps=check_sweeps, sweeps=sweeps,
                              mixing_tolerance=mixing_tolerance)
    letters = np.array(list(model.alphabet))
    return GibbsDraws(["".join(row) for row in letters[X]], sweeps, check_sweeps, tv, mixing_tolerance)


def validate_gibbs_mixing(model: FamilyModel, rng: np.random.Generator, *, n: int, sweeps: int, factor: float) -> dict:
    """Initialisation-dependence check of the Gibbs sampler on one domain; raises ``MixingError`` on failure.

    Two overdispersed initialisations -- A uniform-random residues, B the wild type -- each run ``n`` chains for
    ``sweeps`` sweeps. ``between_init_tv`` is the largest single-site TV between the two final marginals;
    ``noise_floor`` the largest single-site TV between two independent halves of A (pure sampling noise at
    comparable size). Pass when ``between_init_tv <= factor * noise_floor``. Chains stuck near their start would
    differ between A and B, which the half-run-versus-full-run statistic cannot see.
    """
    if n < 2 or n % 2 or sweeps < 1 or factor <= 0:
        raise ValueError("need an even n >= 2, sweeps >= 1 and a positive factor")
    A = rng.integers(0, len(AA20), size=(n, model.length))
    B = np.tile(encode(model.wildtype), (n, 1))
    _gibbs(model, A, sweeps, rng)
    _gibbs(model, B, sweeps, rng)
    between = _column_tv(_site_marginals(A), _site_marginals(B))
    floor = _column_tv(_site_marginals(A[: n // 2]), _site_marginals(A[n // 2:]))
    numbers = {"between_init_tv": between, "noise_floor": floor, "ratio": between / floor if floor else float("inf"),
               "factor": factor, "n_validation_per_init": n, "sweeps": sweeps, "passed": between <= factor * floor}
    if not numbers["passed"]:
        raise MixingError(f"Gibbs chains depend on their initialisation: between-init TV {between:.4f} exceeds "
                          f"{factor} x noise floor {floor:.4f}", **numbers)
    return numbers


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
