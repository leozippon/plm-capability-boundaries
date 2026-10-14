"""Diagnose the gauge of the qualified family Potts references.

E20 contrasts fields-only against couplings-only guidance, so the split
between ``h`` and ``J`` must not be an artifact of an arbitrary
parameterisation. Potts parameters are identifiable only up to a gauge:
one may add an arbitrary function of a single index to ``J`` and
compensate in ``h`` without changing any full-sequence energy. The
Frobenius norm of ``J`` -- and therefore the magnitude of a
"coupling-only" intervention -- is not gauge invariant.

The autoregressive guidance used by E20 adds, at step ``t``,

    b_t(a) = sum_{s<t} J[t, s, a, x_s]

to the logit of residue ``a``. Under the zero-sum transformation this
becomes ``b_t(a) - sum_{s<t} rowmean_ts(a)`` plus a term constant in
``a``, where ``rowmean_ij(a) = mean_b J_ij(a, b)``. The displaced piece
is a *first-order* function of the candidate residue whose size grows
with prefix length. In a gauge where the row means do not vanish, the
coupling-only arm therefore injects position-specific information --
exactly the component the marginal-rich/coupling-poor hypothesis says
the language model already carries -- and a positive result would be
confounded.

This script reports the gauge the qualified fits are actually in,
verifies that the zero-sum transformation preserves full-sequence
energies up to an additive constant, and measures how much of the
coupling-only intervention is gauge-carried first-order information
rather than pairwise covariation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context.potts_gauge import to_zero_sum  # noqa: E402

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA20)}


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name = line[1:].strip().split()[0]
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        records[name] = "".join(chunks)
    return records


def row_means(J: np.ndarray) -> np.ndarray:
    """``rowmean_ij(a) = mean_b J_ij(a, b)``, shape ``(L, L, 20)``."""
    return J.mean(axis=3)


def potts_energy(h: np.ndarray, J: np.ndarray, idx: np.ndarray) -> float:
    """``sum_i h_i(x_i) + sum_{i<j} J_ij(x_i, x_j)`` for one sequence."""
    L = idx.shape[0]
    single = h[np.arange(L), idx].sum()
    iu, ju = np.triu_indices(L, 1)
    pair = J[iu, ju, idx[iu], idx[ju]].sum()
    return float(single + pair)


def centred(v: np.ndarray) -> np.ndarray:
    """Remove the component a softmax cannot see."""
    return v - v.mean(axis=-1, keepdims=True)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def diagnose(domain: str, h: np.ndarray, J: np.ndarray, wild_type: str, rng: np.random.Generator) -> dict:
    h = np.asarray(h, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    L = h.shape[0]

    symmetry = float(np.abs(J - np.transpose(J, (1, 0, 3, 2))).max())
    diagonal = float(np.abs(J[np.arange(L), np.arange(L)]).max())

    ra = row_means(J)
    off = ~np.eye(L, dtype=bool)
    iu, ju = np.triu_indices(L, 1)

    h_zero, J_zero = to_zero_sum(h, J)

    # Gauge invariance of the joint: the energy difference must be one
    # constant shared by every sequence, so its spread is the test.
    deltas = []
    for _ in range(64):
        idx = rng.integers(0, 20, size=L)
        deltas.append(potts_energy(h_zero, J_zero, idx) - potts_energy(h, J, idx))
    deltas = np.asarray(deltas)
    energy_shift_spread = float(deltas.max() - deltas.min())
    energy_scale = float(np.abs([potts_energy(h, J, rng.integers(0, 20, size=L)) for _ in range(16)]).mean())

    # How much of J is gauge rather than covariation.
    norm_J = float(np.linalg.norm(J[iu, ju]))
    norm_J_zero = float(np.linalg.norm(J_zero[iu, ju]))
    induced = (ra * off[..., None]).sum(axis=1)

    # Effect on the realised intervention, along the wild-type prefix.
    idx_wt = np.asarray([AA_INDEX[c] for c in wild_type], dtype=int)
    cos_steps: list[float] = []
    share_steps: list[float] = []
    norm_ratio_steps: list[float] = []
    for t in range(1, L):
        prefix = idx_wt[:t]
        steps = np.arange(t)
        bias = J[t, steps, :, prefix[steps]].sum(axis=0)
        bias_zero = J_zero[t, steps, :, prefix[steps]].sum(axis=0)
        gauge_part = ra[t, steps, :].sum(axis=0)
        b_c, bz_c, g_c = centred(bias), centred(bias_zero), centred(gauge_part)
        cos_steps.append(cosine(b_c, bz_c))
        nb = float(np.linalg.norm(b_c))
        norm_ratio_steps.append(float(np.linalg.norm(bz_c)) / nb if nb > 0 else float("nan"))
        share_steps.append(float(np.linalg.norm(g_c)) / nb if nb > 0 else float("nan"))

    h_c, hz_c = centred(h), centred(h_zero)
    return {
        "domain": domain,
        "length": int(L),
        "symmetry_max_deviation": symmetry,
        "diagonal_max": diagonal,
        "energy_shift_spread": energy_shift_spread,
        "energy_scale": energy_scale,
        "norm_J": norm_J,
        "norm_J_zero_sum": norm_J_zero,
        "fraction_of_J_norm_removable_by_gauge": 1.0 - norm_J_zero / norm_J if norm_J > 0 else float("nan"),
        "norm_h": float(np.linalg.norm(centred(h))),
        "norm_induced_field": float(np.linalg.norm(centred(induced))),
        "induced_over_h": float(np.linalg.norm(centred(induced)) / np.linalg.norm(centred(h))),
        "fields_cosine_current_vs_zero_sum": cosine(h_c.ravel(), hz_c.ravel()),
        "coupling_bias_cosine_median": float(np.nanmedian(cos_steps)),
        "coupling_bias_cosine_min": float(np.nanmin(cos_steps)),
        "coupling_bias_norm_ratio_median": float(np.nanmedian(norm_ratio_steps)),
        "gauge_first_order_share_median": float(np.nanmedian(share_steps)),
        "gauge_first_order_share_max": float(np.nanmax(share_steps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True, help="directory of qualified family npz files")
    parser.add_argument("--wild-type-fasta", type=Path, required=True, help="fasta of the aligned query sequences")
    parser.add_argument("--out", type=Path, required=True, help="destination json")
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    if not args.references.is_dir():
        parser.error(f"--references is not a directory: {args.references}")
    if not args.wild_type_fasta.is_file():
        parser.error(f"--wild-type-fasta does not exist: {args.wild_type_fasta}")

    wild_types = read_fasta(args.wild_type_fasta)
    rng = np.random.default_rng(args.seed)

    reports = []
    for path in sorted(args.references.glob("*.npz")):
        domain = path.stem
        matches = [k for k in wild_types if k.upper().startswith(domain.upper())]
        if not matches:
            raise SystemExit(f"no wild-type sequence for domain {domain} in {args.wild_type_fasta}")
        sequence = wild_types[matches[0]]
        payload = np.load(path, allow_pickle=True)
        h, J = payload["h"], payload["J"]
        if len(sequence) != h.shape[0]:
            raise SystemExit(
                f"{domain}: wild-type length {len(sequence)} != model length {h.shape[0]}"
            )
        reports.append(diagnose(domain, h, J, sequence, rng))

    summary = {
        "references": str(args.references),
        "wild_type_fasta": str(args.wild_type_fasta),
        "amino_acid_order": AA20,
        "domains": reports,
        "aggregate": {
            "max_fraction_of_J_norm_removable_by_gauge": max(
                r["fraction_of_J_norm_removable_by_gauge"] for r in reports
            ),
            "max_gauge_first_order_share_median": max(
                r["gauge_first_order_share_median"] for r in reports
            ),
            "min_coupling_bias_cosine_median": min(
                r["coupling_bias_cosine_median"] for r in reports
            ),
            "max_energy_shift_spread": max(r["energy_shift_spread"] for r in reports),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], indent=2))
    for r in reports:
        print(
            f"{r['domain']}  L={r['length']:3d}  "
            f"gauge_removable_J={r['fraction_of_J_norm_removable_by_gauge']:.4f}  "
            f"induced/h={r['induced_over_h']:.4f}  "
            f"bias_cos={r['coupling_bias_cosine_median']:.4f}  "
            f"first_order_share={r['gauge_first_order_share_median']:.4f}  "
            f"energy_spread={r['energy_shift_spread']:.2e}"
        )


if __name__ == "__main__":
    main()
