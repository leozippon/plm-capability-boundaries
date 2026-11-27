"""Adapter protocol, metadata schema, 1024-d representation packer, and fail-fast validator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.capability.core.amino_acids import AA20
from src.capability.general_plm.eval_utils import project_four_blocks

AA20_SET = frozenset(AA20)
AA20_INDEX = {ch: i for i, ch in enumerate(AA20)}

VALID_CONDITIONING_REGIMES = frozenset(
    {
        "sequence_only",
        "backbone_3d_conditioned_oracle",
        "multiseq_icl_unaligned",
        "msa_2d_axial",
    }
)
VALID_INPUT_ALIGNMENT_FORMATS = frozenset(
    {
        "unaligned_concat",
        "explicit_column_aligned",
        "two_chain_spatial_offset_3d",
    }
)


class CapabilityNotSupportedError(NotImplementedError):
    """Raised when an adapter does not implement a requested benchmark hook."""


class AdapterValidationError(ValueError):
    """Raised when an adapter fails contract, normalization, or anti-leakage checks."""


@dataclass(frozen=True)
class ModelMetadata:
    """Authoritative metadata describing an evaluated model."""

    model_id: str
    display_name: str
    paradigm: str
    modality: str
    conditioning_regime: str = "sequence_only"
    input_alignment_format: str = "unaligned_concat"
    n_params_millions: float = 1.0
    max_positions: int = 1024

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not self.display_name or not self.display_name.strip():
            raise ValueError("display_name must be a non-empty string")
        if not self.paradigm or not self.paradigm.strip():
            raise ValueError("paradigm must be a non-empty string")
        if not self.modality or not self.modality.strip():
            raise ValueError("modality must be a non-empty string")
        if self.conditioning_regime not in VALID_CONDITIONING_REGIMES:
            raise ValueError(
                f"{self.model_id}: invalid conditioning_regime {self.conditioning_regime!r}; "
                f"expected one of {sorted(VALID_CONDITIONING_REGIMES)}"
            )
        if self.input_alignment_format not in VALID_INPUT_ALIGNMENT_FORMATS:
            raise ValueError(
                f"{self.model_id}: invalid input_alignment_format {self.input_alignment_format!r}; "
                f"expected one of {sorted(VALID_INPUT_ALIGNMENT_FORMATS)}"
            )
        if not (math.isfinite(self.n_params_millions) and self.n_params_millions > 0.0):
            raise ValueError(f"{self.model_id}: n_params_millions must be positive finite, got {self.n_params_millions!r}")
        if int(self.max_positions) < 1:
            raise ValueError(f"{self.model_id}: max_positions must be >= 1, got {self.max_positions!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapabilityFlags:
    """Explicit declaration of which benchmark hooks an adapter supports."""

    score_mutations: bool = False
    site_conditional_logprobs: bool = False
    score_sequence_nll: bool = False
    extract_representations_1024: bool = False
    generate_unconditional: bool = False
    infill_local: bool = False
    sample_family_guided: bool = False
    continue_from_prefix: bool = False
    requires_backbone_3d: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@runtime_checkable
class ProteinBenchmarkAdapter(Protocol):
    """Runtime-checkable protocol for any external or first-party protein model adapter."""

    @property
    def metadata(self) -> ModelMetadata: ...

    @property
    def capabilities(self) -> CapabilityFlags: ...

    def score_mutations(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def site_conditional_logprobs(
        self,
        sequence: str,
        positions: Sequence[int] | None = None,
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def score_sequence_nll(
        self,
        sequence: str,
        *,
        context_items: Sequence[str] = (),
        coords: np.ndarray | None = None,
        context_coords: Sequence[np.ndarray] = (),
    ) -> dict[str, float]: ...

    def extract_representations_1024(
        self,
        sequences: Sequence[str],
        *,
        coords_list: Sequence[np.ndarray | None] | None = None,
    ) -> np.ndarray: ...

    def generate_unconditional(
        self,
        lengths: Sequence[int],
        *,
        seed: int = 20261124,
    ) -> list[str]: ...

    def infill_local(
        self,
        template: str,
        mask_positions: Sequence[int],
        *,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> str: ...

    def sample_family_guided(
        self,
        length: int,
        *,
        n_samples: int,
        profile_log_bias: np.ndarray | None = None,
        guidance_scale: float = 0.0,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> list[str]: ...

    def continue_from_prefix(
        self,
        context_items: Sequence[str],
        target_length: int,
        *,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> str: ...


class BaseProteinBenchmarkAdapter:
    """Convenience base class implementing fail-fast ``CapabilityNotSupportedError`` defaults."""

    def __init__(self, metadata: ModelMetadata, capabilities: CapabilityFlags) -> None:
        self._metadata = metadata
        self._capabilities = capabilities

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def capabilities(self) -> CapabilityFlags:
        return self._capabilities

    def score_mutations(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        raise CapabilityNotSupportedError(f"{self.metadata.model_id}: score_mutations is not supported")

    def site_conditional_logprobs(
        self,
        sequence: str,
        positions: Sequence[int] | None = None,
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        raise CapabilityNotSupportedError(f"{self.metadata.model_id}: site_conditional_logprobs is not supported")

    def score_sequence_nll(
        self,
        sequence: str,
        *,
        context_items: Sequence[str] = (),
        coords: np.ndarray | None = None,
        context_coords: Sequence[np.ndarray] = (),
    ) -> dict[str, float]:
        raise CapabilityNotSupportedError(f"{self.metadata.model_id}: score_sequence_nll is not supported")

    def extract_representations_1024(
        self,
        sequences: Sequence[str],
        *,
        coords_list: Sequence[np.ndarray | None] | None = None,
    ) -> np.ndarray:
        raise CapabilityNotSupportedError(f"{self.metadata.model_id}: extract_representations_1024 is not supported")

    def generate_unconditional(
        self,
        lengths: Sequence[int],
        *,
        seed: int = 20261124,
    ) -> list[str]:
        raise CapabilityNotSupportedError(f"{self.metadata.model_id}: generate_unconditional is not supported")

    def infill_local(
        self,
        template: str,
        mask_positions: Sequence[int],
        *,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> str:
        raise CapabilityNotSupportedError(f"{self.metadata.model_id}: infill_local is not supported")

    def sample_family_guided(
        self,
        length: int,
        *,
        n_samples: int,
        profile_log_bias: np.ndarray | None = None,
        guidance_scale: float = 0.0,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> list[str]:
        raise CapabilityNotSupportedError(f"{self.metadata.model_id}: sample_family_guided is not supported")

    def continue_from_prefix(
        self,
        context_items: Sequence[str],
        target_length: int,
        *,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> str:
        raise CapabilityNotSupportedError(f"{self.metadata.model_id}: continue_from_prefix is not supported")


def pack_mid_and_last_hidden_states(
    h_mid: np.ndarray,
    h_last: np.ndarray,
    *,
    seed: int = 20260923,
) -> np.ndarray:
    """Pack raw ``L//2`` and ``L`` hidden states into the canonical 1024-d float32 readout.

    Accepts either:
    * 2D arrays ``(L, D)`` for a single sequence -> returns ``(1024,)`` float32.
    * 3D arrays ``(N, L, D)`` for a batch -> returns ``(N, 1024)`` float32.

    Pools ``[middle_mean, middle_last, final_mean, final_last]`` into ``(N, 4, D)`` and
    applies the exact deterministic Gaussian JL projection (``project_four_blocks``,
    ``seed=20260923``) so ``0:512`` is ``R_mid`` and ``512:1024`` is ``R_last``.
    """
    mid_arr = np.asarray(h_mid, dtype=np.float32)
    last_arr = np.asarray(h_last, dtype=np.float32)
    if mid_arr.shape != last_arr.shape:
        raise ValueError(f"h_mid shape {mid_arr.shape} != h_last shape {last_arr.shape}")
    if mid_arr.ndim == 2:
        squeeze = True
        mid_arr = mid_arr[None, :, :]
        last_arr = last_arr[None, :, :]
    elif mid_arr.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"Expected 2D (L, D) or 3D (N, L, D) hidden states, got ndim={mid_arr.ndim}")
    if mid_arr.shape[1] < 1 or mid_arr.shape[2] < 1:
        raise ValueError(f"Hidden states must have positive length and dimension, got {mid_arr.shape}")
    if not (np.isfinite(mid_arr).all() and np.isfinite(last_arr).all()):
        raise ValueError("Hidden states contain non-finite values")

    blocks = np.stack(
        [
            mid_arr.mean(axis=1),
            mid_arr[:, -1, :],
            last_arr.mean(axis=1),
            last_arr[:, -1, :],
        ],
        axis=1,
    )  # (N, 4, D)
    projected = project_four_blocks(blocks, seed=int(seed)).astype(np.float32, copy=False)
    return projected[0] if squeeze else projected


def _synthetic_alpha_helix_coords(length: int) -> np.ndarray:
    """Deterministic ideal 3D ``(L, 4, 3)`` backbone (N, CA, C, O) for validation."""
    coords = np.zeros((length, 4, 3), dtype=np.float32)
    for i in range(length):
        theta = i * 1.7453292519943295  # 100 deg/res
        z = i * 1.5
        ca = np.array([2.3 * math.cos(theta), 2.3 * math.sin(theta), z], dtype=np.float32)
        n = ca + np.array([-0.52 * math.sin(theta), 0.52 * math.cos(theta), -1.36], dtype=np.float32)
        c = ca + np.array([0.52 * math.sin(theta), -0.52 * math.cos(theta), 1.43], dtype=np.float32)
        o = c + np.array([1.23 * math.cos(theta), 1.23 * math.sin(theta), 0.1], dtype=np.float32)
        coords[i] = np.stack([n, ca, c, o], axis=0)
    return coords


def validate_adapter(
    adapter: ProteinBenchmarkAdapter,
    *,
    coords: np.ndarray | None = None,
    seed: int = 20261124,
    check_leakage: bool = True,
) -> dict[str, Any]:
    """Fail-fast contract, normalization, determinism, and zero-leakage verifier.

    Raises ``AdapterValidationError`` on any contract violation or target-residue leakage.
    """
    if not isinstance(adapter, ProteinBenchmarkAdapter):
        raise AdapterValidationError("Adapter does not satisfy the ProteinBenchmarkAdapter protocol")
    meta = adapter.metadata
    caps = adapter.capabilities
    if not isinstance(meta, ModelMetadata):
        raise AdapterValidationError(f"metadata must be ModelMetadata, got {type(meta).__name__}")
    if not isinstance(caps, CapabilityFlags):
        raise AdapterValidationError(f"capabilities must be CapabilityFlags, got {type(caps).__name__}")

    probe_seq = "MKTAYIAKQRQISFVK"
    probe_len = len(probe_seq)
    eff_coords = coords
    if eff_coords is None and caps.requires_backbone_3d:
        eff_coords = _synthetic_alpha_helix_coords(probe_len)

    checks: dict[str, Any] = {"model_id": meta.model_id, "passed": True, "checks": {}}

    # 1. Site-conditional logprobs: shape, normalization, and STRICT ZERO TARGET-RESIDUE LEAKAGE
    if caps.site_conditional_logprobs:
        pos = 5  # 'I'
        lp = np.asarray(
            adapter.site_conditional_logprobs(probe_seq, positions=[pos], coords=eff_coords),
            dtype=np.float64,
        )
        if lp.shape != (1, 20):
            raise AdapterValidationError(f"site_conditional_logprobs expected shape (1, 20), got {lp.shape}")
        if not np.isfinite(lp).all():
            raise AdapterValidationError("site_conditional_logprobs returned non-finite values")
        lse = float(np.logaddexp.reduce(lp[0]))
        if abs(lse) > 1e-2:
            raise AdapterValidationError(
                f"site_conditional_logprobs must be log-normalized over AA20 (logsumexp=0), got {lse:.6f}"
            )
        max_leak = 0.0
        if check_leakage:
            alt_aa = "W" if probe_seq[pos] != "W" else "G"
            mutated = probe_seq[:pos] + alt_aa + probe_seq[pos + 1 :]
            lp_mut = np.asarray(
                adapter.site_conditional_logprobs(mutated, positions=[pos], coords=eff_coords),
                dtype=np.float64,
            )
            if lp_mut.shape != (1, 20) or not np.isfinite(lp_mut).all():
                raise AdapterValidationError("site_conditional_logprobs failed on mutated probe sequence")
            max_leak = float(np.max(np.abs(lp - lp_mut)))
            if max_leak > 1e-3:
                raise AdapterValidationError(
                    f"Target-residue leakage detected at position {pos} "
                    f"(max |lp_wt - lp_mut| = {max_leak:.6g} > 1e-3). "
                    "site_conditional_logprobs MUST mask or exclude the target residue identity."
                )
        checks["checks"]["site_conditional_logprobs"] = {
            "passed": True,
            "logsumexp": lse,
            "max_target_leakage": max_leak,
        }

    # 2. Mutation scoring
    if caps.score_mutations:
        muts = ["M1A", "K2E", "M1A:K2E"]
        sc = np.asarray(adapter.score_mutations(probe_seq, muts, coords=eff_coords), dtype=np.float64)
        if sc.shape != (3,) or not np.isfinite(sc).all():
            raise AdapterValidationError(f"score_mutations expected finite shape (3,), got {sc.shape}")
        checks["checks"]["score_mutations"] = {"passed": True, "sample_scores": [float(v) for v in sc]}

    # 3. Sequence NLL
    if caps.score_sequence_nll:
        res = adapter.score_sequence_nll(probe_seq, coords=eff_coords)
        if not isinstance(res, dict) or "nll_sum" not in res or "nll_per_residue" not in res:
            raise AdapterValidationError("score_sequence_nll must return dict with 'nll_sum' and 'nll_per_residue'")
        nll_s = float(res["nll_sum"])
        nll_r = float(res["nll_per_residue"])
        if not (math.isfinite(nll_s) and math.isfinite(nll_r) and nll_s >= 0.0 and nll_r >= 0.0):
            raise AdapterValidationError(f"score_sequence_nll returned invalid values: {res}")
        checks["checks"]["score_sequence_nll"] = {"passed": True, "nll_per_residue": nll_r}

    # 4. 1024-d representation readout
    if caps.extract_representations_1024:
        alt_seq = "AKTAYIAKQRQISFVK"
        c_list = [eff_coords, eff_coords] if eff_coords is not None else None
        rep = np.asarray(adapter.extract_representations_1024([probe_seq, alt_seq], coords_list=c_list))
        if rep.shape != (2, 1024):
            raise AdapterValidationError(f"extract_representations_1024 expected shape (2, 1024), got {rep.shape}")
        if rep.dtype != np.float32:
            raise AdapterValidationError(f"extract_representations_1024 expected float32, got {rep.dtype}")
        if not np.isfinite(rep).all():
            raise AdapterValidationError("extract_representations_1024 returned non-finite values")
        checks["checks"]["extract_representations_1024"] = {
            "passed": True,
            "mid_vs_last_l2": float(np.linalg.norm(rep[0, :512] - rep[0, 512:])),
        }

    # 5. Generation hooks (alphabet, length, determinism, scaffold/prefix preservation)
    if caps.generate_unconditional:
        g1 = adapter.generate_unconditional([12], seed=seed)
        g2 = adapter.generate_unconditional([12], seed=seed)
        if len(g1) != 1 or len(g1[0]) != 12 or not (set(g1[0]) <= AA20_SET):
            raise AdapterValidationError(f"generate_unconditional returned invalid sequence: {g1!r}")
        if g1 != g2:
            raise AdapterValidationError("generate_unconditional is not deterministic under identical seed")
        checks["checks"]["generate_unconditional"] = {"passed": True, "deterministic": True}

    if caps.infill_local:
        mask_pos = [4, 5, 6]
        s1 = adapter.infill_local(probe_seq, mask_pos, coords=eff_coords, seed=seed)
        s2 = adapter.infill_local(probe_seq, mask_pos, coords=eff_coords, seed=seed)
        if len(s1) != probe_len or not (set(s1) <= AA20_SET):
            raise AdapterValidationError(f"infill_local returned invalid sequence: {s1!r}")
        if s1[:4] != probe_seq[:4] or s1[7:] != probe_seq[7:]:
            raise AdapterValidationError("infill_local modified unmasked scaffold positions")
        if s1 != s2:
            raise AdapterValidationError("infill_local is not deterministic under identical seed")
        checks["checks"]["infill_local"] = {"passed": True, "deterministic": True}

    if caps.sample_family_guided:
        bias = np.zeros((probe_len, 20), dtype=np.float32)
        bias[:, AA20_INDEX["A"]] = 2.0
        gs1 = adapter.sample_family_guided(
            probe_len, n_samples=1, profile_log_bias=bias, guidance_scale=1.0, coords=eff_coords, seed=seed
        )
        gs2 = adapter.sample_family_guided(
            probe_len, n_samples=1, profile_log_bias=bias, guidance_scale=1.0, coords=eff_coords, seed=seed
        )
        if len(gs1) != 1 or len(gs1[0]) != probe_len or not (set(gs1[0]) <= AA20_SET):
            raise AdapterValidationError(f"sample_family_guided returned invalid output: {gs1!r}")
        if gs1 != gs2:
            raise AdapterValidationError("sample_family_guided is not deterministic under identical seed")
        checks["checks"]["sample_family_guided"] = {"passed": True, "deterministic": True}

    if caps.continue_from_prefix:
        c1 = adapter.continue_from_prefix([probe_seq[:6]], target_length=probe_len, coords=eff_coords, seed=seed)
        c2 = adapter.continue_from_prefix([probe_seq[:6]], target_length=probe_len, coords=eff_coords, seed=seed)
        if len(c1) != probe_len or not (set(c1) <= AA20_SET):
            raise AdapterValidationError(f"continue_from_prefix returned invalid sequence: {c1!r}")
        if c1 != c2:
            raise AdapterValidationError("continue_from_prefix is not deterministic under identical seed")
        checks["checks"]["continue_from_prefix"] = {"passed": True, "deterministic": True}

    return checks
