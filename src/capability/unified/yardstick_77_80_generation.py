"""Unified 15-Model (DPLM-2 x 4 + ProteinMPNN x 11) Generation, Folding & R6 Adaptation Library.

Expands all five generation/folding benchmarks plus R6 structural/objective adaptation
contrasts from the sealed 65-arm baseline to the full 80-arm universe WITHOUT ever
modifying any sealed baseline JSON artifact:

  1. Experiment 4.1 (``r5_experiment_4_1_unified_generation_full.json``):
     Sealed 34 rows + 30 new DPLM-2/ProteinMPNN arm-mode rows strictly partitioned by
     ``conditioning_regime`` (``unconditional_seq_only``, ``unconditional_co_generation``,
     and ``backbone_3d_inverse_folding``).
  2. R5a Local Infilling (``r5a_local_infilling_full.json``):
     Sealed 18-arm progressive scaffold-anchoring ladder + 19 new arm-mode rows
     (4 DPLM-2 seq_only, 4 DPLM-2 struct_conditioned, 11 ProteinMPNN) over all 16
     ``ladder_backbones.jsonl`` parents x 6 window rungs (``k1, k2, k5, k10, k20, k40``).
  3. Experiment 4.2 Eight-Domain Family Phase Space (``r6_experiment_4_2_phase_space_full.json``):
     Sealed 23-PLM + P/H baseline + 4 DPLM-2 seq_only (full 5-arm Potts-guided
     ``N/Pr/Pr_strong/C/Cx`` with ``< 1e-6`` exact bias accounting) + 4 DPLM-2
     ``co_generation`` (``N``) + 4 DPLM-2 ``struct_conditioned`` (``N``) + 11 ProteinMPNN
     inverse folding (``N`` at ``T=0.1`` and ``N_T10`` at ``T=1.0``).
  4. Experiment 4.3 Native Self-Likelihood Ranking (``r6_experiment_4_3_native_likelihood_full.json``):
     Sealed 33-pool baseline + self-likelihood ranking over all newly folded Exp 4.1 pools
     (DPLM-2 ``exact_pll`` in ``seq_only``/``struct_conditioned``; ProteinMPNN
     ``exact_sequence_ll_batch`` on both ESMFold2-refolded and input template backbones).
  5. Result 5 Downstream 2 Continuation (``r4_continuation_unified_full.json``):
     Sealed 35-entry baseline + 4 DPLM-2 ``seq_only`` unaligned homolog-prefix
     continuations (matched ``budget=1024``, ``uniform_random``) + 4 DPLM-2
     ``struct_conditioned`` & 11 ProteinMPNN 50%-WT-prefix 3D-backbone suffix
     continuations (explicitly tagged ``backbone_3d_wt_prefix_suffix``).
  6. Result 6 Structural & Objective Adaptation Contrasts
     (``r6_dplm2_mpnn_adaptation_unified.json``):
     Paired 10,000-draw bootstrap contrasts for SolubleMPNN vs VanillaMPNN (matched
     noise), CA-only vs all-atom VanillaMPNN, ProteinMPNN noise ladder (0.02->0.30 A),
     DPLM-2-Bit-650M vs DPLM-2-650M, and DPLM-2 vs DPLM-1 (150M/650M/3B).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file
from src.capability.memory_theory.io_utils import load_json_file, write_json_with_sha256
from src.capability.unified import r4_continuation_yardstick as r4c
from src.capability.unified import r5_experiment_4_1 as exp41
from src.capability.unified import r6_experiment_4_2 as exp42
from src.capability.unified import r6_experiment_4_3 as exp43
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE
from src.capability.unified.roster import (
    ARM_METADATA,
    DIMA_4_ARMS,
    DPLM2_4_ARMS,
    EXPANDED_15_STRUCT_ARMS,
    EXPANDED_19_NEW_ARMS,
    PROTEINMPNN_11_ARMS,
    REPO_ROOT,
    UNIFIED_OUTPUT_DIR,
)

WORK_DIR_REL = UNIFIED_OUTPUT_DIR / "dplm2_mpnn_generation_pools"
WT_BACKBONES_REL = UNIFIED_OUTPUT_DIR / "wt_backbones_3d"
LADDER_BACKBONES_REL = Path("results/audit_20261011_h200/ladder_original/ladder_backbones.jsonl")
NATURAL_FOLD_400_REL = UNIFIED_OUTPUT_DIR / "exp4_1_pools/natural_fragments_fold_400.jsonl"

R5A_RUNGS: tuple[str, ...] = ("k1", "k2", "k5", "k10", "k20", "k40")
R5A_REPLICATES: int = 2
EXP4_1_POOL_SIZE: int = 200
EXP4_2_POOL_SIZE: int = 100
DIMA_MAX_RESIDUES: int = 510
DEFAULT_BOOTSTRAP_DRAWS: int = 10000
DEFAULT_SEED: int = 20261125

PROTEINMPNN_T10_SENSITIVITY_ARMS: tuple[str, ...] = (
    "proteinmpnn-v-020",
    "proteinmpnn-s-020",
    "proteinmpnn-ca-020",
)
DIMA_ODE_ABLATION_ARMS: tuple[str, ...] = (
    "dima-esm2-8m",
    "dima-esm2-3b",
)
DIMA_ENCODER_CONTRAST_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("esm2_3b_vs_esm2_8m", "dima-esm2-3b", "dima-esm2-8m"),
    ("saprot_35m_vs_esm2_8m", "dima-saprot-35m", "dima-esm2-8m"),
    ("cheap_1024_vs_esm2_3b", "dima-cheap-1024", "dima-esm2-3b"),
    ("cheap_1024_vs_saprot_35m", "dima-cheap-1024", "dima-saprot-35m"),
)
DIMA_CONTINUOUS_VS_DISCRETE_PAIRS: tuple[tuple[str, str, str, str, str], ...] = (
    ("dima_esm2_3b_vs_dplm_3b", "dima-esm2-3b", "dplm-3b|uniform_random", "dplm-3b", "dplm-3b"),
    ("dima_esm2_3b_vs_dplm2_3b_seq", "dima-esm2-3b", "dplm2-3b|seq_only|uniform_random", "dplm2-3b|seq_only", "dplm2-3b|seq_only"),
    ("dima_esm2_8m_vs_dplm_150m", "dima-esm2-8m", "dplm-150m|uniform_random", "dplm-150m", "dplm-150m"),
    ("dima_esm2_8m_vs_dplm2_150m_seq", "dima-esm2-8m", "dplm2-150m|seq_only|uniform_random", "dplm2-150m|seq_only", "dplm2-150m|seq_only"),
    ("dima_saprot_35m_vs_dplm2_150m_seq", "dima-saprot-35m", "dplm2-150m|seq_only|uniform_random", "dplm2-150m|seq_only", "dplm2-150m|seq_only"),
    ("dima_cheap_1024_vs_dplm2_650m_seq", "dima-cheap-1024", "dplm2-650m|seq_only|uniform_random", "dplm2-650m|seq_only", "dplm2-650m|seq_only"),
)

DPLM2_TO_DPLM1_PAIRS: tuple[tuple[str, str], ...] = (
    ("dplm2-150m", "dplm-150m"),
    ("dplm2-650m", "dplm-650m"),
    ("dplm2-3b", "dplm-3b"),
)
SOLUBLE_VS_VANILLA_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("0.02A", "proteinmpnn-s-002", "proteinmpnn-v-002"),
    ("0.10A", "proteinmpnn-s-010", "proteinmpnn-v-010"),
    ("0.20A", "proteinmpnn-s-020", "proteinmpnn-v-020"),
    ("0.30A", "proteinmpnn-s-030", "proteinmpnn-v-030"),
)
CA_VS_VANILLA_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("0.02A", "proteinmpnn-ca-002", "proteinmpnn-v-002"),
    ("0.10A", "proteinmpnn-ca-010", "proteinmpnn-v-010"),
    ("0.20A", "proteinmpnn-ca-020", "proteinmpnn-v-020"),
)
MPNN_NOISE_LADDERS: Mapping[str, tuple[tuple[float, str], ...]] = {
    "vanilla": (
        (0.02, "proteinmpnn-v-002"),
        (0.10, "proteinmpnn-v-010"),
        (0.20, "proteinmpnn-v-020"),
        (0.30, "proteinmpnn-v-030"),
    ),
    "soluble": (
        (0.02, "proteinmpnn-s-002"),
        (0.10, "proteinmpnn-s-010"),
        (0.20, "proteinmpnn-s-020"),
        (0.30, "proteinmpnn-s-030"),
    ),
    "ca_only": (
        (0.02, "proteinmpnn-ca-002"),
        (0.10, "proteinmpnn-ca-010"),
        (0.20, "proteinmpnn-ca-020"),
    ),
}

AA20_SET = frozenset(AA20)


# --------------------------------------------------------------------------- #
# Entry specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Exp41NewSpec:
    """Declarative specification for one DPLM-2 / ProteinMPNN Exp 4.1 entry."""

    entry_key: str
    arm_id: str
    backend: str  # dplm2 | proteinmpnn
    mode: str  # seq_only | co_generation | struct_conditioned | inverse_folding
    conditioning_regime: str  # unconditional_seq_only | unconditional_co_generation | backbone_3d_inverse_folding
    paradigm_group: str
    generation_regime: str
    diffusion_mode: str | None = None
    temperature: float = 1.0
    is_primary_mode: bool = True
    requires_backbone_3d: bool = False

    @property
    def file_slug(self) -> str:
        return entry_file_slug(self.entry_key)


@dataclass(frozen=True)
class R5aNewSpec:
    """Declarative specification for one DPLM-2 / ProteinMPNN R5a infilling entry."""

    entry_key: str
    arm_id: str
    backend: str  # dplm2 | proteinmpnn
    mode: str  # seq_only | struct_conditioned | inverse_folding
    conditioning_regime: str  # sequence_scaffold_infill | backbone_3d_scaffold_infill
    temperature: float = 1.0
    requires_backbone_3d: bool = False

    @property
    def file_slug(self) -> str:
        return entry_file_slug(self.entry_key)


@dataclass(frozen=True)
class Exp42NewSpec:
    """Declarative specification for one DPLM-2 / ProteinMPNN Exp 4.2 entry."""

    entry_key: str
    arm_id: str
    backend: str  # dplm2 | proteinmpnn
    mode: str  # seq_only | co_generation | struct_conditioned | inverse_folding
    conditioning_regime: str  # unconditional_seq_only | unconditional_co_generation | backbone_3d_inverse_folding
    arm_keys: tuple[str, ...]
    requires_backbone_3d: bool = False

    @property
    def file_slug(self) -> str:
        return entry_file_slug(self.entry_key)


@dataclass(frozen=True)
class R4ContNewSpec:
    """Declarative specification for one DPLM-2 / ProteinMPNN Downstream 2 continuation entry."""

    entry_key: str
    arm_id: str
    backend: str  # dplm2 | proteinmpnn
    mode: str  # seq_only | struct_conditioned | inverse_folding
    context_training_regime: str  # single_chain_pretrained | backbone_3d_wt_prefix_suffix
    conditioning_regime: str  # unaligned_homolog_prefix | backbone_3d_wt_prefix_suffix
    requires_backbone_3d: bool = False

    @property
    def file_slug(self) -> str:
        return entry_file_slug(self.entry_key)


def entry_file_slug(entry_key: str) -> str:
    return (
        entry_key.replace("|", "__")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "p")
    )


def _build_exp4_1_specs() -> tuple[Exp41NewSpec, ...]:
    specs: list[Exp41NewSpec] = []
    for arm in DPLM2_4_ARMS:
        specs.append(
            Exp41NewSpec(
                entry_key=f"{arm}|seq_only|uniform_random",
                arm_id=arm,
                backend="dplm2",
                mode="seq_only",
                conditioning_regime="unconditional_seq_only",
                paradigm_group="multimodal_diffusion_seq_only",
                generation_regime="length_matched_diffusion",
                diffusion_mode="uniform_random",
                temperature=1.0,
                is_primary_mode=True,
                requires_backbone_3d=False,
            )
        )
        specs.append(
            Exp41NewSpec(
                entry_key=f"{arm}|seq_only|max_confidence",
                arm_id=arm,
                backend="dplm2",
                mode="seq_only",
                conditioning_regime="unconditional_seq_only",
                paradigm_group="multimodal_diffusion_seq_only_maxconf",
                generation_regime="length_matched_diffusion_maxconf",
                diffusion_mode="max_confidence",
                temperature=1.0,
                is_primary_mode=False,
                requires_backbone_3d=False,
            )
        )
        specs.append(
            Exp41NewSpec(
                entry_key=f"{arm}|co_generation",
                arm_id=arm,
                backend="dplm2",
                mode="co_generation",
                conditioning_regime="unconditional_co_generation",
                paradigm_group="multimodal_diffusion_co_generation",
                generation_regime="length_matched_co_generation",
                diffusion_mode="co_generation",
                temperature=1.0,
                is_primary_mode=True,
                requires_backbone_3d=False,
            )
        )
        specs.append(
            Exp41NewSpec(
                entry_key=f"{arm}|struct_conditioned",
                arm_id=arm,
                backend="dplm2",
                mode="struct_conditioned",
                conditioning_regime="backbone_3d_inverse_folding",
                paradigm_group="multimodal_diffusion_struct_conditioned",
                generation_regime="backbone_3d_inverse_folding",
                diffusion_mode="uniform_random",
                temperature=1.0,
                is_primary_mode=True,
                requires_backbone_3d=True,
            )
        )
    for arm in PROTEINMPNN_11_ARMS:
        specs.append(
            Exp41NewSpec(
                entry_key=f"{arm}|T0.1",
                arm_id=arm,
                backend="proteinmpnn",
                mode="inverse_folding",
                conditioning_regime="backbone_3d_inverse_folding",
                paradigm_group="inverse_folding_gnn",
                generation_regime="backbone_3d_inverse_folding",
                diffusion_mode=None,
                temperature=0.1,
                is_primary_mode=True,
                requires_backbone_3d=True,
            )
        )
    for arm in PROTEINMPNN_T10_SENSITIVITY_ARMS:
        specs.append(
            Exp41NewSpec(
                entry_key=f"{arm}|T1.0",
                arm_id=arm,
                backend="proteinmpnn",
                mode="inverse_folding",
                conditioning_regime="backbone_3d_inverse_folding",
                paradigm_group="inverse_folding_gnn_t10",
                generation_regime="backbone_3d_inverse_folding_t10",
                diffusion_mode=None,
                temperature=1.0,
                is_primary_mode=False,
                requires_backbone_3d=True,
            )
        )
    for arm in DIMA_4_ARMS:
        specs.append(
            Exp41NewSpec(
                entry_key=f"{arm}|sde",
                arm_id=arm,
                backend="dima",
                mode="sde",
                conditioning_regime="unconditional_continuous_latent",
                paradigm_group="continuous_latent_diffusion",
                generation_regime="length_matched_continuous_sde",
                diffusion_mode="euler_sde",
                temperature=1.0,
                is_primary_mode=True,
                requires_backbone_3d=False,
            )
        )
    for arm in DIMA_ODE_ABLATION_ARMS:
        specs.append(
            Exp41NewSpec(
                entry_key=f"{arm}|ode",
                arm_id=arm,
                backend="dima",
                mode="ode",
                conditioning_regime="unconditional_continuous_latent",
                paradigm_group="continuous_latent_diffusion_ode",
                generation_regime="length_matched_continuous_ode",
                diffusion_mode="euler_ode",
                temperature=1.0,
                is_primary_mode=False,
                requires_backbone_3d=False,
            )
        )
    return tuple(specs)


def _build_r5a_specs() -> tuple[R5aNewSpec, ...]:
    specs: list[R5aNewSpec] = []
    for arm in DPLM2_4_ARMS:
        specs.append(
            R5aNewSpec(
                entry_key=f"{arm}|seq_only",
                arm_id=arm,
                backend="dplm2",
                mode="seq_only",
                conditioning_regime="sequence_scaffold_infill",
                temperature=1.0,
                requires_backbone_3d=False,
            )
        )
        specs.append(
            R5aNewSpec(
                entry_key=f"{arm}|struct_conditioned",
                arm_id=arm,
                backend="dplm2",
                mode="struct_conditioned",
                conditioning_regime="backbone_3d_scaffold_infill",
                temperature=1.0,
                requires_backbone_3d=True,
            )
        )
    for arm in PROTEINMPNN_11_ARMS:
        specs.append(
            R5aNewSpec(
                entry_key=arm,
                arm_id=arm,
                backend="proteinmpnn",
                mode="inverse_folding",
                conditioning_regime="backbone_3d_scaffold_infill",
                temperature=0.1,
                requires_backbone_3d=True,
            )
        )
    for arm in DIMA_4_ARMS:
        specs.append(
            R5aNewSpec(
                entry_key=arm,
                arm_id=arm,
                backend="dima",
                mode="sde",
                conditioning_regime="continuous_latent_scaffold_infill",
                temperature=1.0,
                requires_backbone_3d=False,
            )
        )
    return tuple(specs)


def _build_exp4_2_specs() -> tuple[Exp42NewSpec, ...]:
    specs: list[Exp42NewSpec] = []
    for arm in DPLM2_4_ARMS:
        specs.append(
            Exp42NewSpec(
                entry_key=f"{arm}|seq_only",
                arm_id=arm,
                backend="dplm2",
                mode="seq_only",
                conditioning_regime="unconditional_seq_only",
                arm_keys=exp42.PHASE_SPACE_ARM_KEYS,
                requires_backbone_3d=False,
            )
        )
        specs.append(
            Exp42NewSpec(
                entry_key=f"{arm}|co_generation",
                arm_id=arm,
                backend="dplm2",
                mode="co_generation",
                conditioning_regime="unconditional_co_generation",
                arm_keys=("N",),
                requires_backbone_3d=False,
            )
        )
        specs.append(
            Exp42NewSpec(
                entry_key=f"{arm}|struct_conditioned",
                arm_id=arm,
                backend="dplm2",
                mode="struct_conditioned",
                conditioning_regime="backbone_3d_inverse_folding",
                arm_keys=("N",),
                requires_backbone_3d=True,
            )
        )
    for arm in PROTEINMPNN_11_ARMS:
        specs.append(
            Exp42NewSpec(
                entry_key=arm,
                arm_id=arm,
                backend="proteinmpnn",
                mode="inverse_folding",
                conditioning_regime="backbone_3d_inverse_folding",
                arm_keys=("N", "N_T10"),
                requires_backbone_3d=True,
            )
        )
    for arm in DIMA_4_ARMS:
        specs.append(
            Exp42NewSpec(
                entry_key=arm,
                arm_id=arm,
                backend="dima",
                mode="sde",
                conditioning_regime="unconditional_continuous_latent",
                arm_keys=exp42.PHASE_SPACE_ARM_KEYS,
                requires_backbone_3d=False,
            )
        )
    return tuple(specs)


def _build_r4_cont_specs() -> tuple[R4ContNewSpec, ...]:
    specs: list[R4ContNewSpec] = []
    for arm in DPLM2_4_ARMS:
        specs.append(
            R4ContNewSpec(
                entry_key=f"{arm}|seq_only",
                arm_id=arm,
                backend="dplm2",
                mode="seq_only",
                context_training_regime="single_chain_pretrained",
                conditioning_regime="unaligned_homolog_prefix",
                requires_backbone_3d=False,
            )
        )
        specs.append(
            R4ContNewSpec(
                entry_key=f"{arm}|struct_conditioned",
                arm_id=arm,
                backend="dplm2",
                mode="struct_conditioned",
                context_training_regime="backbone_3d_wt_prefix_suffix",
                conditioning_regime="backbone_3d_wt_prefix_suffix",
                requires_backbone_3d=True,
            )
        )
    for arm in PROTEINMPNN_11_ARMS:
        specs.append(
            R4ContNewSpec(
                entry_key=arm,
                arm_id=arm,
                backend="proteinmpnn",
                mode="inverse_folding",
                context_training_regime="backbone_3d_wt_prefix_suffix",
                conditioning_regime="backbone_3d_wt_prefix_suffix",
                requires_backbone_3d=True,
            )
        )
    for arm in DIMA_4_ARMS:
        specs.append(
            R4ContNewSpec(
                entry_key=arm,
                arm_id=arm,
                backend="dima",
                mode="sde",
                context_training_regime="single_chain_pretrained",
                conditioning_regime="unaligned_homolog_prefix",
                requires_backbone_3d=False,
            )
        )
    return tuple(specs)


EXP4_1_NEW_SPECS: tuple[Exp41NewSpec, ...] = _build_exp4_1_specs()
EXP4_1_NEW_SPEC_MAP: Mapping[str, Exp41NewSpec] = {s.entry_key: s for s in EXP4_1_NEW_SPECS}
R5A_NEW_SPECS: tuple[R5aNewSpec, ...] = _build_r5a_specs()
R5A_NEW_SPEC_MAP: Mapping[str, R5aNewSpec] = {s.entry_key: s for s in R5A_NEW_SPECS}
EXP4_2_NEW_SPECS: tuple[Exp42NewSpec, ...] = _build_exp4_2_specs()
EXP4_2_NEW_SPEC_MAP: Mapping[str, Exp42NewSpec] = {s.entry_key: s for s in EXP4_2_NEW_SPECS}
R4_CONT_NEW_SPECS: tuple[R4ContNewSpec, ...] = _build_r4_cont_specs()
R4_CONT_NEW_SPEC_MAP: Mapping[str, R4ContNewSpec] = {s.entry_key: s for s in R4_CONT_NEW_SPECS}


# --------------------------------------------------------------------------- #
# Shared helpers: I/O, PDB parsing, reference cohorts, leak guard, bootstrap
# --------------------------------------------------------------------------- #


def seq_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def assert_no_infra_leaks(payload: Any, *, label: str = "artifact") -> None:
    """Fail fast if serialized JSON contains forbidden pod/node/IP patterns."""
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    hit = FORBIDDEN_INFRA_RE.search(text)
    if hit is not None:
        raise ValueError(f"{label}: forbidden infrastructure token detected: {hit.group(0)!r}")


def write_checked_artifact(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Check zero infra leaks, write JSON + .sha256 atomically, and ensure world-readable."""
    assert_no_infra_leaks(payload, label=path.name)
    digest = write_json_with_sha256(path, payload)
    sha_path = Path(f"{path}.sha256")
    for p in (path, sha_path):
        if p.exists():
            os.chmod(p, 0o644)
    return {"path": str(path), "sha256": digest, "sha256_path": str(sha_path)}


def parse_pdb_backbone_ncaco(pdb_text: str, expected_len: int) -> np.ndarray:
    """Parse ``(L, 4, 3)`` ``[N, CA, C, O]`` backbone coordinates from ESMFold2 PDB output."""
    atom_slot = {"N": 0, "CA": 1, "C": 2, "O": 3}
    residues: dict[int, np.ndarray] = {}
    res_order: list[int] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  "):
            continue
        aname = line[12:16].strip()
        slot = atom_slot.get(aname)
        if slot is None:
            continue
        try:
            resseq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        arr = residues.get(resseq)
        if arr is None:
            arr = np.full((4, 3), np.nan, dtype=np.float32)
            residues[resseq] = arr
            res_order.append(resseq)
        arr[slot] = (x, y, z)
    if len(res_order) != expected_len:
        raise ValueError(f"PDB backbone residue count {len(res_order)} != expected {expected_len}")
    coords = np.stack([residues[r] for r in res_order], axis=0).astype(np.float32)
    if not np.isfinite(coords[:, 1, :]).all():
        raise ValueError("PDB backbone missing finite CA coordinates")
    # Fill any rare missing O/N/C atom with CA so downstream adapters never see NaN.
    for slot in (0, 2, 3):
        bad = ~np.isfinite(coords[:, slot, :]).all(axis=-1)
        if bad.any():
            coords[bad, slot, :] = coords[bad, 1, :]
    return coords


def sequence_identity_exact(seq_a: str, seq_b: str) -> float:
    """Position-by-position percent identity (0..100) for equal-length sequences."""
    if not seq_a or len(seq_a) != len(seq_b):
        return 0.0
    matches = sum(1 for a, b in zip(seq_a, seq_b) if a == b)
    return float(100.0 * matches / len(seq_a))


def paired_bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Paired bootstrap 95% CI for ``mean(a - b)`` over finite paired units."""
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    if xa.shape != xb.shape:
        raise ValueError(f"Paired arrays shape mismatch: {xa.shape} vs {xb.shape}")
    mask = np.isfinite(xa) & np.isfinite(xb)
    diff = xa[mask] - xb[mask]
    n = int(diff.size)
    if n == 0:
        return {"n_units": 0, "mean_a": None, "mean_b": None, "delta": None, "ci95": None, "verdict": "unavailable"}
    mean_a = float(np.mean(xa[mask]))
    mean_b = float(np.mean(xb[mask]))
    delta = float(np.mean(diff))
    if n == 1:
        return {
            "n_units": 1,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta": delta,
            "ci95": [delta, delta],
            "verdict": "single_unit",
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    verdict = "positive" if lo > 0.0 else ("negative" if hi < 0.0 else "unresolved")
    return {
        "n_units": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": delta,
        "ci95": [float(lo), float(hi)],
        "verdict": verdict,
    }


def unpaired_bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Two-sample percentile bootstrap 95% CI for ``mean(a) - mean(b)``."""
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    na, nb = int(xa.size), int(xb.size)
    if na == 0 or nb == 0:
        return {"n_a": na, "n_b": nb, "mean_a": None, "mean_b": None, "delta": None, "ci95": None, "verdict": "unavailable"}
    mean_a, mean_b = float(xa.mean()), float(xb.mean())
    delta = mean_a - mean_b
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, na, size=(draws, na))
    ib = rng.integers(0, nb, size=(draws, nb))
    boot = xa[ia].mean(axis=1) - xb[ib].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    verdict = "positive" if lo > 0.0 else ("negative" if hi < 0.0 else "unresolved")
    return {
        "n_a": na,
        "n_b": nb,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": delta,
        "ci95": [float(lo), float(hi)],
        "verdict": verdict,
    }


def select_exp4_1_reference_targets(
    repo_root: Path = REPO_ROOT,
    *,
    n_targets: int = EXP4_1_POOL_SIZE,
) -> list[dict[str, Any]]:
    """Deterministically select ``n_targets`` natural reference templates stratified by length bin."""
    src = Path(repo_root) / NATURAL_FOLD_400_REL
    rows = read_jsonl(src)
    if not rows:
        raise FileNotFoundError(f"Missing natural reference templates: {src}")
    valid = [
        {
            "seq_id": str(r.get("seq_id") or r.get("id") or f"nat_{idx:04d}"),
            "sequence": str(r["sequence"]).strip().upper(),
            "sequence_sha256": seq_sha256(str(r["sequence"]).strip().upper()),
            "length": len(str(r["sequence"]).strip()),
        }
        for idx, r in enumerate(rows)
        if exp41.is_canonical_aa20(str(r.get("sequence") or "").strip().upper())
    ]
    valid.sort(key=lambda r: (r["length"], r["seq_id"]))
    if n_targets >= len(valid):
        return valid
    # Uniformly stride across the length-sorted natural pool so all 3 length bins stay balanced.
    indices = np.linspace(0, len(valid) - 1, num=n_targets, dtype=np.int64)
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i in indices.tolist():
        while i in seen and i + 1 < len(valid):
            i += 1
        seen.add(i)
        chosen.append(valid[i])
    return chosen


def load_r5a_ladder_backbones(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Load the 16 canonical R5a parent backbones and exact window spans."""
    path = Path(repo_root) / LADDER_BACKBONES_REL
    rows = read_jsonl(path)
    if len(rows) != 16:
        raise ValueError(f"Expected 16 R5a ladder backbones in {path}, got {len(rows)}")
    out: list[dict[str, Any]] = []
    for r in rows:
        seq = str(r["sequence"]).strip().upper()
        out.append(
            {
                "backbone_id": str(r["backbone_id"]),
                "accession": str(r.get("accession") or ""),
                "sequence": seq,
                "sequence_sha256": str(r.get("parent_sequence_sha256") or seq_sha256(seq)),
                "length": len(seq),
                "parent_mean_ca_plddt": float(r.get("parent_mean_ca_plddt") or 0.0),
                "parent_ptm": float(r.get("parent_ptm") or 0.0),
                "window_spans": {
                    str(k): [int(v[0]), int(v[1])] for k, v in (r.get("window_spans") or {}).items()
                },
            }
        )
    return out


def deterministic_seed(*parts: Any, base_seed: int = DEFAULT_SEED) -> int:
    tag = "|".join([str(base_seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = int(hashlib.sha256(tag).hexdigest()[:12], 16)
    return int((digest % 1_900_000_000) + 10_000)


# --------------------------------------------------------------------------- #
# DPLM-2 Exact Guided Potts Sampling (Exp 4.2) & Homolog Continuation (R4)
# --------------------------------------------------------------------------- #


def sample_guided_dplm2(
    *,
    adapter: Any,
    model_id: str,
    family_model: Any,
    arm_spec: exp42.PhaseSpaceArmSpec,
    count: int,
    batch_size: int,
    stream_seed: int,
    mode: str = "seq_only",
    coords: np.ndarray | None = None,
    permutation_seed: int = exp42.PERMUTATION_SEED,
) -> list[dict[str, Any]]:
    """Exact 1-by-1 ``uniform_random`` Potts-guided sampler for ``DPLM2Adapter``.

    Preserves the exact ``sequence_component_sums`` invariant (``< 1e-6`` tolerance)
    across all five Exp 4.2 arms (``N, Pr, Pr_strong, C, Cx``) in either ``seq_only``
    or ``struct_conditioned`` mode.
    """
    import torch

    from src.capability.unified.dplm2_adapter import AA_MASK_ID, STRUCT_MASK_ID

    bias = exp42.build_phase_space_bias(family_model, arm_spec, permutation_seed=permutation_seed)
    L = int(family_model.length)
    single_mod = mode == "seq_only"
    if not single_mod and coords is None:
        raise ValueError(f"{model_id}: struct_conditioned mode requires WT backbone coords")
    st_ids_base = (
        [STRUCT_MASK_ID] * L
        if single_mod
        else list(adapter.encode_backbone(np.asarray(coords, dtype=np.float32)))
    )
    aa_ids_base = [AA_MASK_ID] * L
    aa20_ids = [int(x) for x in adapter.aa20_ids_t.tolist()]
    device = adapter.device
    records: list[dict[str, Any]] = []

    with torch.no_grad(), adapter._precision_ctx():
        for start in range(0, count, max(1, int(batch_size))):
            chunk_n = min(int(batch_size), count - start)
            seeds = [int(stream_seed) + start + i for i in range(chunk_n)]
            gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
            orders = [torch.randperm(L, generator=gens[row]).tolist() for row in range(chunk_n)]
            paired, half = adapter._build_paired_ids(
                [list(aa_ids_base) for _ in range(chunk_n)],
                [list(st_ids_base) for _ in range(chunk_n)],
            )
            committed_pos: list[list[int]] = [[] for _ in range(chunk_n)]
            committed_aa: list[list[int]] = [[] for _ in range(chunk_n)]
            final_codes = np.full((chunk_n, L), -1, dtype=np.int64)
            entropies = np.zeros((chunk_n, L), dtype=np.float64)
            applied = np.zeros(chunk_n, dtype=np.float64)

            for step in range(L):
                net_out = adapter._forward_paired(paired, single_modality=single_mod)
                aa_logits_all = adapter._extract_aa_logits(net_out, half)  # (B, L+2, V)
                step_pos = [orders[row][step] for row in range(chunk_n)]
                row_idx = torch.arange(chunk_n, dtype=torch.long, device=device)
                tok_idx = torch.tensor([1 + p for p in step_pos], dtype=torch.long, device=device)
                aa_logits = aa_logits_all[row_idx, tok_idx].index_select(-1, adapter.aa20_ids_t).float()

                bias_mat = np.zeros((chunk_n, 20), dtype=np.float64)
                for row in range(chunk_n):
                    bias_mat[row] = exp42._diffusion_position_bias(
                        bias,
                        arm_spec.strength,
                        step_pos[row],
                        committed_pos[row],
                        committed_aa[row],
                    )
                scores = aa_logits + torch.as_tensor(
                    bias_mat.astype(np.float32), dtype=torch.float32, device=device
                )
                probs = torch.softmax(scores, dim=-1).cpu()
                for row in range(chunk_n):
                    pos = step_pos[row]
                    aa_idx = int(torch.multinomial(probs[row], 1, generator=gens[row]).item())
                    paired[row, half + 1 + pos] = aa20_ids[aa_idx]
                    committed_pos[row].append(pos)
                    committed_aa[row].append(aa_idx)
                    final_codes[row, pos] = aa_idx
                    applied[row] += float(bias_mat[row, aa_idx])
                    entropies[row, step] = exp42._entropy_from_logits_1d(scores[row])

            for row in range(chunk_n):
                seq = "".join(AA20[int(c)] for c in final_codes[row])
                expected, f_sum, c_sum = exp42._verify_applied_bias(
                    bias,
                    arm_spec.strength,
                    seq,
                    float(applied[row]),
                    label=f"{model_id}/{mode}/{arm_spec.arm_key}",
                )
                records.append(
                    exp42._make_generated_record(
                        model_id=model_id,
                        domain=family_model.domain,
                        arm_spec=arm_spec,
                        seed=seeds[row],
                        sequence_index=start + row,
                        sequence=seq,
                        applied_bias=expected,
                        fields_score=f_sum if bias.has_fields else None,
                        coupling_score=c_sum if bias.has_coupling else None,
                        mean_step_entropy=float(entropies[row].mean()),
                        sampler=f"dplm2_{mode}_uniform_random_guided",
                        token_steps=L,
                    )
                )
    return records


def sample_dplm2_homolog_continuation(
    *,
    adapter: Any,
    context_items: Sequence[str],
    gen_length: int,
    n_samples: int,
    seed: int,
    budget_tokens: int = r4c.BUDGET_TOKENS,
    steps: int = r4c.DIFFUSION_MAX_STEPS,
    batch_size: int = 8,
    temperature: float = 1.0,
) -> tuple[list[str], int, int]:
    """Unaligned homolog-prefix continuation for ``DPLM2Adapter`` (``seq_only``, ``uniform_random``).

    Packs ``[item_1, item_2, ..., <mask>^{gen_length}]`` inside ``budget_tokens`` (matching
    ``DPLM-1`` in ``r4_continuation_yardstick.py``) while holding all prefix context tokens
    strictly fixed and unmasking only the ``gen_length`` continuation span under
    ``uniform_random`` schedule.
    """
    import torch

    from src.capability.unified.dplm2_adapter import (
        AA_MASK_ID,
        STRUCT_MASK_ID,
        _encode_aa_to_ids,
    )

    L_gen = max(r4c.MIN_PRODUCT_RESIDUES, int(gen_length))
    max_ctx_res = max(0, int(budget_tokens) - L_gen - 4)
    ctx_used: list[str] = []
    used_res = 0
    for item in context_items:
        seq = str(item).strip().upper()
        if not seq:
            continue
        if used_res + len(seq) <= max_ctx_res:
            ctx_used.append(seq)
            used_res += len(seq)
        else:
            rem = max_ctx_res - used_res
            if rem >= r4c.MIN_PRODUCT_RESIDUES:
                ctx_used.append(seq[:rem])
                used_res += rem
            break

    ctx_concat = "".join(ctx_used)
    ctx_ids = _encode_aa_to_ids(ctx_concat) if ctx_concat else []
    total_len = len(ctx_ids) + L_gen
    gen_start = len(ctx_ids)
    aa_base = ctx_ids + [AA_MASK_ID] * L_gen
    st_base = [STRUCT_MASK_ID] * total_len
    aa20_ids = [int(x) for x in adapter.aa20_ids_t.tolist()]
    n_steps = max(1, min(int(steps), L_gen))
    device = adapter.device
    sequences: list[str] = []

    with torch.no_grad(), adapter._precision_ctx():
        for start in range(0, n_samples, max(1, int(batch_size))):
            cur_b = min(int(batch_size), n_samples - start)
            seeds = [int(seed) + start + i for i in range(cur_b)]
            gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
            orders = [torch.randperm(L_gen, generator=gens[i]).tolist() for i in range(cur_b)]
            paired, half = adapter._build_paired_ids(
                [list(aa_base) for _ in range(cur_b)],
                [list(st_base) for _ in range(cur_b)],
            )
            cursor = [0] * cur_b
            final_codes = np.zeros((cur_b, L_gen), dtype=np.int64)

            for step in range(n_steps):
                frac_done = float(step + 1) / float(n_steps)
                target_done = L_gen if step == n_steps - 1 else min(L_gen, max(cursor[0] + 1, int(round(L_gen * frac_done))))
                if all(c >= L_gen for c in cursor):
                    break
                net_out = adapter._forward_paired(paired, single_modality=True)
                aa_logits_all = adapter._extract_aa_logits(net_out, half)  # (B, total_len+2, V)
                for row in range(cur_b):
                    c0 = cursor[row]
                    c1 = min(L_gen, target_done)
                    if c1 <= c0:
                        continue
                    pos_list = orders[row][c0:c1]
                    tok_idx = torch.tensor(
                        [1 + gen_start + p for p in pos_list], dtype=torch.long, device=device
                    )
                    logits_sub = aa_logits_all[row].index_select(0, tok_idx).index_select(-1, adapter.aa20_ids_t).float()
                    if temperature <= 1e-6:
                        chosen = logits_sub.argmax(dim=-1).cpu().tolist()
                    else:
                        probs = torch.softmax(logits_sub / float(temperature), dim=-1).cpu()
                        chosen = torch.multinomial(probs, num_samples=1, generator=gens[row]).squeeze(-1).tolist()
                    for p, aa_idx in zip(pos_list, chosen):
                        paired[row, half + 1 + gen_start + p] = aa20_ids[int(aa_idx)]
                        final_codes[row, p] = int(aa_idx)
                    cursor[row] = c1

            for row in range(cur_b):
                sequences.append("".join(AA20[int(c)] for c in final_codes[row]))

    return sequences, len(ctx_used), used_res


# --------------------------------------------------------------------------- #
# DiMA Continuous Latent Diffusion Sampling (Exp 4.2 Guided & R4 Continuation)
# --------------------------------------------------------------------------- #


def sample_guided_dima(
    *,
    adapter: Any,
    model_id: str,
    family_model: Any,
    arm_spec: exp42.PhaseSpaceArmSpec,
    count: int,
    batch_size: int,
    stream_seed: int,
    steps: int = 250,
    ode_sampling: bool = False,
    permutation_seed: int = exp42.PERMUTATION_SEED,
) -> list[dict[str, Any]]:
    """Batched ``Exp 4.2`` continuous-latent sampler for ``DiMAAdapter`` across ``N/Pr/Pr_strong/C/Cx``.

    * ``N``: unguided Euler SDE/ODE reverse diffusion (``mean_applied_bias_norm == 0.0``).
    * ``Pr / Pr_strong``: differentiable decoder log-partition profile guidance via
      ``profile_log_bias = lambda_fields * bias.u``.
    * ``C / Cx``: differentiable soft Potts pairwise coupling guidance on ``p = softmax(lp20(x_0_hat))``,
      maximizing ``0.5 * lambda_coupling * sum_{i!=j} p_i^T C_{ij} p_j`` via ``torch.autograd.grad``.
    """
    import torch

    bias = exp42.build_phase_space_bias(family_model, arm_spec, permutation_seed=permutation_seed)
    L = int(family_model.length)
    has_fields = bias.u is not None and arm_spec.strength.fields != 0.0
    has_coupling = bias.C is not None and arm_spec.strength.coupling != 0.0
    records: list[dict[str, Any]] = []

    u_t = (
        torch.as_tensor(
            (float(arm_spec.strength.fields) * bias.u).astype(np.float32),
            device=adapter.device,
            dtype=torch.float32,
        )
        if has_fields
        else None
    )
    c_t = (
        torch.as_tensor(
            (float(arm_spec.strength.coupling) * bias.C).astype(np.float32),
            device=adapter.device,
            dtype=torch.float32,
        )
        if has_coupling
        else None
    )

    for start in range(0, count, max(1, int(batch_size))):
        chunk_n = min(int(batch_size), count - start)
        chunk_seed = int(stream_seed) + start

        if not has_fields and not has_coupling:
            with torch.no_grad():
                x_mean, attn_mask, bias_norm = adapter._run_reverse_diffusion(
                    [L] * chunk_n,
                    steps=steps,
                    ode_sampling=ode_sampling,
                    seed=chunk_seed,
                )
        elif has_fields and not has_coupling:
            x_mean, attn_mask, bias_norm = adapter._run_reverse_diffusion(
                [L] * chunk_n,
                steps=steps,
                ode_sampling=ode_sampling,
                seed=chunk_seed,
                profile_log_bias=(float(arm_spec.strength.fields) * bias.u).astype(np.float32),
                guidance_scale=1.0,
            )
        else:
            # Differentiable soft Potts (fields + pairwise couplings) Euler reverse loop
            total_t = L + (2 if adapter.spec.has_bos_eos else 0)
            dim = adapter.spec.embedding_dim
            off = adapter.res_offset
            attn_mask = torch.ones((chunk_n, total_t), device=adapter.device, dtype=torch.long)
            gen = torch.Generator(device=adapter.device).manual_seed(chunk_seed)
            x = torch.randn((chunk_n, total_t, dim), generator=gen, device=adapter.device, dtype=adapter.torch_dtype)
            x_0_sc = torch.zeros_like(x)
            n_steps = max(1, int(steps))
            timesteps = torch.linspace(adapter.dynamic.T, 0.05, n_steps + 1, device=adapter.device)
            valid_res = torch.zeros((chunk_n, total_t), device=adapter.device, dtype=torch.bool)
            valid_res[:, off : off + L] = True
            bias_norm_acc = 0.0
            x_mean = x
            with adapter._precision_ctx():
                for idx in range(n_steps):
                    t_cur = torch.full((chunk_n,), float(timesteps[idx].item()), device=adapter.device, dtype=torch.float32)
                    t_nxt = torch.full((chunk_n,), float(timesteps[idx + 1].item()), device=adapter.device, dtype=torch.float32)
                    dt = (t_nxt - t_cur).view(-1, 1, 1).to(dtype=x.dtype)
                    with torch.no_grad():
                        score_out = adapter._calc_score(x, t_cur, attn_mask, x_0_sc)
                        score = score_out["score"]
                        x_0_hat = score_out["x_0"]
                    with torch.enable_grad():
                        x0_g = x_0_hat.detach().float().requires_grad_(True)
                        lp20 = adapter.decode_aa20_logprobs(
                            x0_g.to(dtype=adapter.torch_dtype), attention_mask=attn_mask
                        )[:, off : off + L, :]
                        probs = lp20.exp()
                        pot = torch.zeros((), device=adapter.device, dtype=torch.float32)
                        if u_t is not None:
                            pot = pot + (probs * u_t.unsqueeze(0)).sum()
                        if c_t is not None:
                            # 0.5 * sum_{b,i,j,a,c} p_{b,i,a} C_{i,j,a,c} p_{b,j,c}
                            pot = pot + 0.5 * torch.einsum("bia,ijac,bjc->", probs, c_t, probs)
                        grad_x0 = torch.autograd.grad(pot, x0_g)[0]
                    grad_x0 = grad_x0 * valid_res.unsqueeze(-1).to(dtype=grad_x0.dtype)
                    gnorm = grad_x0.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    grad_clipped = grad_x0 * torch.clamp(1.0 / gnorm, max=1.0)
                    _, std_cur = adapter.dynamic.marginal_params(t_cur)
                    delta_score = (grad_clipped / std_cur.to(device=adapter.device).clamp(min=0.05)).to(dtype=score.dtype)
                    score = score + delta_score
                    bias_norm_acc += float(delta_score.float().norm(dim=-1)[valid_res].mean().item())
                    with torch.no_grad():
                        drift, diffusion = adapter.dynamic.reverse_params(
                            x, t_cur, score, ode_sampling=ode_sampling
                        )
                        x_mean = x + drift * dt
                        if ode_sampling:
                            x = x_mean
                        else:
                            noise = torch.randn(x.shape, generator=gen, device=adapter.device, dtype=x.dtype)
                            x = x_mean + diffusion.view(-1, 1, 1) * torch.sqrt(-dt) * noise
                        x_0_sc = x_0_hat.detach()
            bias_norm = bias_norm_acc / float(n_steps)

        with torch.no_grad():
            seqs, threedis, _, ents = adapter.decode_sequences_from_latents(
                x_mean, [L] * chunk_n, attention_mask=attn_mask
            )
        for i, seq in enumerate(seqs):
            f_sum, c_sum = exp42.gd.sequence_component_sums(bias, seq)
            rec = exp42._make_generated_record(
                model_id=model_id,
                domain=family_model.domain,
                arm_spec=arm_spec,
                seed=chunk_seed + i,
                sequence_index=start + i,
                sequence=seq,
                applied_bias=float(bias_norm),
                fields_score=float(f_sum) if bias.has_fields else None,
                coupling_score=float(c_sum) if bias.has_coupling else None,
                mean_step_entropy=float(ents[i]),
                sampler=f"dima_{'ode' if ode_sampling else 'sde'}_guided",
                token_steps=int(steps),
            )
            if threedis[i] is not None:
                rec["threedi_sequence"] = threedis[i]
            records.append(rec)
    return records


def sample_dima_homolog_continuation(
    *,
    adapter: Any,
    context_items: Sequence[str],
    gen_length: int,
    n_samples: int,
    seed: int,
    budget_tokens: int = DIMA_MAX_RESIDUES,
    steps: int = 250,
    batch_size: int = 16,
    ode_sampling: bool = False,
) -> tuple[list[str], int, int]:
    """Batched RePaint unaligned homolog-prefix continuation for ``DiMAAdapter`` (``<=510`` aa)."""
    import torch

    eff_budget = min(int(budget_tokens), int(adapter.max_residues))
    L_gen = max(r4c.MIN_PRODUCT_RESIDUES, min(int(gen_length), eff_budget))
    max_ctx_res = max(0, eff_budget - L_gen)
    ctx_used: list[str] = []
    used_res = 0
    for item in context_items:
        seq = str(item).strip().upper()
        if not seq:
            continue
        if used_res + len(seq) <= max_ctx_res:
            ctx_used.append(seq)
            used_res += len(seq)
        else:
            rem = max_ctx_res - used_res
            if rem >= r4c.MIN_PRODUCT_RESIDUES:
                ctx_used.append(seq[:rem])
                used_res += rem
            break

    ctx_concat = "".join(ctx_used)
    sequences: list[str] = []
    off = adapter.res_offset

    with torch.no_grad():
        if not ctx_concat:
            for start in range(0, n_samples, max(1, int(batch_size))):
                cur_b = min(int(batch_size), n_samples - start)
                res_list = adapter.generate_unconditional(
                    L_gen,
                    n_samples=cur_b,
                    steps=steps,
                    ode_sampling=ode_sampling,
                    seed=int(seed) + start,
                )
                sequences.extend(str(r["sequence"]).strip().upper() for r in res_list)
            return sequences, 0, 0

        total_len = len(ctx_concat) + L_gen
        placeholder = ctx_concat + ("A" * L_gen)
        suffix_set = set(range(len(ctx_concat), total_len))
        z_ctx_1, attn_1 = adapter.encode_latents([placeholder], mask_positions_list=[suffix_set])
        ctx_mask_1 = torch.zeros_like(attn_1, dtype=torch.bool)
        if adapter.spec.has_bos_eos:
            ctx_mask_1[0, 0] = True
            ctx_mask_1[0, off + total_len] = True
        ctx_mask_1[0, off : off + len(ctx_concat)] = True

        for start in range(0, n_samples, max(1, int(batch_size))):
            cur_b = min(int(batch_size), n_samples - start)
            x_mean, attn_b, _ = adapter._run_reverse_diffusion(
                [total_len] * cur_b,
                steps=steps,
                ode_sampling=ode_sampling,
                seed=int(seed) + start,
                z_ctx=z_ctx_1.expand(cur_b, -1, -1),
                ctx_mask=ctx_mask_1.expand(cur_b, -1),
            )
            decoded, _, _, _ = adapter.decode_sequences_from_latents(
                x_mean, [total_len] * cur_b, attention_mask=attn_b
            )
            for full_s in decoded:
                sequences.append(full_s[len(ctx_concat) : total_len])

    return sequences, len(ctx_used), used_res


# --------------------------------------------------------------------------- #
# 1. Experiment 4.1 Full Assembler (34 sealed + 30 new DPLM-2/ProteinMPNN)
# --------------------------------------------------------------------------- #


def _load_new_exp4_1_entry(
    spec: Exp41NewSpec,
    work_dir: Path,
    *,
    require_structures: bool = True,
) -> dict[str, Any] | None:
    gen_path = work_dir / "exp4_1" / "generated" / f"{spec.file_slug}.jsonl"
    fold_path = work_dir / "exp4_1" / "folded" / f"{spec.file_slug}.jsonl"
    if not gen_path.is_file():
        if require_structures:
            raise FileNotFoundError(f"Missing Exp 4.1 generated file: {gen_path}")
        return None
    if require_structures and not fold_path.is_file():
        raise FileNotFoundError(f"Missing Exp 4.1 folded file: {fold_path}")

    fold_by_id: dict[str, dict[str, Any]] = {}
    fold_by_sha: dict[str, dict[str, Any]] = {}
    for frow in read_jsonl(fold_path):
        parsed = exp41._parse_fold_row(frow)
        if parsed is None:
            continue
        parsed["backbone_sc_tm"] = (
            float(frow["backbone_sc_tm"]) if frow.get("backbone_sc_tm") is not None else None
        )
        sid = str(frow.get("seq_id") or "")
        if sid:
            fold_by_id[sid] = parsed
        sha = str(frow.get("sequence_sha256") or "")
        if sha:
            fold_by_sha[sha] = parsed

    rows: list[dict[str, Any]] = []
    for idx, r in enumerate(read_jsonl(gen_path)):
        seq = str(r.get("sequence") or "").strip().upper()
        sha = str(r.get("sequence_sha256") or (seq_sha256(seq) if seq else ""))
        rid = str(r.get("seq_id") or f"{spec.file_slug}_{idx:04d}")
        valid = exp41.is_canonical_aa20(seq)
        supported = valid and len(seq) >= 16
        st = fold_by_id.get(rid) or (fold_by_sha.get(sha) if sha else None)
        rows.append(
            {
                "seq_id": rid,
                "sequence": seq,
                "sequence_sha256": sha,
                "length": len(seq),
                "valid_aa20": valid,
                "supported": supported,
                "budget_censored": False,
                "cached_any_profile": None,
                "structure": st,
                "template_sequence_recovery": (
                    float(r["template_sequence_recovery"])
                    if r.get("template_sequence_recovery") is not None
                    else None
                ),
                "template_id": r.get("template_id"),
            }
        )

    pool_spec = exp41.PoolEntrySpec(
        entry_key=spec.entry_key,
        arm_id=spec.arm_id,
        source_kind="dplm2_mpnn_h200",
        paradigm_group=spec.paradigm_group,
        generation_regime=spec.generation_regime,
        diffusion_mode=spec.diffusion_mode,
        is_control=False,
        is_primary_mode=spec.is_primary_mode,
        params_m=ARM_METADATA[spec.arm_id].params_m,
    )
    return {
        "spec": pool_spec,
        "new_spec": spec,
        "rows": rows,
        "source_paths": [str(gen_path), *( [str(fold_path)] if fold_path.is_file() else [] )],
    }


def assemble_exp4_1_full(
    repo_root: Path = REPO_ROOT,
    work_dir: Path | None = None,
    *,
    allow_partial: bool = False,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
    diamond_threads: int = 32,
    hmmscan_workers: int = 16,
    skip_external_oracles: bool = False,
) -> dict[str, Any]:
    """Assemble ``r5_experiment_4_1_unified_generation_full.json`` (sealed 34 + 30 new)."""
    root = Path(repo_root)
    wdir = Path(work_dir) if work_dir is not None else (root / WORK_DIR_REL)
    sealed_path = root / UNIFIED_OUTPUT_DIR / "r5_experiment_4_1_unified_generation.json"
    sealed = load_json_file(sealed_path)

    loaded_new: list[dict[str, Any]] = []
    missing_entries: list[str] = []
    for spec in EXP4_1_NEW_SPECS:
        item = _load_new_exp4_1_entry(spec, wdir, require_structures=not allow_partial)
        if item is None:
            missing_entries.append(spec.entry_key)
        else:
            loaded_new.append(item)

    unique_seqs: dict[str, str] = {}
    for item in loaded_new:
        for r in item["rows"]:
            if r["valid_aa20"] and r["sequence_sha256"]:
                unique_seqs[r["sequence_sha256"]] = r["sequence"]

    if skip_external_oracles:
        dm_cache: dict[str, dict[str, Any]] = {}
        pf_cache: dict[str, dict[str, Any]] = {}
    else:
        dm_cache = exp41.ensure_diamond_annotations(unique_seqs, root, threads=diamond_threads)
        pf_cache = exp41.ensure_pfam_annotations(
            unique_seqs, loaded_new, root, workers=hmmscan_workers, threads_per_worker=2
        )

    # Preserve all 34 sealed entries verbatim (annotate conditioning_regime explicitly on copy).
    entries_all: dict[str, Any] = {}
    for ek, entry in (sealed.get("entries_full_34") or {}).items():
        copied = dict(entry)
        copied["conditioning_regime"] = (
            "control_reference" if copied.get("is_control") else "unconditional_seq_only"
        )
        entries_all[ek] = copied

    new_entries_out: dict[str, Any] = {}
    for idx, item in enumerate(loaded_new):
        nspec: Exp41NewSpec = item["new_spec"]
        agg = exp41._aggregate_one_entry(
            item,
            dm_cache,
            pf_cache,
            {},
            draws=draws,
            seed=seed + idx * 100,
        )
        agg["conditioning_regime"] = nspec.conditioning_regime
        agg["mode"] = nspec.mode
        agg["temperature"] = nspec.temperature
        agg["requires_backbone_3d"] = nspec.requires_backbone_3d
        # Inverse-folding specific summaries (template sequence recovery & self-consistency TM)
        recs = [
            float(r["template_sequence_recovery"])
            for r in item["rows"]
            if r.get("template_sequence_recovery") is not None
        ]
        sc_tms = [
            float(r["structure"]["backbone_sc_tm"])
            for r in item["rows"]
            if r.get("structure") and r["structure"].get("backbone_sc_tm") is not None
        ]
        agg["inverse_folding_metrics"] = {
            "template_sequence_recovery_pct": (
                exp41.bootstrap_mean_ci95(recs, draws=draws, seed=seed + idx * 100 + 71)
                if recs
                else None
            ),
            "backbone_sc_tm": (
                exp41.bootstrap_mean_ci95(sc_tms, draws=draws, seed=seed + idx * 100 + 72)
                if sc_tms
                else None
            ),
            "sc_tm_ge_0_5_rate": (
                float(np.mean([1.0 if v >= 0.5 else 0.0 for v in sc_tms])) if sc_tms else None
            ),
        }
        entries_all[nspec.entry_key] = agg
        new_entries_out[nspec.entry_key] = agg

    def _stage3_row(e: Mapping[str, Any]) -> dict[str, Any]:
        ax1 = e["axis1_full_pool"]
        ax2 = e["axis2_structure_joined"]
        if_m = e.get("inverse_folding_metrics") or {}
        return {
            "entry_key": e["entry_key"],
            "arm_id": e["arm_id"],
            "paradigm_group": e["paradigm_group"],
            "conditioning_regime": e.get("conditioning_regime"),
            "generation_regime": e["generation_regime"],
            "params_m": e["params_m"],
            "is_control": e["is_control"],
            "is_primary_mode": e["is_primary_mode"],
            "n_folded": ax2["n_folded"],
            "unfiltered_mean_plddt": ax2["all_folded"]["mean_plddt"],
            "unfiltered_foldable_ge70": ax2["all_folded"]["foldable_rate_plddt_ge_70"],
            "unfiltered_mean_ptm": ax2["all_folded"]["mean_ptm"],
            "uniref50_frac_lt30_novel": ax1["uniref50_frac_lt30_novel"],
            "mean_entropy_bits": ax1["mean_entropy_bits"],
            "pfam_complete_domain_rate_all": ax1["pfam_complete_domain_rate_all"],
            "gated_lt30_h30_n": ax2["gated_lt30_h_ge_3_0"]["n"],
            "gated_lt30_h30_yield_of_pool": ax2["gated_lt30_h_ge_3_0"]["gated_yield_of_pool"],
            "gated_lt30_h30_mean_plddt": ax2["gated_lt30_h_ge_3_0"]["mean_plddt"],
            "template_sequence_recovery_pct": if_m.get("template_sequence_recovery_pct"),
            "backbone_sc_tm": if_m.get("backbone_sc_tm"),
        }

    def _sort_yield(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda r: -(
                r["gated_lt30_h30_yield_of_pool"]["mean"]
                if r["gated_lt30_h30_yield_of_pool"]["mean"] is not None
                else -1.0
            ),
        )

    uncond_rows = _sort_yield(
        [
            _stage3_row(e)
            for e in entries_all.values()
            if e.get("conditioning_regime")
            in (
                "control_reference",
                "unconditional_seq_only",
                "unconditional_co_generation",
                "unconditional_continuous_latent",
            )
        ]
    )
    invfold_rows = _sort_yield(
        [
            _stage3_row(e)
            for e in entries_all.values()
            if e.get("conditioning_regime") == "backbone_3d_inverse_folding"
        ]
    )

    # Stage 2 extension: DPLM-2 max_confidence vs uniform_random vs co_generation
    dplm2_ablation: list[dict[str, Any]] = []
    for arm in DPLM2_4_ARMS:
        u = entries_all.get(f"{arm}|seq_only|uniform_random")
        m = entries_all.get(f"{arm}|seq_only|max_confidence")
        cg = entries_all.get(f"{arm}|co_generation")
        sc = entries_all.get(f"{arm}|struct_conditioned")
        if any(x is not None for x in (u, m, cg, sc)):
            def _brief(ent: Mapping[str, Any] | None) -> dict[str, Any] | None:
                if ent is None:
                    return None
                return {
                    "mean_entropy_bits": ent["axis1_full_pool"]["mean_entropy_bits"],
                    "uniref50_frac_lt30_novel": ent["axis1_full_pool"]["uniref50_frac_lt30_novel"],
                    "pfam_complete_domain_rate_all": ent["axis1_full_pool"]["pfam_complete_domain_rate_all"],
                    "unfiltered_mean_plddt": ent["axis2_structure_joined"]["all_folded"]["mean_plddt"],
                    "gated_lt30_h30_yield_of_pool": ent["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["gated_yield_of_pool"],
                }
            dplm2_ablation.append(
                {
                    "arm_id": arm,
                    "params_m": ARM_METADATA[arm].params_m,
                    "seq_only_uniform_random": _brief(u),
                    "seq_only_max_confidence": _brief(m),
                    "co_generation": _brief(cg),
                    "struct_conditioned": _brief(sc),
                }
            )

    return {
        "schema": "unified_experiment_4_1_generation_full_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_baseline_source": {
            "path": str(sealed_path.relative_to(root)),
            "sha256": sha256_file(sealed_path),
            "n_sealed_entries": len(sealed.get("entries_full_34") or {}),
        },
        "coverage": {
            "n_sealed_entries": len(sealed.get("entries_full_34") or {}),
            "n_new_expected": len(EXP4_1_NEW_SPECS),
            "n_new_loaded": len(new_entries_out),
            "missing_entries": missing_entries,
            "is_complete": len(missing_entries) == 0,
        },
        "conditioning_separation_policy": (
            "Unconditional sequence-only and unconditional co-generation pools are ranked "
            "strictly separately from 3D-backbone-conditioned inverse-folding pools "
            "(ProteinMPNN and DPLM-2 struct_conditioned)."
        ),
        "unconditional_ladder_ranked": uncond_rows,
        "backbone_3d_inverse_folding_ladder_ranked": invfold_rows,
        "dplm2_decoding_and_modality_ablation": dplm2_ablation,
        "sealed_stage2_dplm1_ablation": sealed.get("stage2_decoding_artifact_ablation"),
        "sealed_stage4_anchored_controls": sealed.get("stage4_anchored_vs_unanchored_dual_controls"),
        "entries_new_dplm2_mpnn": new_entries_out,
        "entries_all": entries_all,
    }


# --------------------------------------------------------------------------- #
# 2. R5a Local Infilling Full Assembler (18 sealed + 19 new DPLM-2/ProteinMPNN)
# --------------------------------------------------------------------------- #


def assemble_r5a_full(
    repo_root: Path = REPO_ROOT,
    work_dir: Path | None = None,
    *,
    allow_partial: bool = False,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Assemble ``r5a_local_infilling_full.json`` (sealed 18 arms + 19 new arm-modes)."""
    root = Path(repo_root)
    wdir = Path(work_dir) if work_dir is not None else (root / WORK_DIR_REL)
    sealed_path = root / UNIFIED_OUTPUT_DIR / "r5_experiment_4_1_unified_generation.json"
    sealed = load_json_file(sealed_path)
    sealed_r5a = (
        (sealed.get("stage4_anchored_vs_unanchored_dual_controls") or {}).get(
            "progressive_scaffold_anchoring_r5a"
        )
        or {}
    )

    new_entries: dict[str, Any] = {}
    missing_entries: list[str] = []

    for idx, spec in enumerate(R5A_NEW_SPECS):
        gen_path = wdir / "r5a" / "generated" / f"{spec.file_slug}.jsonl"
        fold_path = wdir / "r5a" / "folded" / f"{spec.file_slug}.jsonl"
        if not gen_path.is_file() or (not allow_partial and not fold_path.is_file()):
            if not allow_partial:
                raise FileNotFoundError(
                    f"Missing R5a artifacts for {spec.entry_key}: {gen_path} / {fold_path}"
                )
            missing_entries.append(spec.entry_key)
            continue

        fold_by_id = {str(r["seq_id"]): r for r in read_jsonl(fold_path)}
        by_rung: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in read_jsonl(gen_path):
            sid = str(r["seq_id"])
            f = fold_by_id.get(sid)
            row = dict(r)
            if f is not None:
                row["mean_ca_plddt"] = float(f["mean_ca_plddt"])
                row["ptm"] = float(f["ptm"]) if f.get("ptm") is not None else None
                row["tm_score_to_parent"] = (
                    float(f["tm_score_to_parent"])
                    if f.get("tm_score_to_parent") is not None
                    else None
                )
            by_rung[str(r["rung"])].append(row)

        rungs_out: dict[str, Any] = {}
        for r_idx, rung in enumerate(R5A_RUNGS):
            rows_r = by_rung.get(rung, [])
            # Aggregate per backbone first so bootstrap units are the 16 independent backbones.
            by_bb: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows_r:
                by_bb[str(row["backbone_id"])].append(row)
            bb_plddt: list[float] = []
            bb_ptm: list[float] = []
            bb_tm: list[float] = []
            bb_rec: list[float] = []
            bb_ent: list[float] = []
            for bb_id in sorted(by_bb):
                sub = by_bb[bb_id]
                p_vals = [x["mean_ca_plddt"] for x in sub if x.get("mean_ca_plddt") is not None]
                ptm_vals = [x["ptm"] for x in sub if x.get("ptm") is not None]
                tm_vals = [x["tm_score_to_parent"] for x in sub if x.get("tm_score_to_parent") is not None]
                rec_vals = [float(x["window_sequence_recovery"]) for x in sub if x.get("window_sequence_recovery") is not None]
                ent_vals = [exp41.sequence_shannon_entropy_bits(str(x["sequence"])) for x in sub]
                if p_vals:
                    bb_plddt.append(float(np.mean(p_vals)))
                if ptm_vals:
                    bb_ptm.append(float(np.mean(ptm_vals)))
                if tm_vals:
                    bb_tm.append(float(np.mean(tm_vals)))
                if rec_vals:
                    bb_rec.append(float(np.mean(rec_vals)))
                if ent_vals:
                    bb_ent.append(float(np.mean(ent_vals)))

            bseed = seed + idx * 100 + r_idx * 7
            rungs_out[rung] = {
                "n_generated": len(rows_r),
                "n_folded": sum(1 for x in rows_r if x.get("mean_ca_plddt") is not None),
                "n_backbones": len(by_bb),
                "mean_plddt": exp41.bootstrap_mean_ci95(bb_plddt, draws=draws, seed=bseed),
                "mean_ptm": exp41.bootstrap_mean_ci95(bb_ptm, draws=draws, seed=bseed + 1),
                "mean_tm_score_to_parent": exp41.bootstrap_mean_ci95(bb_tm, draws=draws, seed=bseed + 2),
                "window_sequence_recovery_pct": exp41.bootstrap_mean_ci95(bb_rec, draws=draws, seed=bseed + 3),
                "mean_entropy_bits": exp41.bootstrap_mean_ci95(bb_ent, draws=draws, seed=bseed + 4),
                "per_backbone_plddt": {
                    bb_id: float(np.mean([x["mean_ca_plddt"] for x in by_bb[bb_id] if x.get("mean_ca_plddt") is not None]))
                    for bb_id in sorted(by_bb)
                    if any(x.get("mean_ca_plddt") is not None for x in by_bb[bb_id])
                },
                "per_backbone_tm": {
                    bb_id: float(np.mean([x["tm_score_to_parent"] for x in by_bb[bb_id] if x.get("tm_score_to_parent") is not None]))
                    for bb_id in sorted(by_bb)
                    if any(x.get("tm_score_to_parent") is not None for x in by_bb[bb_id])
                },
            }

        meta = ARM_METADATA[spec.arm_id]
        new_entries[spec.entry_key] = {
            "entry_key": spec.entry_key,
            "arm_id": spec.arm_id,
            "display_name": meta.display_name,
            "family": meta.family,
            "paradigm": meta.paradigm,
            "params_m": meta.params_m,
            "mode": spec.mode,
            "conditioning_regime": spec.conditioning_regime,
            "temperature": spec.temperature,
            "rungs": rungs_out,
        }

    return {
        "schema": "r5a_local_infilling_full_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_baseline_source": {
            "path": str(sealed_path.relative_to(root)),
            "sha256": sha256_file(sealed_path),
        },
        "protocol": {
            "n_parent_backbones": 16,
            "rungs": list(R5A_RUNGS),
            "replicates_per_cell": R5A_REPLICATES,
            "bootstrap_unit": "parent_backbone",
            "bootstrap_draws": int(draws),
        },
        "coverage": {
            "n_sealed_gp_esm3_arms": sealed_r5a.get("n_gp_and_esm3_evals", 0),
            "n_sealed_original_ladder_arms": sealed_r5a.get("n_original_ladder_entries", 0),
            "n_new_expected": len(R5A_NEW_SPECS),
            "n_new_loaded": len(new_entries),
            "missing_entries": missing_entries,
            "is_complete": len(missing_entries) == 0,
        },
        "sealed_r5a_progressive_anchoring": sealed_r5a,
        "entries_new_dplm2_mpnn": new_entries,
    }


# --------------------------------------------------------------------------- #
# 3. Experiment 4.2 Full Assembler (23 sealed PLMs + P/H + 15 new models)
# --------------------------------------------------------------------------- #


def assemble_exp4_2_full(
    repo_root: Path = REPO_ROOT,
    work_dir: Path | None = None,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Assemble ``r6_experiment_4_2_phase_space_full.json`` (sealed 23 PLMs + P/H + 15 new)."""
    root = Path(repo_root)
    wdir = Path(work_dir) if work_dir is not None else (root / WORK_DIR_REL)
    sealed_path = root / UNIFIED_OUTPUT_DIR / "r6_experiment_4_2_phase_space_unified.json"
    sealed = load_json_file(sealed_path)

    # Reconstruct reference P & H cells from sealed artifact for paired contrasts (P-N, H-N, etc.).
    ref_cell_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arm_code, ref_block in (sealed.get("reference_arms") or {}).items():
        mlabel = str(ref_block.get("model_label") or arm_code)
        for dom, csum in (ref_block.get("per_domain") or {}).items():
            ref_cell_map[(mlabel, dom, arm_code)] = csum

    fold_dir = wdir / "exp4_2" / "folded"
    new_models_out: dict[str, Any] = {}
    new_cells_out: list[dict[str, Any]] = []
    missing_cells: list[tuple[str, str, str]] = []

    for spec in EXP4_2_NEW_SPECS:
        cell_map: dict[tuple[str, str, str], dict[str, Any]] = dict(ref_cell_map)
        arms_present: dict[str, Any] = {}
        any_loaded = False
        for arm_key in spec.arm_keys:
            for dom in exp42.SEALED_8_DOMAINS:
                p = fold_dir / f"{spec.file_slug}__{dom}__{arm_key}.jsonl"
                if not p.is_file():
                    missing_cells.append((spec.entry_key, dom, arm_key))
                    continue
                rows = read_jsonl(p)
                if not rows:
                    missing_cells.append((spec.entry_key, dom, arm_key))
                    continue
                csum = exp42.summarise_phase_space_cell(rows)
                csum["entry_key"] = spec.entry_key
                csum["conditioning_regime"] = spec.conditioning_regime
                cell_map[(spec.entry_key, dom, arm_key)] = csum
                new_cells_out.append(csum)
                any_loaded = True
            agg = exp42._aggregate_arm_across_domains(
                cell_map, spec.entry_key, arm_key, exp42.SEALED_8_DOMAINS
            )
            if agg is not None:
                arms_present[arm_key] = agg

        if not any_loaded:
            continue

        # Full 5-arm contrasts when C/Cx/Pr exist (DPLM-2 seq_only); P-N / H-N for single-arm modes.
        contrasts: dict[str, Any] = {}
        if set(exp42.PHASE_SPACE_ARM_KEYS) <= set(arms_present.keys()):
            contrasts = exp42._compute_model_contrasts(
                cell_map, spec.entry_key, exp42.SEALED_8_DOMAINS
            )
        else:
            for arm_key in spec.arm_keys:
                if arm_key not in arms_present:
                    continue
                for ref_code, ref_label in (("P", "potts-joint"), ("H", "natural")):
                    for stratum, metric in (
                        ("novel_lt20", "tm_score"),
                        ("whole_pool", "tm_score"),
                    ):
                        per: dict[str, dict[str, float]] = {}
                        miss: dict[str, str] = {}
                        for d in exp42.SEALED_8_DOMAINS:
                            ca = cell_map.get((ref_label, d, ref_code))
                            cb = cell_map.get((spec.entry_key, d, arm_key))
                            if ca is None or cb is None:
                                miss[d] = "missing cell"
                                continue
                            va, vb = ca[stratum][metric], cb[stratum][metric]
                            if va is None or vb is None:
                                miss[d] = "below admissible floor"
                                continue
                            per[d] = {
                                "diff": float(va - vb),
                                "identity_diff": float(
                                    ca[stratum]["wt_identity_pct"] - cb[stratum]["wt_identity_pct"]
                                ),
                                "msa_identity_diff": float(
                                    ca[stratum]["msa_max_identity_pct"]
                                    - cb[stratum]["msa_max_identity_pct"]
                                ),
                            }
                        contrasts.setdefault(f"{stratum}_{metric}", {})[f"{ref_code}-{arm_key}"] = (
                            exp42._estimate_domain_contrast(per, miss)
                        )

        meta = ARM_METADATA[spec.arm_id]
        new_models_out[spec.entry_key] = {
            "entry_key": spec.entry_key,
            "model_id": spec.arm_id,
            "display_name": meta.display_name,
            "family": meta.family,
            "paradigm": meta.paradigm,
            "modality": meta.modality,
            "architecture": meta.architecture,
            "params_m": meta.params_m,
            "mode": spec.mode,
            "conditioning_regime": spec.conditioning_regime,
            "arms": arms_present,
            "contrasts": contrasts,
        }

    if missing_cells and not allow_partial:
        raise FileNotFoundError(
            f"Missing {len(missing_cells)} Exp 4.2 folded cells (first 5: {missing_cells[:5]}). "
            "Pass --allow-partial for interim assembly."
        )

    all_models = dict(sealed.get("models") or {})
    all_models.update(new_models_out)

    return {
        "schema_version": "r6_experiment_4_2_phase_space_full_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domains": list(exp42.SEALED_8_DOMAINS),
        "domain_lengths": sealed.get("domain_lengths"),
        "sealed_baseline_source": {
            "path": str(sealed_path.relative_to(root)),
            "sha256": sha256_file(sealed_path),
            "n_sealed_plms": len(sealed.get("models") or {}),
        },
        "coverage": {
            "n_sealed_plms": len(sealed.get("models") or {}),
            "n_new_expected_entries": len(EXP4_2_NEW_SPECS),
            "n_new_loaded_entries": len(new_models_out),
            "n_missing_cells": len(missing_cells),
            "missing_cells_preview": [list(x) for x in missing_cells[:10]],
            "is_complete": len(missing_cells) == 0,
        },
        "reference_arms": sealed.get("reference_arms"),
        "reproduction_gates": sealed.get("reproduction_gates"),
        "sealed_paradigm_summary": sealed.get("paradigm_summary"),
        "models_new_dplm2_mpnn": new_models_out,
        "models_all": all_models,
        "cells_new_dplm2_mpnn": new_cells_out,
    }


# --------------------------------------------------------------------------- #
# 4. Experiment 4.3 Full Assembler (33 sealed pools + new DPLM-2/ProteinMPNN)
# --------------------------------------------------------------------------- #


def assemble_exp4_3_full(
    repo_root: Path = REPO_ROOT,
    work_dir: Path | None = None,
    *,
    allow_partial: bool = False,
    n_draws: int = exp43.BOOTSTRAP_DRAWS,
    fwl_draws: int = exp43.FWL_BOOTSTRAP_DRAWS,
    seed: int = exp43.BOOTSTRAP_SEED,
    diamond_threads: int = 32,
    hmmscan_workers: int = 16,
    skip_external_oracles: bool = False,
) -> dict[str, Any]:
    """Assemble ``r6_experiment_4_3_native_likelihood_full.json`` (sealed 33 + new pools)."""
    root = Path(repo_root)
    wdir = Path(work_dir) if work_dir is not None else (root / WORK_DIR_REL)
    sealed_path = root / UNIFIED_OUTPUT_DIR / "r6_experiment_4_3_native_likelihood_unified.json"
    sealed = load_json_file(sealed_path)

    scores_dir = wdir / "exp4_3" / "scores"
    score_files = sorted(scores_dir.glob("*.jsonl")) if scores_dir.is_dir() else []
    if not score_files and not allow_partial:
        raise FileNotFoundError(f"Missing Exp 4.3 self-likelihood score files under {scores_dir}")

    # Collect unique sequences for DIAMOND / Pfam lookup (reusing shared Exp 4.1 caches).
    raw_pools: dict[str, list[dict[str, Any]]] = {}
    unique_seqs: dict[str, str] = {}
    for sf in score_files:
        rows = read_jsonl(sf)
        if not rows:
            continue
        pid = str(rows[0].get("pool_id") or sf.stem)
        raw_pools[pid] = rows
        for r in rows:
            seq = str(r.get("sequence") or "").strip().upper()
            if exp41.is_canonical_aa20(seq):
                unique_seqs[seq_sha256(seq)] = seq

    if skip_external_oracles:
        dm_cache: dict[str, dict[str, Any]] = {}
        pf_cache: dict[str, dict[str, Any]] = {}
    else:
        dm_cache = exp41.ensure_diamond_annotations(unique_seqs, root, threads=diamond_threads)
        pf_cache = exp43.ensure_pfam_annotations(
            unique_seqs, root=root, workers=hmmscan_workers, threads_per_worker=2
        )

    new_pools_eval: dict[str, Any] = {}
    for pid, rows in sorted(raw_pools.items()):
        enriched: list[dict[str, Any]] = []
        arm_id = str(rows[0].get("arm_id") or pid.split("|")[0])
        conditioning_regime = str(rows[0].get("conditioning_regime") or "unknown")
        scoring_backbone_source = str(rows[0].get("scoring_backbone_source") or "none")
        for r in rows:
            seq = str(r["sequence"]).strip().upper()
            sha = str(r.get("sequence_sha256") or seq_sha256(seq))
            nll_tok, nll_res = exp43._extract_score_fields(r)
            if not (math.isfinite(nll_tok) and math.isfinite(nll_res)):
                continue
            dm = dm_cache.get(sha)
            ident = float(dm["identity_over_query"]) if dm is not None else float(r.get("uniref50_identity", 0.0))
            pf = pf_cache.get(sha) or {"families": []}
            comp = exp43.sequence_complexity_features(seq)
            enriched.append(
                {
                    "pool_id": pid,
                    "arm_id": arm_id,
                    "seq_id": str(r["seq_id"]),
                    "sequence": seq,
                    "sequence_sha256": sha,
                    "unit_id": str(r["seq_id"]),
                    "plddt": float(r["plddt"]),
                    "ptm": float(r["ptm"]),
                    "mean_nll_per_scored_token": nll_tok,
                    "mean_nll_per_residue": nll_res,
                    "uniref50_identity": ident,
                    "pfam_families": list(pf.get("families") or []),
                    **comp,
                }
            )
        if not enriched:
            continue
        ev = exp43.evaluate_pool_across_regimes(
            pid, enriched, n_draws=n_draws, fwl_draws=fwl_draws, seed=seed
        )
        meta = ARM_METADATA.get(arm_id)
        if meta is not None:
            ev["arm_id"] = arm_id
            ev["display_name"] = meta.display_name
            ev["family"] = meta.family
            ev["paradigm"] = meta.paradigm
            ev["params_m"] = meta.params_m
        ev["conditioning_regime"] = conditioning_regime
        ev["scoring_backbone_source"] = scoring_backbone_source
        new_pools_eval[pid] = ev

    expected_min_pools = len(EXP4_1_NEW_SPECS)
    if len(new_pools_eval) < expected_min_pools and not allow_partial:
        raise FileNotFoundError(
            f"Expected at least {expected_min_pools} new Exp 4.3 scored pools under {scores_dir}, "
            f"found {len(new_pools_eval)}"
        )

    all_pools = dict(sealed.get("pools") or {})
    all_pools.update(new_pools_eval)

    return {
        "schema": "r6_experiment_4_3_native_likelihood_full_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_baseline_source": {
            "path": str(sealed_path.relative_to(root)),
            "sha256": sha256_file(sealed_path),
            "n_sealed_pools": len(sealed.get("pools") or {}),
        },
        "coverage": {
            "n_sealed_pools": len(sealed.get("pools") or {}),
            "n_new_scored_pools": len(new_pools_eval),
            "total_pools": len(all_pools),
            "is_complete": len(new_pools_eval) >= expected_min_pools,
        },
        "e17_reproduction_gate": sealed.get("e17_reproduction_gate"),
        "cross_model_summary_all": exp43._aggregate_cross_model_summary(all_pools),
        "cross_model_summary_new_only": (
            exp43._aggregate_cross_model_summary(new_pools_eval) if new_pools_eval else {}
        ),
        "pools_new_dplm2_mpnn": new_pools_eval,
        "pools_all": all_pools,
    }


# --------------------------------------------------------------------------- #
# 5. Result 5 Downstream 2 Continuation Full Assembler (35 sealed + 19 new)
# --------------------------------------------------------------------------- #


def assemble_r4_continuation_full(
    repo_root: Path = REPO_ROOT,
    work_dir: Path | None = None,
    *,
    allow_partial: bool = False,
    diamond_threads: int = 32,
    hmmscan_workers: int = 16,
    skip_external_oracles: bool = False,
) -> dict[str, Any]:
    """Assemble ``r4_continuation_unified_full.json`` (sealed 35 + 19 new DPLM-2/MPNN entries)."""
    root = Path(repo_root)
    wdir = Path(work_dir) if work_dir is not None else (root / WORK_DIR_REL)
    sealed_path = root / UNIFIED_OUTPUT_DIR / "r4_continuation_unified.json"
    sealed = load_json_file(sealed_path) if sealed_path.is_file() else {}
    if not sealed and not allow_partial:
        raise FileNotFoundError(f"Missing sealed baseline {sealed_path}")

    homologs_path = root / r4c.DEFAULT_HOMOLOGS_JSON
    contexts = load_json_file(homologs_path)
    plans = r4c.build_canonical_target_plans(contexts, budget=r4c.BUDGET_TOKENS)

    gen_dir = wdir / "r4_cont" / "generated"
    fold_dir = wdir / "r4_cont" / "folded"

    loaded_attempts: dict[str, list[dict[str, Any]]] = {}
    loaded_folds: dict[str, dict[str, dict[str, Any]]] = {}
    missing_entries: list[str] = []

    for spec in R4_CONT_NEW_SPECS:
        gp = gen_dir / f"{spec.file_slug}.jsonl"
        fp = fold_dir / f"{spec.file_slug}.jsonl"
        if not gp.is_file() or (not allow_partial and not fp.is_file()):
            if not allow_partial:
                raise FileNotFoundError(f"Missing R4-Cont artifacts for {spec.entry_key}: {gp} / {fp}")
            missing_entries.append(spec.entry_key)
            continue
        loaded_attempts[spec.entry_key] = read_jsonl(gp)
        loaded_folds[spec.entry_key] = {
            str(r["attempt_id"]): r for r in read_jsonl(fp)
        }

    # Build UniRef50 + Pfam + own-context caches (seeding from sealed r4_continuation_pools caches).
    all_valid_seqs: dict[str, str] = {}
    non_copy_seqs: dict[str, str] = {}
    unaligned_attempts_all: list[dict[str, Any]] = []
    wt_sha_by_tid: dict[str, str] = {}
    wt_len_by_tid: dict[str, int] = {}
    for tid, plan in plans.items():
        wt_seq = str(plan["wildtype"]).strip().upper()
        wt_sha = seq_sha256(wt_seq)
        wt_sha_by_tid[tid] = wt_sha
        wt_len_by_tid[tid] = len(wt_seq)
        non_copy_seqs[wt_sha] = wt_seq

    for spec in R4_CONT_NEW_SPECS:
        ek = spec.entry_key
        if ek not in loaded_attempts:
            continue
        att_rows = loaded_attempts[ek]
        if spec.conditioning_regime == "unaligned_homolog_prefix":
            unaligned_attempts_all.extend(att_rows)
        for r in att_rows:
            seq = str(r.get("sequence") or "").strip().upper()
            if exp41.is_canonical_aa20(seq) and len(seq) >= r4c.MIN_PRODUCT_RESIDUES:
                sha = str(r.get("sequence_sha256") or seq_sha256(seq))
                all_valid_seqs[sha] = seq
                if not bool(r.get("copied_by_lcs_or_kmer", False)):
                    non_copy_seqs[sha] = seq

    ann_dir = wdir / "r4_cont" / "annotation_cache"
    seed_ann_dir = root / r4c.DEFAULT_WORK_DIR / "annotation_cache"
    if skip_external_oracles:
        uniref_cache: dict[str, dict[str, Any]] = {}
        pfam_cache: dict[str, dict[str, Any]] = {}
        ctx_pair_ident: dict[tuple[str, str], float] = {}
    else:
        ctx_pair_ident = r4c.run_prompt_context_diamond(
            unaligned_attempts_all,
            plans,
            repo_root=root,
            threads=diamond_threads,
        )
        uniref_cache = r4c.ensure_uniref50_diamond_cache(
            all_valid_seqs,
            cache_path=ann_dir / "diamond_uniref50_by_sha256.json",
            seed_cache_path=seed_ann_dir / "diamond_uniref50_by_sha256.json",
            repo_root=root,
            threads=diamond_threads,
        )
        pfam_cache = r4c.ensure_pfam_hmmscan_cache(
            non_copy_seqs,
            cache_path=ann_dir / "pfam_hmmscan_by_sha256.json",
            seed_cache_path=seed_ann_dir / "pfam_hmmscan_by_sha256.json",
            repo_root=root,
            workers=hmmscan_workers,
            threads_per_worker=2,
        )

    target_wt_pfams: dict[str, set[str]] = {
        tid: set((pfam_cache.get(wt_sha_by_tid[tid]) or {}).get("families") or ())
        for tid in plans
    }

    def _summarise_unaligned_entry(
        ek: str,
        spec: R4ContNewSpec,
        att_rows: list[dict[str, Any]],
        fold_map: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any]:
        meta = ARM_METADATA[spec.arm_id]
        enriched: list[dict[str, Any]] = []
        for r in att_rows:
            seq = str(r["sequence"]).strip().upper()
            sha = str(r.get("sequence_sha256") or (seq_sha256(seq) if seq else ""))
            subjs = r.get("context_subjects") or []
            own_id = (
                max((ctx_pair_ident.get((sha, str(s)), 0.0) for s in subjs), default=0.0)
                if subjs and ctx_pair_ident
                else (0.0 if subjs else None)
            )
            verdict = r4c.H.copy_verdict(r, identity_percent=own_id)
            is_copy = bool(verdict["is_copy"])
            residues = int(r.get("residues", len(seq)))
            valid_len = residues >= r4c.MIN_PRODUCT_RESIDUES
            stop_st = str(r.get("stop_status") or ("native_terminal" if valid_len else "empty_or_noncanonical"))
            complete = valid_len and stop_st == "native_terminal"
            non_copy_complete = complete and (not is_copy)
            h_bits = float(r.get("shannon_H_bits", r4c.shannon_entropy_bits(seq) if seq else 0.0))
            d5 = float(r.get("distinct_5mer", r4c.distinct_kmer_fraction(seq, 5) if seq else 0.0))
            nat_ent_non_copy = non_copy_complete and h_bits >= r4c.NATURAL_ENTROPY_THRESHOLD

            pf_rec = pfam_cache.get(sha, {})
            prod_fams = set(pf_rec.get("families") or ())
            tgt_fams = target_wt_pfams.get(str(r["target_id"]), set())
            tgt_has_pfam = bool(tgt_fams)
            tgt_pfam_hit = bool(valid_len and (not is_copy) and tgt_has_pfam and (prod_fams & tgt_fams))

            ur_rec = uniref_cache.get(sha, {})
            ur_id = ur_rec.get("identity_over_query")
            ur_novel_lt30 = (float(ur_id) < 30.0) if ur_id is not None else None

            fold_rec = fold_map.get(str(r["attempt_id"]))
            plddt = float(fold_rec["plddt_mean"]) if fold_rec else None
            ptm = float(fold_rec["ptm"]) if fold_rec else None
            pae = float(fold_rec["mean_pae_angstrom"]) if fold_rec else None
            tm_wt = (
                float(fold_rec["tm_score_to_target_wt"])
                if fold_rec and fold_rec.get("tm_score_to_target_wt") is not None
                else None
            )
            fold_high_qual = (
                bool((not is_copy) and h_bits >= r4c.NATURAL_ENTROPY_THRESHOLD and plddt is not None and plddt >= 70.0)
                if fold_rec
                else None
            )
            tm_recap = (tm_wt >= r4c.TM_RECAP_THRESHOLD) if tm_wt is not None else None
            enriched.append(
                {
                    **r,
                    "residues": residues,
                    "shannon_H_bits": h_bits,
                    "distinct_5mer": d5,
                    "max_lcs_to_context": float(r.get("max_lcs_to_context", 0.0)),
                    "context_alignment_identity": own_id,
                    "is_copy": is_copy,
                    "complete": complete,
                    "non_copy_complete": non_copy_complete,
                    "natural_entropy_non_copy": nat_ent_non_copy,
                    "any_pfam": bool(pf_rec.get("any_family", False)) if valid_len and not is_copy else False,
                    "complete_pfam80": bool(pf_rec.get("complete_domain", False)) if valid_len and not is_copy else False,
                    "target_has_pfam": tgt_has_pfam,
                    "target_pfam_recalled": tgt_pfam_hit,
                    "uniref50_identity_pct": float(ur_id) if ur_id is not None else None,
                    "uniref50_novel_lt30": ur_novel_lt30,
                    "plddt_mean": plddt,
                    "ptm": ptm,
                    "mean_pae_angstrom": pae,
                    "tm_score_to_target_wt": tm_wt,
                    "folded_high_qual_non_copy": fold_high_qual,
                    "tm_ge_0_5": tm_recap,
                }
            )

        def _fold_subset_summary(sub_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            if not sub_rows:
                return {"n_folded": 0, "mean_plddt": None, "mean_ptm": None, "mean_tm_score_to_target_wt": None}
            return {
                "n_folded": len(sub_rows),
                "mean_plddt": r4c._safe_mean([x["plddt_mean"] for x in sub_rows]),
                "mean_ptm": r4c._safe_mean([x["ptm"] for x in sub_rows]),
                "mean_tm_score_to_target_wt": r4c._safe_mean([x["tm_score_to_target_wt"] for x in sub_rows]),
            }

        def _summarise_panel(panel_targets: Sequence[str], panel_conditions: Sequence[str]) -> dict[str, Any]:
            tset = set(panel_targets)
            sub = [x for x in enriched if str(x["target_id"]) in tset]
            per_cond: dict[str, Any] = {}
            by_cluster_metric: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
            for cond in panel_conditions:
                c_rows = [x for x in sub if x["condition"] == cond]
                n_att = len(c_rows)
                valid_rows = [x for x in c_rows if int(x["residues"]) >= r4c.MIN_PRODUCT_RESIDUES]
                pfam_elig = [x for x in c_rows if x["target_has_pfam"]]
                folded_rows = [x for x in c_rows if x["plddt_mean"] is not None]
                ur_rows = [x for x in valid_rows if x["uniref50_identity_pct"] is not None]

                matched_rows = [
                    x for x in folded_rows
                    if 0.75 * wt_len_by_tid[str(x["target_id"])] <= int(x["residues"]) <= 1.25 * wt_len_by_tid[str(x["target_id"])]
                ]
                unmatched_rows = [x for x in folded_rows if x not in matched_rows]
                band_16_128 = [x for x in folded_rows if 16 <= int(x["residues"]) <= 128]
                band_129_256 = [x for x in folded_rows if 129 <= int(x["residues"]) <= 256]
                band_257_512 = [x for x in folded_rows if 257 <= int(x["residues"]) <= 512]

                per_cond[cond] = {
                    "attempts": n_att,
                    "valid_ge32": len(valid_rows),
                    "complete_yield": float(np.mean([x["complete"] for x in c_rows])) if n_att else None,
                    "copy_rate_all": float(np.mean([x["is_copy"] for x in c_rows])) if n_att else None,
                    "copy_rate_valid": float(np.mean([x["is_copy"] for x in valid_rows])) if valid_rows else None,
                    "non_copy_complete_yield": float(np.mean([x["non_copy_complete"] for x in c_rows])) if n_att else None,
                    "natural_entropy_non_copy_yield": float(np.mean([x["natural_entropy_non_copy"] for x in c_rows])) if n_att else None,
                    "mean_shannon_H_bits": r4c._safe_mean([x["shannon_H_bits"] for x in valid_rows]),
                    "mean_distinct_5mer": r4c._safe_mean([x["distinct_5mer"] for x in valid_rows]),
                    "mean_max_lcs": r4c._safe_mean([x["max_lcs_to_context"] for x in valid_rows]),
                    "target_pfam_recall_non_copy": float(np.mean([x["target_pfam_recalled"] for x in pfam_elig])) if pfam_elig else None,
                    "any_pfam_rate_all": float(np.mean([x["any_pfam"] for x in c_rows])) if n_att else None,
                    "complete_pfam80_rate_all": float(np.mean([x["complete_pfam80"] for x in c_rows])) if n_att else None,
                    "uniref50_novel_lt30_share": float(np.mean([x["uniref50_novel_lt30"] for x in ur_rows])) if ur_rows else None,
                    "uniref50_identity_median": float(np.median([x["uniref50_identity_pct"] for x in ur_rows])) if ur_rows else None,
                    "n_folded": len(folded_rows),
                    "mean_plddt": r4c._safe_mean([x["plddt_mean"] for x in folded_rows]),
                    "mean_ptm": r4c._safe_mean([x["ptm"] for x in folded_rows]),
                    "mean_pae": r4c._safe_mean([x["mean_pae_angstrom"] for x in folded_rows]),
                    "folded_high_qual_non_copy_rate": float(np.mean([x["folded_high_qual_non_copy"] for x in folded_rows])) if folded_rows else None,
                    "mean_tm_score_to_target_wt": r4c._safe_mean([x["tm_score_to_target_wt"] for x in folded_rows]),
                    "tm_ge_0_5_rate": float(np.mean([x["tm_ge_0_5"] for x in folded_rows])) if folded_rows else None,
                    "length_stratified_folding": {
                        "length_matched": _fold_subset_summary(matched_rows),
                        "unmatched": _fold_subset_summary(unmatched_rows),
                        "len_16_128": _fold_subset_summary(band_16_128),
                        "len_129_256": _fold_subset_summary(band_129_256),
                        "len_257_512": _fold_subset_summary(band_257_512),
                    },
                }
                by_cl: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for x in c_rows:
                    by_cl[str(x["cluster"])].append(x)
                for cl, cl_rows in by_cl.items():
                    by_cluster_metric["non_copy_complete_yield"][cond][cl] = float(np.mean([x["non_copy_complete"] for x in cl_rows]))
                    by_cluster_metric["natural_entropy_non_copy_yield"][cond][cl] = float(np.mean([x["natural_entropy_non_copy"] for x in cl_rows]))
                    cl_pf = [x for x in cl_rows if x["target_has_pfam"]]
                    if cl_pf:
                        by_cluster_metric["target_pfam_recall_non_copy"][cond][cl] = float(np.mean([x["target_pfam_recalled"] for x in cl_pf]))
                    cl_fold = [x for x in cl_rows if x["plddt_mean"] is not None]
                    if cl_fold:
                        by_cluster_metric["mean_plddt"][cond][cl] = float(np.mean([x["plddt_mean"] for x in cl_fold]))
                        by_cluster_metric["mean_ptm"][cond][cl] = float(np.mean([x["ptm"] for x in cl_fold]))
                        tm_vals = [x["tm_score_to_target_wt"] for x in cl_fold if x["tm_score_to_target_wt"] is not None]
                        if tm_vals:
                            by_cluster_metric["mean_tm_score_to_target_wt"][cond][cl] = float(np.mean(tm_vals))
                            by_cluster_metric["tm_ge_0_5_rate"][cond][cl] = float(np.mean([v >= r4c.TM_RECAP_THRESHOLD for v in tm_vals]))

            contrast_pairs = [
                (f"{c}_minus_unrelated", c, "unrelated")
                for c in panel_conditions
                if c not in ("no_context", "unrelated")
            ]
            contrasts: dict[str, dict[str, Any]] = {}
            for metric_name in (
                "non_copy_complete_yield",
                "natural_entropy_non_copy_yield",
                "target_pfam_recall_non_copy",
                "mean_plddt",
                "mean_ptm",
                "mean_tm_score_to_target_wt",
                "tm_ge_0_5_rate",
            ):
                contrasts[metric_name] = {}
                for cname, left_c, right_c in contrast_pairs:
                    contrasts[metric_name][cname] = r4c._paired_cluster_bootstrap(
                        by_cluster_metric[metric_name].get(left_c, {}),
                        by_cluster_metric[metric_name].get(right_c, {}),
                    )
            return {"targets": list(panel_targets), "conditions": per_cond, "contrasts": contrasts}

        return {
            "model_id": ek,
            "entry_key": ek,
            "arm_id": spec.arm_id,
            "base_model_id": spec.arm_id,
            "display_name": meta.display_name,
            "family": meta.family,
            "paradigm": meta.paradigm,
            "modality": meta.modality,
            "architecture": meta.architecture,
            "params_m": meta.params_m,
            "vocab_type": meta.vocab_type,
            "backend": spec.backend,
            "context_budget_tokens": DIMA_MAX_RESIDUES if spec.backend == "dima" else r4c.BUDGET_TOKENS,
            "input_alignment_format": "unaligned_concat",
            "context_training_regime": spec.context_training_regime,
            "conditioning_regime": spec.conditioning_regime,
            "is_text_control": False,
            "primary_12": _summarise_panel(r4c.PRIMARY_12_TARGETS, r4c.PRIMARY_CONDITIONS),
            "twilight_12": _summarise_panel(r4c.TWILIGHT_12_TARGETS, r4c.TWILIGHT_CONDITIONS),
        }

    new_models_out: dict[str, Any] = {}
    for spec in R4_CONT_NEW_SPECS:
        ek = spec.entry_key
        if ek not in loaded_attempts:
            continue
        att_rows = loaded_attempts[ek]
        fold_map = loaded_folds.get(ek, {})
        meta = ARM_METADATA[spec.arm_id]

        if spec.conditioning_regime == "unaligned_homolog_prefix":
            new_models_out[ek] = _summarise_unaligned_entry(ek, spec, att_rows, fold_map)
        else:
            # 3D-backbone 50%-WT-prefix suffix continuation across the 22 targets.
            per_target: dict[str, dict[str, Any]] = {}
            by_tid: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in att_rows:
                by_tid[str(r["target_id"])].append(r)
            all_plddt: list[float] = []
            all_tm: list[float] = []
            all_rec: list[float] = []
            all_pfam: list[float] = []
            all_uniref: list[float] = []
            for tid in sorted(by_tid):
                sub = by_tid[tid]
                target_pfams = target_wt_pfams.get(tid, set())
                t_plddt: list[float] = []
                t_tm: list[float] = []
                t_rec: list[float] = []
                t_pfam: list[float] = []
                t_uid: list[float] = []
                for r in sub:
                    sha = str(r.get("sequence_sha256") or seq_sha256(str(r["sequence"])))
                    f = fold_map.get(str(r["attempt_id"]))
                    if f is not None:
                        t_plddt.append(float(f["plddt_mean"]))
                        t_tm.append(float(f["tm_score_to_target_wt"]))
                    if r.get("suffix_sequence_recovery") is not None:
                        t_rec.append(float(r["suffix_sequence_recovery"]))
                    pf = pfam_cache.get(sha) or {}
                    fams = set(pf.get("families") or [])
                    t_pfam.append(1.0 if (target_pfams and (fams & target_pfams)) else 0.0)
                    dm = uniref_cache.get(sha) or {}
                    t_uid.append(float(dm.get("identity_over_query", 0.0)))
                per_target[tid] = {
                    "n_attempts": len(sub),
                    "n_folded": len(t_plddt),
                    "mean_plddt": float(np.mean(t_plddt)) if t_plddt else None,
                    "mean_tm_score_to_target_wt": float(np.mean(t_tm)) if t_tm else None,
                    "suffix_sequence_recovery_pct": float(np.mean(t_rec)) if t_rec else None,
                    "target_pfam_recall": float(np.mean(t_pfam)) if t_pfam else None,
                    "mean_uniref50_identity": float(np.mean(t_uid)) if t_uid else None,
                }
                if t_plddt:
                    all_plddt.append(float(np.mean(t_plddt)))
                if t_tm:
                    all_tm.append(float(np.mean(t_tm)))
                if t_rec:
                    all_rec.append(float(np.mean(t_rec)))
                if t_pfam:
                    all_pfam.append(float(np.mean(t_pfam)))
                if t_uid:
                    all_uniref.append(float(np.mean(t_uid)))

            new_models_out[ek] = {
                "entry_key": ek,
                "arm_id": spec.arm_id,
                "display_name": meta.display_name,
                "family": meta.family,
                "paradigm": meta.paradigm,
                "params_m": meta.params_m,
                "context_training_regime": spec.context_training_regime,
                "conditioning_regime": spec.conditioning_regime,
                "n_targets": len(per_target),
                "n_attempts_total": len(att_rows),
                "n_folded_total": len(fold_map),
                "summary": {
                    "mean_plddt": exp41.bootstrap_mean_ci95(all_plddt, draws=DEFAULT_BOOTSTRAP_DRAWS, seed=DEFAULT_SEED),
                    "mean_tm_score_to_target_wt": exp41.bootstrap_mean_ci95(all_tm, draws=DEFAULT_BOOTSTRAP_DRAWS, seed=DEFAULT_SEED + 1),
                    "suffix_sequence_recovery_pct": exp41.bootstrap_mean_ci95(all_rec, draws=DEFAULT_BOOTSTRAP_DRAWS, seed=DEFAULT_SEED + 2),
                    "target_pfam_recall": exp41.bootstrap_mean_ci95(all_pfam, draws=DEFAULT_BOOTSTRAP_DRAWS, seed=DEFAULT_SEED + 3),
                    "mean_uniref50_identity": exp41.bootstrap_mean_ci95(all_uniref, draws=DEFAULT_BOOTSTRAP_DRAWS, seed=DEFAULT_SEED + 4),
                },
                "per_target": per_target,
            }

    sealed_models = dict(sealed.get("models") or {})
    all_models = dict(sealed_models)
    all_models.update(new_models_out)

    return {
        "schema_version": "r4_continuation_unified_full_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_baseline_source": (
            {
                "path": str(sealed_path.relative_to(root)),
                "sha256": sha256_file(sealed_path),
                "n_sealed_entries": len(sealed_models),
            }
            if sealed_path.is_file()
            else None
        ),
        "coverage": {
            "n_sealed_entries": len(sealed_models),
            "n_new_expected": len(R4_CONT_NEW_SPECS),
            "n_new_loaded": len(new_models_out),
            "n_unaligned_prefix_total": len(sealed_models)
            + sum(
                1
                for m in new_models_out.values()
                if m.get("conditioning_regime") == "unaligned_homolog_prefix"
            ),
            "n_backbone_3d_suffix_entries": sum(
                1
                for m in new_models_out.values()
                if m.get("conditioning_regime") == "backbone_3d_wt_prefix_suffix"
            ),
            "missing_entries": missing_entries,
            "is_complete": len(missing_entries) == 0 and bool(sealed_models),
        },
        "e11_reproduction_gate": sealed.get("e11_reproduction_gate"),
        "models_new_dplm2_mpnn": new_models_out,
        "models_all": all_models,
    }


# --------------------------------------------------------------------------- #
# 6. Result 6 Structural & Objective Adaptation Contrasts
# --------------------------------------------------------------------------- #


def _extract_exp4_1_scalar(entry: Mapping[str, Any] | None, metric: str) -> float | None:
    if not entry:
        return None
    ax1 = entry.get("axis1_full_pool") or {}
    ax2 = entry.get("axis2_structure_joined") or {}
    ifm = entry.get("inverse_folding_metrics") or {}
    if metric == "mean_plddt":
        return (ax2.get("all_folded") or {}).get("mean_plddt", {}).get("mean")
    if metric == "mean_ptm":
        return (ax2.get("all_folded") or {}).get("mean_ptm", {}).get("mean")
    if metric == "foldable_ge70":
        return (ax2.get("all_folded") or {}).get("foldable_rate_plddt_ge_70", {}).get("mean")
    if metric == "gated_yield":
        return (ax2.get("gated_lt30_h_ge_3_0") or {}).get("gated_yield_of_pool", {}).get("mean")
    if metric == "uniref50_frac_lt30":
        return ax1.get("uniref50_frac_lt30_novel")
    if metric == "mean_entropy_bits":
        return ax1.get("mean_entropy_bits")
    if metric == "pfam_complete_domain":
        return ax1.get("pfam_complete_domain_rate_all")
    if metric == "template_seq_recovery":
        blk = ifm.get("template_sequence_recovery_pct")
        return blk.get("mean") if isinstance(blk, dict) else None
    if metric == "backbone_sc_tm":
        blk = ifm.get("backbone_sc_tm")
        return blk.get("mean") if isinstance(blk, dict) else None
    return None


def _extract_exp4_2_domain_values(
    cells: Sequence[Mapping[str, Any]],
    entry_key: str,
    arm_key: str,
    stratum: str,
    metric: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in cells:
        ckey = str(c.get("entry_key") or c.get("model_id") or "")
        if ckey == entry_key and str(c.get("arm_key") or c.get("arm") or "") == arm_key:
            val = (c.get(stratum) or {}).get(metric)
            if val is not None and math.isfinite(float(val)):
                out[str(c["domain"])] = float(val)
    return out


def assemble_r6_dplm2_mpnn_adaptation(
    repo_root: Path = REPO_ROOT,
    *,
    exp4_1_doc: Mapping[str, Any] | None = None,
    r5a_doc: Mapping[str, Any] | None = None,
    exp4_2_doc: Mapping[str, Any] | None = None,
    exp4_3_doc: Mapping[str, Any] | None = None,
    r4_cont_doc: Mapping[str, Any] | None = None,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Assemble ``r6_dplm2_mpnn_adaptation_unified.json`` across all 5 adaptation axes."""
    e41_entries = (exp4_1_doc or {}).get("entries_all") or {}
    r5a_entries = (r5a_doc or {}).get("entries_new_dplm2_mpnn") or {}
    e42_models = (exp4_2_doc or {}).get("models_all") or {}
    e42_cells = list((exp4_2_doc or {}).get("cells_new_dplm2_mpnn") or [])
    e43_pools = (exp4_3_doc or {}).get("pools_all") or {}
    r4_models = (r4_cont_doc or {}).get("models_all") or {}

    def _pair_summary(
        label: str,
        key_a_e41: str,
        key_b_e41: str,
        key_a_r5a: str,
        key_b_r5a: str,
        key_a_e42: str,
        key_b_e42: str,
        e42_arm: str,
        key_a_r4: str,
        key_b_r4: str,
        pair_seed: int,
    ) -> dict[str, Any]:
        # 1. Exp 4.1 deltas
        e41_metrics = (
            "mean_plddt",
            "mean_ptm",
            "foldable_ge70",
            "gated_yield",
            "uniref50_frac_lt30",
            "mean_entropy_bits",
            "pfam_complete_domain",
            "template_seq_recovery",
            "backbone_sc_tm",
        )
        e41_cmp: dict[str, Any] = {}
        ea, eb = e41_entries.get(key_a_e41), e41_entries.get(key_b_e41)
        for m in e41_metrics:
            va, vb = _extract_exp4_1_scalar(ea, m), _extract_exp4_1_scalar(eb, m)
            e41_cmp[m] = {
                "a": va,
                "b": vb,
                "delta": (float(va - vb) if va is not None and vb is not None else None),
            }

        # 2. R5a paired backbone bootstrap at k40 (hardest span)
        r5a_cmp: dict[str, Any] = {}
        ra, rb = r5a_entries.get(key_a_r5a), r5a_entries.get(key_b_r5a)
        if ra and rb:
            for rung in ("k10", "k20", "k40"):
                pa = ((ra.get("rungs") or {}).get(rung) or {}).get("per_backbone_plddt") or {}
                pb = ((rb.get("rungs") or {}).get(rung) or {}).get("per_backbone_plddt") or {}
                shared = sorted(set(pa) & set(pb))
                r5a_cmp[f"{rung}_plddt"] = paired_bootstrap_diff(
                    [pa[k] for k in shared], [pb[k] for k in shared], draws=draws, seed=pair_seed
                )
                ta = ((ra.get("rungs") or {}).get(rung) or {}).get("per_backbone_tm") or {}
                tb = ((rb.get("rungs") or {}).get(rung) or {}).get("per_backbone_tm") or {}
                shared_t = sorted(set(ta) & set(tb))
                r5a_cmp[f"{rung}_tm"] = paired_bootstrap_diff(
                    [ta[k] for k in shared_t], [tb[k] for k in shared_t], draws=draws, seed=pair_seed + 1
                )

        # 3. Exp 4.2 paired 8-domain bootstrap
        e42_cmp: dict[str, Any] = {}
        for stratum, metric in (("whole_pool", "tm_score"), ("whole_pool", "wt_identity_pct"), ("novel_lt20", "tm_score")):
            da = _extract_exp4_2_domain_values(e42_cells, key_a_e42, e42_arm, stratum, metric)
            db = _extract_exp4_2_domain_values(e42_cells, key_b_e42, e42_arm, stratum, metric)
            shared = sorted(set(da) & set(db))
            if shared:
                e42_cmp[f"{stratum}_{metric}"] = paired_bootstrap_diff(
                    [da[d] for d in shared], [db[d] for d in shared], draws=draws, seed=pair_seed + 2
                )
            else:
                ma = ((e42_models.get(key_a_e42) or {}).get("arms") or {}).get(e42_arm, {}).get(stratum, {}).get(metric)
                mb = ((e42_models.get(key_b_e42) or {}).get("arms") or {}).get(e42_arm, {}).get(stratum, {}).get(metric)
                e42_cmp[f"{stratum}_{metric}"] = {
                    "mean_a": ma,
                    "mean_b": mb,
                    "delta": (float(ma - mb) if ma is not None and mb is not None else None),
                }

        # 4. Exp 4.3 Spearman(high_lik, pLDDT) delta on whole_pool & novel_lt30_natural_entropy
        e43_cmp: dict[str, Any] = {}
        for reg in ("whole_pool", "novel_lt30_natural_entropy"):
            pa_e43 = (((e43_pools.get(key_a_e41) or {}).get("regimes") or {}).get(reg) or {}).get("spearman_high_likelihood", {}).get("plddt", {}).get("point")
            pb_e43 = (((e43_pools.get(key_b_e41) or {}).get("regimes") or {}).get(reg) or {}).get("spearman_high_likelihood", {}).get("plddt", {}).get("point")
            e43_cmp[f"{reg}_spearman_plddt"] = {
                "a": pa_e43,
                "b": pb_e43,
                "delta": (float(pa_e43 - pb_e43) if pa_e43 is not None and pb_e43 is not None else None),
            }

        # 5. R4 Continuation paired target contrast (if 3D suffix continuation)
        r4_cmp: dict[str, Any] = {}
        m4a, m4b = r4_models.get(key_a_r4), r4_models.get(key_b_r4)
        if m4a and m4b and m4a.get("per_target") and m4b.get("per_target"):
            shared_t = sorted(set(m4a["per_target"]) & set(m4b["per_target"]))
            for field in ("mean_plddt", "mean_tm_score_to_target_wt", "suffix_sequence_recovery_pct"):
                va = [m4a["per_target"][t][field] for t in shared_t if m4a["per_target"][t].get(field) is not None and m4b["per_target"][t].get(field) is not None]
                vb = [m4b["per_target"][t][field] for t in shared_t if m4a["per_target"][t].get(field) is not None and m4b["per_target"][t].get(field) is not None]
                if va:
                    r4_cmp[field] = paired_bootstrap_diff(va, vb, draws=draws, seed=pair_seed + 3)

        return {
            "label": label,
            "model_a": key_a_e41,
            "model_b": key_b_e41,
            "exp4_1": e41_cmp,
            "r5a": r5a_cmp,
            "exp4_2": e42_cmp,
            "exp4_3": e43_cmp,
            "r4_continuation": r4_cmp,
        }

    # Axis 1: SolubleMPNN vs VanillaMPNN (matched noise 0.02..0.30 A)
    soluble_vs_vanilla = [
        _pair_summary(
            f"soluble_vs_vanilla_{noise}",
            f"{a}|T0.1",
            f"{b}|T0.1",
            a,
            b,
            a,
            b,
            "N",
            a,
            b,
            seed + 10 * idx,
        )
        for idx, (noise, a, b) in enumerate(SOLUBLE_VS_VANILLA_PAIRS)
    ]

    # Axis 2: CA-only vs All-Atom VanillaMPNN (matched noise 0.02..0.20 A)
    ca_vs_vanilla = [
        _pair_summary(
            f"ca_vs_vanilla_{noise}",
            f"{a}|T0.1",
            f"{b}|T0.1",
            a,
            b,
            a,
            b,
            "N",
            a,
            b,
            seed + 100 + 10 * idx,
        )
        for idx, (noise, a, b) in enumerate(CA_VS_VANILLA_PAIRS)
    ]

    # Axis 3: ProteinMPNN Noise Ladder (0.02 -> 0.10 -> 0.20 -> 0.30 A)
    noise_ladders: dict[str, list[dict[str, Any]]] = {}
    for fam_name, rungs in MPNN_NOISE_LADDERS.items():
        ladder_rows: list[dict[str, Any]] = []
        for noise_a, arm_id in rungs:
            e = e41_entries.get(f"{arm_id}|T0.1")
            ladder_rows.append(
                {
                    "noise_angstrom": noise_a,
                    "arm_id": arm_id,
                    "exp4_1_mean_plddt": _extract_exp4_1_scalar(e, "mean_plddt"),
                    "exp4_1_foldable_ge70": _extract_exp4_1_scalar(e, "foldable_ge70"),
                    "exp4_1_gated_yield": _extract_exp4_1_scalar(e, "gated_yield"),
                    "exp4_1_uniref50_frac_lt30": _extract_exp4_1_scalar(e, "uniref50_frac_lt30"),
                    "exp4_1_template_seq_recovery": _extract_exp4_1_scalar(e, "template_seq_recovery"),
                    "exp4_1_backbone_sc_tm": _extract_exp4_1_scalar(e, "backbone_sc_tm"),
                }
            )
        noise_ladders[fam_name] = ladder_rows

    # Axis 4: DPLM-2-Bit-650M vs DPLM-2-650M (13-bit LFQ vs 8192-codebook across all 3 modes)
    dplm2_bit_vs_vq = [
        _pair_summary(
            f"dplm2_bit_vs_vq_{mode_tag}",
            f"dplm2-bit-650m|{e41_suf}",
            f"dplm2-650m|{e41_suf}",
            f"dplm2-bit-650m|{r5a_suf}",
            f"dplm2-650m|{r5a_suf}",
            f"dplm2-bit-650m|{e42_suf}",
            f"dplm2-650m|{e42_suf}",
            "N",
            f"dplm2-bit-650m|{r5a_suf}",
            f"dplm2-650m|{r5a_suf}",
            seed + 200 + 10 * idx,
        )
        for idx, (mode_tag, e41_suf, r5a_suf, e42_suf) in enumerate(
            (
                ("seq_only", "seq_only|uniform_random", "seq_only", "seq_only"),
                ("co_generation", "co_generation", "seq_only", "co_generation"),
                ("struct_conditioned", "struct_conditioned", "struct_conditioned", "struct_conditioned"),
            )
        )
    ]

    # Axis 5: DPLM-2 vs DPLM-1 (matched 150M / 650M / 3B scales)
    dplm2_vs_dplm1 = [
        _pair_summary(
            f"{d2}_vs_{d1}",
            f"{d2}|seq_only|uniform_random",
            f"{d1}|uniform_random",
            f"{d2}|seq_only",
            d1,
            f"{d2}|seq_only",
            d1,
            "N",
            f"{d2}|seq_only",
            d1,
            seed + 300 + 10 * idx,
        )
        for idx, (d2, d1) in enumerate(DPLM2_TO_DPLM1_PAIRS)
    ]

    # Axis 6: DiMA Euler SDE vs deterministic ODE ablation
    dima_sde_vs_ode = [
        _pair_summary(
            f"{arm}_sde_vs_ode",
            f"{arm}|sde",
            f"{arm}|ode",
            arm,
            arm,
            arm,
            arm,
            "N",
            arm,
            arm,
            seed + 400 + 10 * idx,
        )
        for idx, arm in enumerate(DIMA_ODE_ABLATION_ARMS)
    ]

    # Axis 7: DiMA encoder scale & latent modality contrasts
    dima_encoder_contrasts = [
        _pair_summary(
            label,
            f"{a}|sde",
            f"{b}|sde",
            a,
            b,
            a,
            b,
            "N",
            a,
            b,
            seed + 500 + 10 * idx,
        )
        for idx, (label, a, b) in enumerate(DIMA_ENCODER_CONTRAST_PAIRS)
    ]

    # Axis 8: Continuous latent diffusion (DiMA) vs discrete absorbing diffusion (DPLM-1 / DPLM-2)
    continuous_vs_discrete = [
        _pair_summary(
            label,
            f"{dima_arm}|sde",
            disc_e41,
            dima_arm,
            disc_r5a,
            dima_arm,
            disc_e42,
            "N",
            dima_arm,
            disc_e42,
            seed + 600 + 10 * idx,
        )
        for idx, (label, dima_arm, disc_e41, disc_r5a, disc_e42) in enumerate(
            DIMA_CONTINUOUS_VS_DISCRETE_PAIRS
        )
    ]

    return {
        "schema": "r6_dplm2_mpnn_adaptation_unified_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "bootstrap_draws": int(draws),
            "bootstrap_seed": int(seed),
            "adaptation_axes": [
                "soluble_vs_vanilla_proteinmpnn",
                "ca_only_vs_all_atom_proteinmpnn",
                "proteinmpnn_backbone_noise_ladder",
                "dplm2_bit_650m_vs_dplm2_650m",
                "dplm2_seq_only_vs_dplm1_matched_scale",
                "dima_sde_vs_ode_ablation",
                "dima_encoder_scale_and_modality",
                "continuous_vs_discrete_diffusion",
            ],
        },
        "soluble_vs_vanilla_proteinmpnn": soluble_vs_vanilla,
        "ca_only_vs_all_atom_proteinmpnn": ca_vs_vanilla,
        "proteinmpnn_backbone_noise_ladder": noise_ladders,
        "dplm2_bit_650m_vs_dplm2_650m": dplm2_bit_vs_vq,
        "dplm2_vs_dplm1_matched_scale": dplm2_vs_dplm1,
        "dima_sde_vs_ode_ablation": dima_sde_vs_ode,
        "dima_encoder_scale_and_modality": dima_encoder_contrasts,
        "continuous_vs_discrete_diffusion": continuous_vs_discrete,
    }


__all__ = [
    "DIMA_4_ARMS",
    "DIMA_CONTINUOUS_VS_DISCRETE_PAIRS",
    "DIMA_ENCODER_CONTRAST_PAIRS",
    "DIMA_MAX_RESIDUES",
    "DIMA_ODE_ABLATION_ARMS",
    "DPLM2_4_ARMS",
    "DPLM2_TO_DPLM1_PAIRS",
    "EXP4_1_NEW_SPECS",
    "EXP4_1_NEW_SPEC_MAP",
    "EXP4_1_POOL_SIZE",
    "EXP4_2_NEW_SPECS",
    "EXP4_2_NEW_SPEC_MAP",
    "EXP4_2_POOL_SIZE",
    "EXPANDED_15_STRUCT_ARMS",
    "EXPANDED_19_NEW_ARMS",
    "PROTEINMPNN_11_ARMS",
    "PROTEINMPNN_T10_SENSITIVITY_ARMS",
    "R4_CONT_NEW_SPECS",
    "R4_CONT_NEW_SPEC_MAP",
    "R5A_NEW_SPECS",
    "R5A_NEW_SPEC_MAP",
    "R5A_REPLICATES",
    "R5A_RUNGS",
    "WORK_DIR_REL",
    "WT_BACKBONES_REL",
    "assemble_exp4_1_full",
    "assemble_exp4_2_full",
    "assemble_exp4_3_full",
    "assemble_r4_continuation_full",
    "assemble_r5a_full",
    "assemble_r6_dplm2_mpnn_adaptation",
    "assert_no_infra_leaks",
    "atomic_write_jsonl",
    "deterministic_seed",
    "entry_file_slug",
    "load_r5a_ladder_backbones",
    "paired_bootstrap_diff",
    "parse_pdb_backbone_ncaco",
    "read_jsonl",
    "sample_dima_homolog_continuation",
    "sample_dplm2_homolog_continuation",
    "sample_guided_dima",
    "sample_guided_dplm2",
    "select_exp4_1_reference_targets",
    "seq_sha256",
    "sequence_identity_exact",
    "unpaired_bootstrap_diff",
    "write_checked_artifact",
]
