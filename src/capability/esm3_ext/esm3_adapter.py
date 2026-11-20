"""Offline adapter for EvolutionaryScale/biohub ESM3 (esm3-sm-open-v1, 1.4B).

Guarantees
----------
- 100% offline execution under ``HF_HUB_OFFLINE=1`` by verifying all local
  weight and tokenizer assets up front and overriding ``esm``'s ``data_root``
  hook so ``huggingface_hub.snapshot_download`` is never invoked.
- Safe batched forward pass: explicitly expands every omitted multimodal track
  to batch size ``B`` (avoiding upstream ``ESM3.forward``'s ``(1, L)`` default
  broadcast bug when ``B > 1``).
- Exact deterministic scoring primitives for R1-R4 (single-site masked marginal,
  native 1-pass 20-AA site conditional, exact O(L) leave-one-out PLL with
  arbitrary multimodal tracks or chainbreak ``|`` sequence context), ``p = 0``
  hidden-state extraction from ``model.transformer.blocks``, Track B4 VQ-VAE
  tokenization geometry probes, and deterministic AA20-restricted cosine-
  schedule unmasking for R5.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_SET = frozenset(AA20)
ESM3_MAX_SEQ_LEN = 2046  # 2048 total positions minus <cls> and <eos>

REQUIRED_LOCAL_ASSETS: tuple[str, ...] = (
    "data/weights/esm3_sm_open_v1.pth",
    "data/weights/esm3_structure_encoder_v0.pth",
    "data/weights/esm3_structure_decoder_v0.pth",
    "data/weights/esm3_function_decoder_v0.pth",
    "data/tag_dict_4_safety_filtered.json",
    "data/hyperplanes_8bit_58641.npz",
    "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv",
)

DEFAULT_MODEL_SEARCH_ROOTS: tuple[Path, ...] = (
    Path("/gpfs/jiaotongdamoxing/zhk_zip/models/esm3-sm-open-v1"),
    Path("/home/lzp/models/general_plm_20261120/esm3-sm-open-v1"),
    Path("/Data/lzp/models/general_plm_20261120/esm3-sm-open-v1"),
    Path.home() / "models/general_plm_20261120/esm3-sm-open-v1",
    Path.home() / "models/esm3-sm-open-v1",
)

# Standard 3-letter -> 1-letter amino acid map for PDB/CIF parsing
_AA3_TO_1: dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


@dataclass
class ESM3Handle:
    """Loaded offline ESM3 checkpoint and resolved token-ID metadata."""

    model: Any
    device: torch.device
    dtype: torch.dtype
    model_dir: Path
    aa20_token_ids: tuple[int, ...]
    aa_to_id: dict[str, int]
    id_to_aa: dict[int, str]
    bos_id: int
    eos_id: int
    pad_id: int
    mask_id: int
    chainbreak_id: int
    structure_mask_id: int
    structure_bos_id: int
    structure_eos_id: int
    structure_pad_id: int
    structure_chainbreak_id: int
    n_layers: int
    d_model: int
    facts: dict[str, Any] = field(default_factory=dict)


def resolve_esm3_model_dir(explicit: str | Path | None = None) -> Path:
    """Locate a staged local ``esm3-sm-open-v1`` directory with required assets."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_dir = os.environ.get("ESM3_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    model_root = os.environ.get("MODEL_ROOT")
    if model_root:
        candidates.append(Path(model_root) / "esm3-sm-open-v1")
    candidates.extend(DEFAULT_MODEL_SEARCH_ROOTS)

    for candidate in candidates:
        if candidate.is_dir() and (candidate / REQUIRED_LOCAL_ASSETS[0]).is_file():
            return candidate.resolve()

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find staged offline esm3-sm-open-v1 directory; searched: {searched}"
    )


def verify_offline_assets(model_dir: Path) -> dict[str, int]:
    """Verify all weights and auxiliary tokenizer assets exist locally."""

    sizes: dict[str, int] = {}
    missing: list[str] = []
    for rel in REQUIRED_LOCAL_ASSETS:
        full = model_dir / rel
        if not full.is_file() or full.stat().st_size == 0:
            missing.append(str(full))
        else:
            sizes[rel] = int(full.stat().st_size)
    if missing:
        raise FileNotFoundError(
            "ESM3 offline checkpoint is incomplete; missing required assets:\n  "
            + "\n  ".join(missing)
        )
    return sizes


def _patch_esm_offline_data_root(model_dir: Path) -> None:
    """Redirect all ``esm`` data_root lookups to ``model_dir`` with zero network I/O."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from esm import pretrained as esm_pretrained
    from esm.utils.constants import esm3 as esm3_const

    resolved = Path(model_dir).resolve()

    def _local_data_root(model: str = "esm3") -> Path:
        if model.startswith("esm3"):
            return resolved
        raise ValueError(
            f"Offline ESM3 adapter refuses non-esm3 data_root request: {model!r}"
        )

    esm3_const.data_root = _local_data_root  # type: ignore[assignment]
    esm_pretrained.data_root = _local_data_root  # type: ignore[assignment]


def _quick_file_fingerprint(path: Path, head_bytes: int = 1 << 20) -> str:
    """Deterministic fingerprint from file size + first/last 1 MiB (fast on 2.8 GB .pth)."""

    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(f"size={size}:".encode("ascii"))
    with path.open("rb") as fh:
        h.update(fh.read(head_bytes))
        if size > head_bytes:
            fh.seek(max(0, size - head_bytes))
            h.update(fh.read(head_bytes))
    return h.hexdigest()


def load_esm3_offline(
    model_dir: str | Path | None = None,
    *,
    device: str | torch.device = "cuda:0",
    dtype: str = "bfloat16",
    eager_load_aux: bool = False,
) -> ESM3Handle:
    """Load ``esm3-sm-open-v1`` strictly offline into ``device`` and ``dtype``.

    Parameters
    ----------
    model_dir:
        Path to local staged ``esm3-sm-open-v1`` directory. Auto-resolved if omitted.
    device:
        Target torch device (e.g., ``"cuda:0"``, ``"cpu"``).
    dtype:
        ``"bfloat16"`` (ESM3 native fast precision on Ampere/Hopper) or ``"float32"``
        (repo strict scoring precision; disables TF32 matmul during scoring).
    eager_load_aux:
        If True, immediately loads the local structure encoder/decoder and function
        decoder to verify offline asset integrity end-to-end.
    """

    resolved_dir = resolve_esm3_model_dir(model_dir)
    asset_sizes = verify_offline_assets(resolved_dir)
    _patch_esm_offline_data_root(resolved_dir)

    dev = torch.device(device)
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported ESM3 dtype {dtype!r}; expected 'bfloat16' or 'float32'")

    from esm.pretrained import ESM3_sm_open_v0
    from esm.utils.constants import esm3 as C

    model = ESM3_sm_open_v0(device=dev)
    model = model.eval().to(device=dev, dtype=torch_dtype)

    if eager_load_aux:
        _ = model.get_structure_encoder()
        _ = model.get_structure_decoder()
        _ = model.get_function_decoder()

    seq_tok = model.tokenizers.sequence
    aa_to_id: dict[str, int] = {}
    id_to_aa: dict[int, str] = {}
    for aa in AA20:
        tid = int(seq_tok.convert_tokens_to_ids(aa))
        if tid < 0 or tid == int(seq_tok.unk_token_id):
            raise RuntimeError(f"ESM3 sequence tokenizer missing canonical AA {aa!r}")
        aa_to_id[aa] = tid
        id_to_aa[tid] = aa

    aa20_ids = tuple(aa_to_id[aa] for aa in AA20)
    n_layers = len(model.transformer.blocks)
    d_model = int(model.encoder.sequence_embed.embedding_dim)
    n_params = int(sum(p.numel() for p in model.parameters()))

    facts = {
        "model_name": "esm3-sm-open-v1",
        "model_dir": str(resolved_dir),
        "device": str(dev),
        "dtype_requested": dtype,
        "dtype_active": str(torch_dtype),
        "n_layers": n_layers,
        "d_model": d_model,
        "n_parameters": n_params,
        "offline_enforced": True,
        "main_weights_fingerprint": _quick_file_fingerprint(
            resolved_dir / "data/weights/esm3_sm_open_v1.pth"
        ),
        "asset_sizes_bytes": asset_sizes,
    }

    return ESM3Handle(
        model=model,
        device=dev,
        dtype=torch_dtype,
        model_dir=resolved_dir,
        aa20_token_ids=aa20_ids,
        aa_to_id=aa_to_id,
        id_to_aa=id_to_aa,
        bos_id=int(C.SEQUENCE_BOS_TOKEN),
        eos_id=int(C.SEQUENCE_EOS_TOKEN),
        pad_id=int(C.SEQUENCE_PAD_TOKEN),
        mask_id=int(C.SEQUENCE_MASK_TOKEN),
        chainbreak_id=int(C.SEQUENCE_CHAINBREAK_TOKEN),
        structure_mask_id=int(C.STRUCTURE_MASK_TOKEN),
        structure_bos_id=int(C.STRUCTURE_BOS_TOKEN),
        structure_eos_id=int(C.STRUCTURE_EOS_TOKEN),
        structure_pad_id=int(C.STRUCTURE_PAD_TOKEN),
        structure_chainbreak_id=int(C.STRUCTURE_CHAINBREAK_TOKEN),
        n_layers=n_layers,
        d_model=d_model,
        facts=facts,
    )


def _precision_context(handle: ESM3Handle):
    """Context manager matching requested precision (strict FP32 or BF16 autocast)."""

    if handle.dtype == torch.float32:
        if handle.device.type == "cuda":
            @contextlib.contextmanager
            def _fp32_strict():
                prev_matmul = torch.backends.cuda.matmul.allow_tf32
                prev_cudnn = torch.backends.cudnn.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                try:
                    yield
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = prev_matmul
                    torch.backends.cudnn.allow_tf32 = prev_cudnn
            return _fp32_strict()
        return contextlib.nullcontext()
    if handle.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ------------------------------------------------------------------ Tokenizers


def tokenize_sequence_ids(
    handle: ESM3Handle,
    sequence: str,
    *,
    allow_mask: bool = False,
    allow_chainbreak: bool = False,
) -> torch.Tensor:
    """Encode an AA20 string (optionally containing ``_`` or ``|``) to ``(L+2,)`` IDs."""

    ids = [handle.bos_id]
    for ch in sequence:
        if ch in handle.aa_to_id:
            ids.append(handle.aa_to_id[ch])
        elif allow_mask and ch == "_":
            ids.append(handle.mask_id)
        elif allow_chainbreak and ch == "|":
            ids.append(handle.chainbreak_id)
        else:
            raise ValueError(
                f"Invalid residue character {ch!r} for ESM3 sequence tokenization "
                f"(allow_mask={allow_mask}, allow_chainbreak={allow_chainbreak})"
            )
    ids.append(handle.eos_id)
    if len(ids) > ESM3_MAX_SEQ_LEN + 2:
        raise ValueError(
            f"Tokenized sequence length {len(ids)} exceeds ESM3 max context {ESM3_MAX_SEQ_LEN + 2}"
        )
    return torch.tensor(ids, dtype=torch.long, device=handle.device)


def encode_ss8_tokens(handle: ESM3Handle, ss8: str) -> torch.Tensor:
    """Encode an SS8 string (``GHITEBSC`` and ``_`` for mask) to ``(L+2,)`` tensor."""

    from esm.utils import encoding

    toks = encoding.tokenize_secondary_structure(
        ss8, handle.model.tokenizers.secondary_structure, add_special_tokens=True
    )
    if not isinstance(toks, torch.Tensor):
        toks = torch.tensor(toks, dtype=torch.long)
    return toks.to(device=handle.device, dtype=torch.long)


def encode_structure_tokens(
    handle: ESM3Handle,
    coords_atom37: torch.Tensor | np.ndarray,
    *,
    reference_sequence: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``(L, 37, 3)`` backbone/atom37 coordinates via the local VQ-VAE encoder.

    Returns ``(coords_padded [L+2, 37, 3], structure_tokens [L+2])`` on ``handle.device``.
    Unaligned / missing residues should have ``NaN`` coordinates; those positions
    are explicitly forced to ``STRUCTURE_MASK_TOKEN`` (4096) after encoding.
    """

    from esm.utils import encoding

    if isinstance(coords_atom37, np.ndarray):
        coords_t = torch.as_tensor(coords_atom37, dtype=torch.float32)
    else:
        coords_t = coords_atom37.detach().to(dtype=torch.float32, device="cpu")

    if coords_t.ndim == 3 and coords_t.shape[1] == 3 and coords_t.shape[2] == 3:
        # Promote (L, 3, 3) [N, CA, C] into atom37 (L, 37, 3)
        atom37 = torch.full((coords_t.shape[0], 37, 3), float("nan"), dtype=torch.float32)
        atom37[:, :3, :] = coords_t
        coords_t = atom37
    elif coords_t.ndim != 3 or coords_t.shape[1] != 37 or coords_t.shape[2] != 3:
        raise ValueError(f"Expected coordinates (L, 37, 3) or (L, 3, 3); got {tuple(coords_t.shape)}")

    valid_backbone = torch.isfinite(coords_t[:, :3, :]).all(dim=(-1, -2))  # (L,)
    ref_seq = reference_sequence if reference_sequence is not None else ("A" * coords_t.shape[0])

    # StructureTokenEncoder expects float32 or model device
    struct_enc = handle.model.get_structure_encoder()
    with torch.no_grad():
        coords_pad, _, struct_tokens = encoding.tokenize_structure(
            coords_t,
            struct_enc,
            structure_tokenizer=handle.model.tokenizers.structure,
            reference_sequence=ref_seq,
            add_special_tokens=True,
        )

    coords_pad = coords_pad.to(device=handle.device, dtype=torch.float32)
    struct_tokens = struct_tokens.to(device=handle.device, dtype=torch.long)

    # Guarantee unaligned/missing coordinates remain strictly masked (never leak undefined token)
    missing_mask = ~valid_backbone.to(device=handle.device)
    if missing_mask.any():
        struct_tokens[1:-1] = torch.where(
            missing_mask,
            torch.full_like(struct_tokens[1:-1], handle.structure_mask_id),
            struct_tokens[1:-1],
        )
        coords_pad[1:-1][missing_mask] = float("nan")

    return coords_pad, struct_tokens


def encode_function_annotations(
    handle: ESM3Handle,
    length: int,
    annotations: Sequence[tuple[str, int, int]],
    *,
    reference_sequence: str | None = None,
    strict: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode function keyword/InterPro annotations ``[(label, start_1based, end_1based), ...]``.

    When ``strict=False`` (default), filters out labels absent from ESM3's
    safety-filtered InterPro/TF-IDF/residue vocabularies so callers never crash
    on external accessions (e.g., raw Pfam IDs); falls back to ``("enzyme", 1, length)``
    only if ``annotations`` was non-empty and all labels were filtered out.
    Returns ``(function_tokens [L+2, 8], residue_annotation_tokens [L+2, 16])`` on device.
    """

    import re
    from esm.utils import encoding
    from esm.utils.types import FunctionAnnotation

    ft_tok = handle.model.tokenizers.function
    ra_tok = handle.model.tokenizers.residue_annotations

    def _is_supported(label: str) -> bool:
        m = re.search(r"IPR\d+", label)
        if m and m.group() in ft_tok.interpro_to_index:
            return True
        if label in ft_tok._tfidf.vocab_to_index:
            return True
        if label in ra_tok._labels:
            return True
        return False

    fa_list: list[FunctionAnnotation] = []
    for lbl, s, e in annotations:
        s_i = max(1, min(length, int(s)))
        e_i = max(s_i, min(length, int(e)))
        lbl_s = str(lbl)
        if strict or _is_supported(lbl_s):
            fa_list.append(FunctionAnnotation(label=lbl_s, start=s_i, end=e_i))

    if annotations and not fa_list and not strict and _is_supported("enzyme"):
        fa_list.append(FunctionAnnotation(label="enzyme", start=1, end=length))

    ref_seq = reference_sequence if reference_sequence is not None else ("_" * length)
    func_toks, res_toks = encoding.tokenize_function_annotations(
        fa_list,
        reference_sequence=ref_seq,
        function_tokenizer=ft_tok,
        residue_annotation_tokenizer=ra_tok,
        add_special_tokens=True,
    )
    return (
        func_toks.to(device=handle.device, dtype=torch.long),
        res_toks.to(device=handle.device, dtype=torch.long),
    )


# --------------------------------------------------------------- Structure I/O


def _biotite_sse_to_ss8(sse_arr: np.ndarray) -> str:
    mapping = {"a": "H", "b": "E", "c": "C"}
    return "".join(mapping.get(str(x), "_") for x in sse_arr)


def load_backbone_from_structure_file(
    path: str | Path,
    *,
    chain_id: str | None = None,
) -> dict[str, Any]:
    """Parse `.pdb`, `.pdb.gz`, `.cif`, or `.cif.gz` into AA sequence, atom37 coords, and SS8.

    Uses ``biotite`` directly (no network, works on gzipped AFDB CIF/PDB and CATH PDBs).
    """

    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.pdbx as pdbx_io

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Structure file not found: {p}")

    raw_bytes = gzip.open(p, "rb").read() if p.suffix == ".gz" else p.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    lower_name = p.name.lower()

    if lower_name.endswith(".cif") or lower_name.endswith(".cif.gz"):
        cif_file = pdbx_io.CIFFile.read(io.StringIO(text))
        atoms = pdbx_io.get_structure(cif_file, model=1)
    else:
        pdb_file = pdb_io.PDBFile.read(io.StringIO(text))
        atoms = pdb_io.get_structure(pdb_file, model=1)

    atoms = atoms[struc.filter_amino_acids(atoms)]
    if len(atoms) == 0:
        raise ValueError(f"No amino acid atoms found in {p}")

    chains = sorted(set(str(c) for c in atoms.chain_id))
    chosen_chain = chain_id if (chain_id and chain_id in chains) else chains[0]
    atoms = atoms[atoms.chain_id == chosen_chain]

    res_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    n_res = len(res_starts) - 1
    seq_chars: list[str] = []
    coords_atom37 = np.full((n_res, 37, 3), np.nan, dtype=np.float32)

    # Atom37 canonical indices for backbone N=0, CA=1, C=2, CB=3, O=4
    atom_to_idx = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4}
    for i in range(n_res):
        sl = atoms[res_starts[i] : res_starts[i + 1]]
        res_name = str(sl.res_name[0]).strip().upper()
        seq_chars.append(_AA3_TO_1.get(res_name, "X"))
        for aname, aidx in atom_to_idx.items():
            hits = sl[sl.atom_name == aname]
            if len(hits) > 0:
                coords_atom37[i, aidx] = hits.coord[0].astype(np.float32)

    try:
        sse = struc.annotate_sse(atoms)
        ss8 = _biotite_sse_to_ss8(sse)
        if len(ss8) != n_res:
            ss8 = "_" * n_res
    except Exception:
        ss8 = "_" * n_res

    seq = "".join(seq_chars)
    return {
        "path": str(p),
        "chain_id": chosen_chain,
        "sequence": seq,
        "coords_atom37": coords_atom37,
        "ss8": ss8,
        "length": n_res,
    }


# ------------------------------------------------------------ Safe Batched Forward


def _expand_batch(tensor: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand((batch_size,) + tuple(tensor.shape[1:])).contiguous()
    raise ValueError(f"Cannot broadcast batch dim {tensor.shape[0]} to {batch_size}")


def forward_tracks(
    handle: ESM3Handle,
    *,
    sequence_tokens: torch.Tensor,
    structure_tokens: torch.Tensor | None = None,
    structure_coords: torch.Tensor | None = None,
    ss8_tokens: torch.Tensor | None = None,
    sasa_tokens: torch.Tensor | None = None,
    function_tokens: torch.Tensor | None = None,
    residue_annotation_tokens: torch.Tensor | None = None,
    average_plddt: torch.Tensor | None = None,
    per_res_plddt: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    enable_grad: bool = False,
) -> Any:
    """Safe batched forward pass through ``handle.model`` returning ``ESMOutput``.

    Always expands every track explicitly to ``(B, L, ...)`` so ``B > 1`` never hits
    upstream ``ESM3.forward``'s ``(1, L)`` default bug.
    """

    from esm.utils.constants import esm3 as C

    if sequence_tokens.ndim == 1:
        sequence_tokens = sequence_tokens.unsqueeze(0)
    B, L = sequence_tokens.shape
    dev = handle.device
    sequence_tokens = sequence_tokens.to(device=dev, dtype=torch.long)

    def _norm_2d(t: torch.Tensor | None, fill: int) -> torch.Tensor:
        if t is None:
            return torch.full((B, L), fill, dtype=torch.long, device=dev)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return _expand_batch(t.to(device=dev, dtype=torch.long), B)

    struct_toks = _norm_2d(structure_tokens, C.STRUCTURE_MASK_TOKEN)
    ss8_toks = _norm_2d(ss8_tokens, C.SS8_PAD_TOKEN)
    sasa_toks = _norm_2d(sasa_tokens, C.SASA_PAD_TOKEN)

    if function_tokens is None:
        func_toks = torch.full((B, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=dev)
    else:
        ft = function_tokens.unsqueeze(0) if function_tokens.ndim == 2 else function_tokens
        func_toks = _expand_batch(ft.to(device=dev, dtype=torch.long), B)

    if residue_annotation_tokens is None:
        res_toks = torch.full((B, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=dev)
    else:
        rt = (
            residue_annotation_tokens.unsqueeze(0)
            if residue_annotation_tokens.ndim == 2
            else residue_annotation_tokens
        )
        res_toks = _expand_batch(rt.to(device=dev, dtype=torch.long), B)

    if structure_coords is None:
        coords = torch.full((B, L, 3, 3), float("nan"), dtype=torch.float32, device=dev)
        if per_res_plddt is None:
            per_res = torch.zeros((B, L), dtype=torch.float32, device=dev)
        else:
            pr = per_res_plddt.unsqueeze(0) if per_res_plddt.ndim == 1 else per_res_plddt
            per_res = _expand_batch(pr.to(device=dev, dtype=torch.float32), B)
    else:
        sc = structure_coords.unsqueeze(0) if structure_coords.ndim == 3 else structure_coords
        coords = _expand_batch(sc[..., :3, :].to(device=dev, dtype=torch.float32), B)
        if per_res_plddt is None:
            per_res = torch.isfinite(coords).all(dim=-1).all(dim=-1).float()
        else:
            pr = per_res_plddt.unsqueeze(0) if per_res_plddt.ndim == 1 else per_res_plddt
            per_res = _expand_batch(pr.to(device=dev, dtype=torch.float32), B)

    if average_plddt is None:
        avg_plddt = torch.ones((B, L), dtype=torch.float32, device=dev)
    else:
        ap = average_plddt.unsqueeze(0) if average_plddt.ndim == 1 else average_plddt
        avg_plddt = _expand_batch(ap.to(device=dev, dtype=torch.float32), B)

    cid = _norm_2d(chain_id, 0)
    if sequence_id is not None:
        sid = sequence_id.unsqueeze(0) if sequence_id.ndim == 1 else sequence_id
        sid = _expand_batch(sid.to(device=dev), B)
    else:
        sid = None

    grad_ctx = contextlib.nullcontext() if enable_grad else torch.no_grad()
    with grad_ctx, _precision_context(handle):
        return handle.model.forward(
            sequence_tokens=sequence_tokens,
            structure_tokens=struct_toks,
            ss8_tokens=ss8_toks,
            sasa_tokens=sasa_toks,
            function_tokens=func_toks,
            residue_annotation_tokens=res_toks,
            average_plddt=avg_plddt,
            per_res_plddt=per_res,
            structure_coords=coords,
            chain_id=cid,
            sequence_id=sid,
        )


# --------------------------------------------------------------- R1-R4 Scoring


def single_site_marginals(
    handle: ESM3Handle,
    sequence: str,
    sites: Sequence[int],
    *,
    batch_size: int = 32,
    restrict_aa20: bool = False,
    **track_kwargs: Any,
) -> dict[int, np.ndarray]:
    """Compute masked log-probabilities at 0-based residue indices ``sites``.

    Masks one site per forward pass (batched). Returns ``{site_0based: logprobs_aa20}``
    where ``logprobs_aa20`` has shape ``(20,)`` aligned to ``AA20 = 'ACDEFGHIKLMNPQRSTVWY'``.
    When ``restrict_aa20=False`` (default), ``log_softmax`` is computed over the full
    64-token vocabulary before slicing AA20 (matching repo ``MaskedScorer`` contract);
    when ``restrict_aa20=True``, ``log_softmax`` is restricted to AA20 (for 1-pass
    ``family_conditionals`` 20-way proper log-score).
    """

    unique_sites = sorted({int(s) for s in sites})
    L = len(sequence)
    for s in unique_sites:
        if not 0 <= s < L:
            raise ValueError(f"Site {s} out of range 0..{L - 1}")
    if not unique_sites:
        return {}

    base_ids = tokenize_sequence_ids(handle, sequence)
    aa_idx = torch.tensor(handle.aa20_token_ids, dtype=torch.long, device=handle.device)
    out: dict[int, np.ndarray] = {}

    for start in range(0, len(unique_sites), batch_size):
        chunk = unique_sites[start : start + batch_size]
        batch = base_ids.unsqueeze(0).repeat(len(chunk), 1)
        for row, pos in enumerate(chunk):
            batch[row, pos + 1] = handle.mask_id
        logits = forward_tracks(handle, sequence_tokens=batch, **track_kwargs).sequence_logits.float()
        for row, pos in enumerate(chunk):
            site_logits = logits[row, pos + 1]
            if restrict_aa20:
                lp = F.log_softmax(site_logits.index_select(0, aa_idx), dim=-1)
            else:
                lp = F.log_softmax(site_logits, dim=-1).index_select(0, aa_idx)
            arr = lp.detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(arr).all():
                raise RuntimeError(f"Non-finite ESM3 masked marginal at site {pos}")
            out[pos] = arr
    return out


def native_site_conditionals(
    handle: ESM3Handle,
    sequence: str,
    site_0based: int,
    observed_aa: str,
) -> dict[str, float]:
    """1-pass native masked 20-AA conditional log-probability for R1 ``family_conditionals``."""

    if observed_aa not in AA20_SET:
        raise ValueError(f"Observed residue {observed_aa!r} is not in AA20")
    lp = single_site_marginals(
        handle, sequence, [site_0based], batch_size=1, restrict_aa20=True
    )[site_0based]
    aa_index = AA20.index(observed_aa)
    obs_logp = float(lp[aa_index])
    entropy_nats = float(-np.sum(np.exp(lp) * lp))
    return {
        "observed_logp_nats": obs_logp,
        "entropy_nats": entropy_nats,
        "entropy_bits": entropy_nats / math.log(2.0),
    }


def exact_pll(
    handle: ESM3Handle,
    sequence: str,
    *,
    batch_size: int = 32,
    positions: Sequence[int] | None = None,
    score_span: tuple[int, int] | None = None,
    allow_chainbreak: bool = False,
    **track_kwargs: Any,
) -> dict[str, Any]:
    """Exact leave-one-out masked pseudo-log-likelihood (full-vocab & AA20-restricted).

    Parameters
    ----------
    sequence:
        AA20 sequence (or ``homolog|target`` when ``allow_chainbreak=True``).
    positions:
        Optional explicit list of 0-based indices into ``sequence`` to mask & score.
    score_span:
        Optional ``(start, stop)`` half-open slice of 0-based indices in ``sequence``
        (e.g., target span after a chainbreak ``|``). Ignored if ``positions`` is set.
    track_kwargs:
        Optional multimodal tracks (``structure_tokens``, ``structure_coords``,
        ``ss8_tokens``, ``function_tokens``, ``chain_id``, etc.) of length ``len(sequence)+2``.
    """

    base_ids = tokenize_sequence_ids(
        handle, sequence, allow_chainbreak=allow_chainbreak
    )
    if positions is not None:
        target_positions = [int(p) for p in positions]
    elif score_span is not None:
        s0, s1 = int(score_span[0]), int(score_span[1])
        target_positions = list(range(s0, s1))
    else:
        target_positions = [i for i, ch in enumerate(sequence) if ch in AA20_SET]

    if not target_positions:
        raise ValueError("exact_pll received zero scorable positions")

    aa_idx = torch.tensor(handle.aa20_token_ids, dtype=torch.long, device=handle.device)
    aa_pos_map = {tid: idx for idx, tid in enumerate(handle.aa20_token_ids)}

    full_terms = np.empty(len(target_positions), dtype=np.float64)
    aa20_terms = np.empty(len(target_positions), dtype=np.float64)

    for start in range(0, len(target_positions), batch_size):
        chunk = target_positions[start : start + batch_size]
        batch = base_ids.unsqueeze(0).repeat(len(chunk), 1)
        for row, pos in enumerate(chunk):
            batch[row, pos + 1] = handle.mask_id
        logits = forward_tracks(handle, sequence_tokens=batch, **track_kwargs).sequence_logits.float()
        logprobs_full = F.log_softmax(logits, dim=-1)
        for row, pos in enumerate(chunk):
            true_tid = int(base_ids[pos + 1])
            full_terms[start + row] = float(logprobs_full[row, pos + 1, true_tid])
            aa_lp = F.log_softmax(logits[row, pos + 1].index_select(0, aa_idx), dim=-1)
            aa20_terms[start + row] = float(aa_lp[aa_pos_map[true_tid]])

    if not (np.isfinite(full_terms).all() and np.isfinite(aa20_terms).all()):
        raise RuntimeError("Non-finite term encountered in ESM3 exact_pll")

    n_scored = len(target_positions)
    pll_sum = float(full_terms.sum())
    aa20_pll_sum = float(aa20_terms.sum())
    return {
        "length": len(sequence),
        "scored_tokens": n_scored,
        "pll_sum_nats": pll_sum,
        "nll_sum_nats": -pll_sum,
        "mean_nll_per_residue_nats": -pll_sum / n_scored,
        "aa20_nll_sum_nats": -aa20_pll_sum,
        "aa20_mean_nll_per_residue_nats": -aa20_pll_sum / n_scored,
        "per_position_logprobs": full_terms.tolist(),
        "likelihood_kind": "masked_pseudo_log_likelihood",
    }


# ------------------------------------------------ Hidden-State Readouts (p=0)


def extract_hidden_states(
    handle: ESM3Handle,
    sequences: Sequence[str],
    *,
    block_indices: Sequence[int] | None = None,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Extract ``p = 0`` unmasked hidden-state summaries from ``model.transformer.blocks``.

    Default blocks match repo readout contract: middle ``(n_layers - 1) // 2`` (23)
    and final ``n_layers - 1`` (47). Returns ``mean``, ``cls``, and ``last`` pooled
    float32 arrays of shape ``(N, d_model)`` per hooked block.
    """

    n_layers = handle.n_layers
    if block_indices is None:
        chosen = ((n_layers - 1) // 2, n_layers - 1)
    else:
        chosen = tuple(int(i) for i in block_indices)
        for idx in chosen:
            if not 0 <= idx < n_layers:
                raise ValueError(f"Block index {idx} out of range 0..{n_layers - 1}")

    n_seq = len(sequences)
    results: dict[str, np.ndarray] = {}
    for pos_idx, blk in enumerate(chosen):
        tag = "middle" if pos_idx == 0 else ("final" if pos_idx == 1 else f"block{blk}")
        for pool in ("mean", "cls", "last"):
            results[f"{tag}_{pool}"] = np.empty((n_seq, handle.d_model), dtype=np.float32)

    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def _make_hook(b_idx: int):
        def _hook(_module: Any, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            captured[b_idx] = t.detach()
        return _hook

    try:
        for blk in chosen:
            hooks.append(handle.model.transformer.blocks[blk].register_forward_hook(_make_hook(blk)))

        # Process equal-length or padded batches safely via singleton/length-bucketed forwards
        for start in range(0, n_seq, batch_size):
            sub = list(sequences[start : start + batch_size])
            lengths = [len(s) for s in sub]
            max_tok = max(lengths) + 2
            batch = torch.full((len(sub), max_tok), handle.pad_id, dtype=torch.long, device=handle.device)
            seq_id = torch.zeros((len(sub), max_tok), dtype=torch.bool, device=handle.device)
            for row, seq in enumerate(sub):
                ids = tokenize_sequence_ids(handle, seq)
                batch[row, : len(ids)] = ids
                seq_id[row, : len(ids)] = True

            captured.clear()
            _ = forward_tracks(handle, sequence_tokens=batch, sequence_id=seq_id)

            for pos_idx, blk in enumerate(chosen):
                tag = "middle" if pos_idx == 0 else ("final" if pos_idx == 1 else f"block{blk}")
                h = captured[blk].float()  # (B, max_tok, d_model)
                for row, length in enumerate(lengths):
                    res_h = h[row, 1 : 1 + length]
                    cls_h = h[row, 0]
                    last_h = h[row, length]
                    out_row = start + row
                    results[f"{tag}_mean"][out_row] = res_h.mean(dim=0).cpu().numpy()
                    results[f"{tag}_cls"][out_row] = cls_h.cpu().numpy()
                    results[f"{tag}_last"][out_row] = last_h.cpu().numpy()
    finally:
        for hk in hooks:
            hk.remove()

    for k, arr in results.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite hidden state in {k}")
    return results


# ----------------------------------------- Track B4 VQ-VAE Tokenization Probe


def _roy_vetterli_effective_rank_ratio(matrix: np.ndarray) -> float:
    """Normalized Roy-Vetterli effective rank ``exp(H_vN(Sigma)) / d``."""

    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return float("nan")
    x = matrix.astype(np.float64) - matrix.mean(axis=0, keepdims=True)
    _, svals, _ = np.linalg.svd(x, full_matrices=False)
    eig = svals ** 2
    total = float(eig.sum())
    if total <= 0.0:
        return 0.0
    p = eig / total
    p = p[p > 1e-15]
    entropy = float(-np.sum(p * np.log(p)))
    return float(math.exp(entropy) / matrix.shape[1])


def probe_vqvae_tokenization_geometry(
    handle: ESM3Handle,
    sequences: Sequence[str],
    coords_list: Sequence[np.ndarray],
    *,
    block_indices: Sequence[int] = (0, 12, 23, 35, 47),
    max_tokens_per_protein: int = 128,
) -> dict[str, Any]:
    """Track B4 probe: compare ESM3 ``seq_only`` vs ``3d_conditioned`` residual gradient
    effective rank ``r_eff(l) / d_model`` and VQ-VAE 4096-codebook entropy/utilization.
    """

    if len(sequences) != len(coords_list):
        raise ValueError("sequences and coords_list must have equal length")

    all_struct_codes: list[int] = []
    grad_buffers: dict[str, dict[int, list[np.ndarray]]] = {
        "seq_only": {int(b): [] for b in block_indices},
        "3d_conditioned": {int(b): [] for b in block_indices},
    }
    hidden_cos_shifts: dict[int, list[float]] = {int(b): [] for b in block_indices}

    for seq, coords in zip(sequences, coords_list):
        L = len(seq)
        seq_ids = tokenize_sequence_ids(handle, seq)
        coords_pad, struct_toks = encode_structure_tokens(
            handle, coords, reference_sequence=seq
        )
        valid_codes = struct_toks[1 : 1 + L].detach().cpu().numpy()
        valid_codes = valid_codes[(valid_codes >= 0) & (valid_codes < 4096)]
        all_struct_codes.extend(int(c) for c in valid_codes)

        # Evaluate residue-level masked-loss residual gradients at sampled positions
        rng_step = max(1, L // max_tokens_per_protein)
        sample_idx = list(range(0, L, rng_step))[:max_tokens_per_protein]

        mode_hiddens: dict[str, dict[int, np.ndarray]] = {}
        for mode in ("seq_only", "3d_conditioned"):
            acts: dict[int, torch.Tensor] = {}
            hooks = []

            def _make_retain_hook(b_idx: int):
                def _hook(_mod: Any, _inp: Any, out: Any) -> None:
                    t = out[0] if isinstance(out, tuple) else out
                    if t.requires_grad:
                        t.retain_grad()
                    acts[b_idx] = t
                return _hook

            try:
                for b in block_indices:
                    hooks.append(
                        handle.model.transformer.blocks[int(b)].register_forward_hook(
                            _make_retain_hook(int(b))
                        )
                    )
                handle.model.zero_grad(set_to_none=True)
                kwargs: dict[str, Any] = {}
                if mode == "3d_conditioned":
                    kwargs["structure_tokens"] = struct_toks
                    kwargs["structure_coords"] = coords_pad
                # Lightly mask 15% deterministic sites for non-trivial sequence cross-entropy
                masked_ids = seq_ids.clone()
                mask_pos = [p for idx_i, p in enumerate(sample_idx) if idx_i % 6 == 0] or [0]
                for mp in mask_pos:
                    masked_ids[mp + 1] = handle.mask_id
                out = forward_tracks(
                    handle, sequence_tokens=masked_ids, enable_grad=True, **kwargs
                )
                logits = out.sequence_logits[0].float()  # (L+2, 64)
                pos_t = torch.tensor([p + 1 for p in mask_pos], dtype=torch.long, device=handle.device)
                targets = seq_ids.index_select(0, pos_t)
                loss = F.cross_entropy(logits.index_select(0, pos_t), targets)
                loss.backward()

                mode_hiddens[mode] = {}
                for b in block_indices:
                    bi = int(b)
                    t = acts[bi]
                    sel = torch.tensor([p + 1 for p in sample_idx], dtype=torch.long, device=handle.device)
                    mode_hiddens[mode][bi] = t[0].detach().float().index_select(0, sel).cpu().numpy()
                    if t.grad is not None:
                        g = t.grad[0].detach().float().index_select(0, sel).cpu().numpy()
                        grad_buffers[mode][bi].append(g)
            finally:
                for hk in hooks:
                    hk.remove()
                handle.model.zero_grad(set_to_none=True)

        for b in block_indices:
            bi = int(b)
            h0 = mode_hiddens["seq_only"][bi]
            h1 = mode_hiddens["3d_conditioned"][bi]
            n0 = np.linalg.norm(h0, axis=-1, keepdims=True) + 1e-12
            n1 = np.linalg.norm(h1, axis=-1, keepdims=True) + 1e-12
            cos = np.sum((h0 / n0) * (h1 / n1), axis=-1)
            hidden_cos_shifts[bi].append(float(np.mean(cos)))

    # Codebook statistics
    if all_struct_codes:
        counts = np.bincount(np.asarray(all_struct_codes, dtype=np.int64), minlength=4096)
        used = int((counts > 0).sum())
        probs = counts[counts > 0].astype(np.float64) / float(len(all_struct_codes))
        codebook_entropy_bits = float(-np.sum(probs * np.log2(probs)))
    else:
        used = 0
        codebook_entropy_bits = 0.0

    rank_summary: dict[str, dict[str, float]] = {"seq_only": {}, "3d_conditioned": {}}
    for mode in ("seq_only", "3d_conditioned"):
        for b in block_indices:
            bi = int(b)
            mats = grad_buffers[mode][bi]
            if mats:
                stacked = np.concatenate(mats, axis=0)
                rank_summary[mode][str(bi)] = _roy_vetterli_effective_rank_ratio(stacked)
            else:
                rank_summary[mode][str(bi)] = float("nan")

    return {
        "n_proteins": len(sequences),
        "n_structure_tokens_total": len(all_struct_codes),
        "vqvae_codebook_size": 4096,
        "vqvae_unique_codes_used": used,
        "vqvae_codebook_utilization": float(used / 4096.0),
        "vqvae_entropy_bits": codebook_entropy_bits,
        "vqvae_max_entropy_bits": float(math.log2(4096)),
        "gradient_effective_rank_ratio": rank_summary,
        "seq_vs_3d_hidden_cosine_mean": {
            str(int(b)): float(np.mean(hidden_cos_shifts[int(b)])) if hidden_cos_shifts[int(b)] else float("nan")
            for b in block_indices
        },
    }


# -------------------------------------------- R5 Cosine-Schedule Unmasking


def _nucleus_logprobs(logprobs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1]")
    if top_p == 1.0:
        return logprobs
    ordered, order = torch.sort(logprobs, descending=True, dim=-1)
    probs = ordered.exp()
    cumulative = torch.cumsum(probs, dim=-1)
    keep = (cumulative - probs) < top_p
    keep[..., 0] = True
    masked = torch.full_like(logprobs, float("-inf"))
    masked.scatter_(-1, order, torch.where(keep, ordered, torch.full_like(ordered, float("-inf"))))
    return F.log_softmax(masked, dim=-1)


def _cosine_unmask_steps(total_masked: int, num_steps: int) -> list[int]:
    """Compute per-step commit counts summing to ``total_masked`` along a cosine schedule."""

    steps = max(1, min(int(num_steps), int(total_masked)))
    counts: list[int] = []
    committed = 0
    for step in range(1, steps + 1):
        if step == steps:
            target = total_masked
        else:
            frac = 1.0 - math.cos(0.5 * math.pi * (step / steps))
            target = min(total_masked, max(committed + 1, int(round(total_masked * frac))))
            # Reserve at least 1 token per remaining step if steps == total_masked
            remaining_steps = steps - step
            if total_masked - target < remaining_steps and total_masked >= steps:
                target = total_masked - remaining_steps
        n_now = max(0, target - committed)
        if n_now > 0:
            counts.append(n_now)
            committed += n_now
    if committed < total_masked:
        counts[-1] += total_masked - committed
    return counts


def _iterative_cosine_unmask(
    handle: ESM3Handle,
    ids: torch.Tensor,
    masked_positions: Sequence[int],
    *,
    seed: int,
    num_steps: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    **track_kwargs: Any,
) -> tuple[str, list[int]]:
    """Deterministic AA20-restricted cosine-schedule iterative unmasking."""

    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive")

    cur = ids.clone().to(device=handle.device, dtype=torch.long)
    pending = list(int(p) for p in masked_positions)
    for pos in pending:
        cur[pos + 1] = handle.mask_id

    support = torch.tensor(handle.aa20_token_ids, dtype=torch.long, device=handle.device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    step_counts = _cosine_unmask_steps(len(pending), num_steps)
    commit_order: list[int] = []

    with torch.no_grad():
        for n_commit in step_counts:
            if not pending:
                break
            logits = forward_tracks(handle, sequence_tokens=cur.unsqueeze(0), **track_kwargs).sequence_logits.float()[0]
            rows = torch.tensor([p + 1 for p in pending], dtype=torch.long, device=handle.device)
            aa_logits = logits.index_select(0, rows).index_select(1, support) / float(temperature)
            logp = _nucleus_logprobs(F.log_softmax(aa_logits, dim=-1), top_p)
            probs = logp.exp().cpu()
            sampled_idx = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
            conf = probs.gather(1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            k_take = min(n_commit, len(pending))
            top_indices = torch.topk(conf, k=k_take, largest=True).indices.tolist()
            # Commit selected positions (sort descending when popping from pending)
            chosen_pairs = [(pending[i], int(support[int(sampled_idx[i])])) for i in top_indices]
            for i in sorted(top_indices, reverse=True):
                pending.pop(i)
            for pos, tok_id in chosen_pairs:
                cur[pos + 1] = tok_id
                commit_order.append(pos)

    L = len(cur) - 2
    seq = "".join(handle.id_to_aa[int(cur[i + 1])] for i in range(L))
    if len(seq) != L or set(seq) - AA20_SET:
        raise RuntimeError("ESM3 cosine unmasking produced invalid length or non-AA20 symbol")
    return seq, commit_order


def infill_window_cosine(
    handle: ESM3Handle,
    parent: str,
    *,
    start: int,
    extent: int,
    seed: int,
    num_steps: int | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    **track_kwargs: Any,
) -> dict[str, Any]:
    """R5a window infilling draw (``k = 1, 2, 5, 10, 20, 40``) with both flanks visible."""

    if extent < 1:
        raise ValueError("extent must be >= 1")
    stop = start + extent
    if not 1 <= start < stop <= len(parent) - 1:
        raise ValueError(f"window [{start}, {stop}) is not strictly interior to parent of length {len(parent)}")

    base_ids = tokenize_sequence_ids(handle, parent)
    steps = extent if num_steps is None else int(num_steps)
    full_seq, order = _iterative_cosine_unmask(
        handle,
        base_ids,
        list(range(start, stop)),
        seed=seed,
        num_steps=steps,
        temperature=temperature,
        top_p=top_p,
        **track_kwargs,
    )
    return {
        "status": "filled",
        "sequence": full_seq,
        "window": full_seq[start:stop],
        "commit_order": order,
        "num_steps": min(steps, extent),
        "sampler": "esm3_cosine_max_confidence_unmasking",
        "support": "aa20_restricted_softmax",
    }


def generate_sequence_cosine(
    handle: ESM3Handle,
    length: int,
    *,
    seed: int,
    num_steps: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    **track_kwargs: Any,
) -> dict[str, Any]:
    """100%-masked cosine-schedule generation for R5a ``full_length_matched``,
    R5b de novo seq-only, and R5c 3D/SS8/function-conditioned generation.
    """

    if not 1 <= length <= ESM3_MAX_SEQ_LEN:
        raise ValueError(f"Requested generation length {length} outside 1..{ESM3_MAX_SEQ_LEN}")

    ids = torch.full((length + 2,), handle.mask_id, dtype=torch.long, device=handle.device)
    ids[0] = handle.bos_id
    ids[-1] = handle.eos_id
    steps = max(1, min(int(num_steps), length))
    seq, order = _iterative_cosine_unmask(
        handle,
        ids,
        list(range(length)),
        seed=seed,
        num_steps=steps,
        temperature=temperature,
        top_p=top_p,
        **track_kwargs,
    )
    return {
        "status": "filled",
        "sequence": seq,
        "length": length,
        "commit_order": order,
        "num_steps": steps,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "sampler": "esm3_cosine_max_confidence_unmasking",
        "support": "aa20_restricted_softmax",
    }
