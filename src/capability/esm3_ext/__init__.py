"""Isolated offline ESM3 (esm3-sm-open-v1) adapter and R1-R5 protocol helpers.

Designed to execute inside ``runtimes/esm3/bin/python`` with ``HF_HUB_OFFLINE=1``
and zero network access at load, encode, score, or generation time.
"""

from .esm3_adapter import (
    AA20,
    ESM3_MAX_SEQ_LEN,
    ESM3Handle,
    encode_function_annotations,
    encode_ss8_tokens,
    encode_structure_tokens,
    exact_pll,
    extract_hidden_states,
    forward_tracks,
    generate_sequence_cosine,
    infill_window_cosine,
    load_backbone_from_structure_file,
    load_esm3_offline,
    native_site_conditionals,
    probe_vqvae_tokenization_geometry,
    resolve_esm3_model_dir,
    single_site_marginals,
    tokenize_sequence_ids,
)

__all__ = [
    "AA20",
    "ESM3_MAX_SEQ_LEN",
    "ESM3Handle",
    "encode_function_annotations",
    "encode_ss8_tokens",
    "encode_structure_tokens",
    "exact_pll",
    "extract_hidden_states",
    "forward_tracks",
    "generate_sequence_cosine",
    "infill_window_cosine",
    "load_backbone_from_structure_file",
    "load_esm3_offline",
    "native_site_conditionals",
    "probe_vqvae_tokenization_geometry",
    "resolve_esm3_model_dir",
    "single_site_marginals",
    "tokenize_sequence_ids",
]
