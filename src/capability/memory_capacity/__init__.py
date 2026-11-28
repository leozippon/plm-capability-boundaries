"""Native parametric storage capacity measurement engine for AR and MLM Pre-LN Transformers.

Pure-Python and NumPy exports from ``spec`` and ``data`` load eagerly so CPU-only analysis
tools do not import PyTorch; GPU model, evaluation, and training symbols load lazily via PEP 562.
"""
from __future__ import annotations

import importlib
from typing import Any

from .data import (
    DEFAULT_DATA_CHUNK_SIZE,
    compute_row_hashes_uint64,
    dataset_cache_filename,
    generate_iid_aa_dataset,
    generate_iid_sequences,
    verify_and_deduplicate_sequences,
)
from .spec import (
    C_GRID_PRIOR_BITS_PER_PARAM,
    CAMPAIGN_TRACKS,
    CAPACITY_MULTIPLIERS,
    EXPOSURE_CHECKPOINTS,
    FORBIDDEN_INFRA_RE,
    INPUT_VOCAB_SIZE,
    LN2,
    LOG2_20,
    PARADIGMS,
    PRECISIONS,
    SCALE_ORDER,
    SCALE_SPECS,
    SEQ_LEN,
    SPECIAL_TOKEN_ID,
    VOCAB_AA,
    RunSpec,
    ScaleSpec,
    assert_no_forbidden_infra_strings,
    build_campaign_runs,
    choose_global_batch_size,
    dataset_entropy_bits,
    get_campaign_run_map,
    gibbs_memorizable_bits,
    null_ordered_entropy_bits,
    slugify_token,
    solve_sequence_count_for_entropy,
)

_LAZY_SUBMODULES: dict[str, str] = {
    # model
    "RMSNorm": ".model",
    "SelfAttention": ".model",
    "SwiGLUFFN": ".model",
    "TransformerBlock": ".model",
    "UniformVocabTransformer": ".model",
    "apply_rotary_pos_emb": ".model",
    "count_model_parameters": ".model",
    "precompute_rope_cos_sin": ".model",
    # evaluation
    "DEFAULT_EVAL_MAX_SEQUENCES_MLM": ".evaluation",
    "DEFAULT_LOO_MAX_SEQUENCES_MLM": ".evaluation",
    "DEFAULT_LOO_PROBES_PER_SEQ": ".evaluation",
    "MLM_NUM_STRATA": ".evaluation",
    "compute_mlm_stratum_mask_counts": ".evaluation",
    "deterministic_mask_for_stratum": ".evaluation",
    "evaluate_ar_joint_bits": ".evaluation",
    "evaluate_mlm_joint_bits_16pt": ".evaluation",
    "evaluate_mlm_loo_diagnostic": ".evaluation",
    "evaluate_run_checkpoint": ".evaluation",
    "summarize_memorization_metrics": ".evaluation",
    # training
    "CHECKPOINT_SCHEMA": ".training",
    "DistributedContext": ".training",
    "RUN_RECEIPT_SCHEMA": ".training",
    "build_optimizer": ".training",
    "compute_ar_batch_loss": ".training",
    "compute_cosine_lr": ".training",
    "compute_mlm_batch_loss": ".training",
    "compute_peak_lr": ".training",
    "compute_warmup_steps": ".training",
    "execute_run": ".training",
    "find_resume_checkpoint": ".training",
    "run_training": ".training",
    "sample_mlm_stratified_batch": ".training",
    "setup_distributed": ".training",
    "train_memory_capacity_run": ".training",
}


def __getattr__(name: str) -> Any:
    mod_rel = _LAZY_SUBMODULES.get(name)
    if mod_rel is not None:
        mod = importlib.import_module(mod_rel, package=__name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CAMPAIGN_TRACKS",
    "CAPACITY_MULTIPLIERS",
    "CHECKPOINT_SCHEMA",
    "C_GRID_PRIOR_BITS_PER_PARAM",
    "DEFAULT_DATA_CHUNK_SIZE",
    "DEFAULT_EVAL_MAX_SEQUENCES_MLM",
    "DEFAULT_LOO_MAX_SEQUENCES_MLM",
    "DEFAULT_LOO_PROBES_PER_SEQ",
    "DistributedContext",
    "EXPOSURE_CHECKPOINTS",
    "FORBIDDEN_INFRA_RE",
    "INPUT_VOCAB_SIZE",
    "LN2",
    "LOG2_20",
    "MLM_NUM_STRATA",
    "PARADIGMS",
    "PRECISIONS",
    "RMSNorm",
    "RUN_RECEIPT_SCHEMA",
    "RunSpec",
    "SCALE_ORDER",
    "SCALE_SPECS",
    "SEQ_LEN",
    "SPECIAL_TOKEN_ID",
    "ScaleSpec",
    "SelfAttention",
    "SwiGLUFFN",
    "TransformerBlock",
    "UniformVocabTransformer",
    "VOCAB_AA",
    "apply_rotary_pos_emb",
    "assert_no_forbidden_infra_strings",
    "build_campaign_runs",
    "build_optimizer",
    "choose_global_batch_size",
    "compute_ar_batch_loss",
    "compute_cosine_lr",
    "compute_mlm_batch_loss",
    "compute_mlm_stratum_mask_counts",
    "compute_peak_lr",
    "compute_row_hashes_uint64",
    "compute_warmup_steps",
    "count_model_parameters",
    "dataset_cache_filename",
    "dataset_entropy_bits",
    "deterministic_mask_for_stratum",
    "evaluate_ar_joint_bits",
    "evaluate_mlm_joint_bits_16pt",
    "evaluate_mlm_loo_diagnostic",
    "evaluate_run_checkpoint",
    "execute_run",
    "find_resume_checkpoint",
    "generate_iid_aa_dataset",
    "generate_iid_sequences",
    "get_campaign_run_map",
    "gibbs_memorizable_bits",
    "null_ordered_entropy_bits",
    "precompute_rope_cos_sin",
    "run_training",
    "sample_mlm_stratified_batch",
    "setup_distributed",
    "slugify_token",
    "solve_sequence_count_for_entropy",
    "summarize_memorization_metrics",
    "train_memory_capacity_run",
    "verify_and_deduplicate_sequences",
]
