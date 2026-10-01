"""Crossed composition/local-pattern/profile control sets on the frozen Readout cohort.

The Readout diagnostic asks whether a frozen representation adds predictive
information beyond one control set (profile scalar, likelihood scalar and 444
sequence descriptors). This module builds the nested control sets that the
information-hierarchy design pre-registers instead -- composition C, C plus
declared short-range patterns L, C plus mutation-local profile statistics P, and
C+L+P -- and reports the paired likelihood and representation increments over
each of them separately.

Every estimation decision is imported from the admitted Readout recipe:
:func:`~.readout_analysis.sequence_features` supplies C unchanged,
:func:`~.readout_analysis.nested_predict` supplies the nested held-group folds,
equal group/assay/variant weights, training-only weighted feature scaling and
the inner rank-mean-squared-error ridge selection, and
:func:`~.profile_increment.summarize` supplies the group-bootstrap intervals.
Nothing here refits, reorders or reweights those choices.

Two blocks are compressed, because a linear readout of a 8,000-coordinate
tripeptide difference is not affordable at the admitted fold and bootstrap
budget. The compression is the device the Readout protocol already declares for
the representation blocks: a fixed seeded Gaussian projection that depends only
on the target width and the recorded seed, never on labels or observed vectors.
A null increment under a projected block bounds that projected readout class and
does not establish that the uncompressed block carries no information.

The tokenisation block T is not a level of biological organisation. It is the
interface descriptor set the design requires so that a representation increment
cannot be attributed to segmentation without being seen: per-state token counts,
the token-count cycle that shifts every downstream position, and the number of
tokens whose segmentation changes between the two states.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib

import numpy as np
from scipy.stats import rankdata

from ..core.amino_acids import AA20
from ..models.fitness import parse_mutant
from .profile_increment import correlation, standardized_rank, summarize
from ..readouts.readout_analysis import nested_predict

#: Declared ordered blocks. ``S`` is the admitted 444 Readout sequence
#: descriptors, reused as the composition rung without modification.
BLOCK_ORDER = ("S", "L", "P", "T")

#: Nested control sets. ``C`` is contained in ``C+L`` and in ``C+P``; both are
#: contained in ``C+L+P``; ``C+L+P+T`` adds the tokenisation interface block.
#: ``C+L`` and ``C+P`` are not nested in one another, which is why the design is
#: crossed rather than a single ladder.
CONTROL_SETS: dict[str, tuple[str, ...]] = {
    "C": ("S",),
    "C_L": ("S", "L"),
    "C_P": ("S", "P"),
    "C_L_P": ("S", "L", "P"),
    "C_L_P_T": ("S", "L", "P", "T"),
}

#: Additions evaluated separately over every control set.
ADDITIONS: dict[str, tuple[str, ...]] = {"": (), "+M": ("M",), "+R": ("R",), "+M+R": ("M", "R")}

#: Fixed Gaussian projection seed for the 8,000-coordinate tripeptide difference
#: block. The Readout recipe consumes 20260923 through 20260926 for its four
#: representation blocks; this is the next seed in that declared order and is
#: fixed before any outcome was inspected.
LOCAL_PROJECTION_SEED = 20260927
LOCAL_PROJECTION_DIM = 256

#: Declared order of the 14 mutation-local profile coordinates.
PROFILE_FEATURE_ORDER = (
    "profile_score_within_assay_rank",
    "mean_wildtype_column_frequency",
    "mean_mutant_column_frequency",
    "minimum_mutant_column_frequency",
    "maximum_wildtype_column_frequency",
    "mean_mutated_column_entropy_nats",
    "minimum_mutated_column_entropy_nats",
    "maximum_mutated_column_entropy_nats",
    "mean_mutated_column_supported",
    "mean_per_substitution_log_odds",
    "wildtype_supported_column_fraction",
    "wildtype_log10_neff",
    "wildtype_max_identity_over_query_fraction",
    "wildtype_mean_supported_column_entropy_nats",
)

#: Declared order of the 8 tokenisation interface coordinates.
TOKENISATION_FEATURE_ORDER = (
    "wildtype_pooled_tokens_over_budget",
    "mutant_pooled_tokens_over_budget",
    "pooled_token_count_cycle",
    "common_prefix_token_fraction",
    "common_suffix_token_fraction",
    "segmentation_changed_tokens",
    "wildtype_tokens_per_residue",
    "mutant_tokens_per_residue",
)

_CODE = {residue: index for index, residue in enumerate(AA20)}


def _codes(sequence: str) -> np.ndarray:
    try:
        return np.fromiter((_CODE[residue] for residue in sequence), dtype=np.int64, count=len(sequence))
    except KeyError as error:
        raise ValueError(f"noncanonical residue {error.args[0]!r} in a cohort sequence") from error


def substitution_table(wildtype: str, mutant: str, mutant_sequence: str) -> list[tuple[int, int, int]]:
    """``(position0, wild code, mutant code)`` validated against both states.

    The mutation string grammar comes from :func:`~.fitness.parse_mutant`, the
    single declaration in this package. The retained mutant sequence is then
    required to be exactly the wild type with those substitutions applied, so a
    cohort whose sequence and mutation string disagree fails here rather than
    producing silently mismatched local-pattern features.
    """
    length = len(wildtype)
    rows: list[tuple[int, int, int]] = []
    rebuilt = list(wildtype)
    seen: set[int] = set()
    for wild, position, mutated in parse_mutant(mutant):
        index = position - 1
        if wild not in _CODE or mutated not in _CODE:
            raise ValueError(f"substitution {wild}{position}{mutated} leaves the canonical alphabet")
        if not 0 <= index < length or wildtype[index] != wild:
            raise ValueError(f"substitution {wild}{position}{mutated} disagrees with the wild type")
        if index in seen:
            raise ValueError(f"substitution {wild}{position}{mutated} repeats a position")
        seen.add(index)
        rows.append((index, _CODE[wild], _CODE[mutated]))
        rebuilt[index] = mutated
    if not rows:
        raise ValueError("a variant must carry at least one substitution")
    if "".join(rebuilt) != mutant_sequence:
        raise ValueError("retained mutant sequence does not match its mutation string")
    return rows


def _kmer_difference(wild_codes, mutant_codes, positions, k: int) -> dict[int, float]:
    """Exact mutant-minus-wild-type k-mer count difference, as a sparse map.

    Only windows containing a mutated position can differ, so the difference is
    accumulated over the union of those windows. Taking the union rather than
    one window set per substitution is what keeps overlapping substitutions
    exact: two mutations three residues apart share tripeptide windows, and
    per-substitution accounting would count those windows twice.
    """
    length = len(wild_codes)
    starts: set[int] = set()
    for position in positions:
        for start in range(max(0, position - k + 1), min(position, length - k) + 1):
            starts.add(start)
    difference: dict[int, float] = {}
    for start in sorted(starts):
        wild_index = 0
        mutant_index = 0
        for offset in range(k):
            wild_index = wild_index * len(AA20) + int(wild_codes[start + offset])
            mutant_index = mutant_index * len(AA20) + int(mutant_codes[start + offset])
        if wild_index == mutant_index:
            continue
        difference[wild_index] = difference.get(wild_index, 0.0) - 1.0
        difference[mutant_index] = difference.get(mutant_index, 0.0) + 1.0
    return difference


def local_projection(dim: int = LOCAL_PROJECTION_DIM, seed: int = LOCAL_PROJECTION_SEED) -> np.ndarray:
    """Fixed Gaussian map from 8,000 tripeptide coordinates to ``dim``."""
    if dim < 1:
        raise ValueError("local projection width must be positive")
    size = (len(AA20) ** 3, dim)
    return np.random.default_rng(seed).normal(0.0, 1.0 / np.sqrt(dim), size=size).astype(np.float64)


def local_pattern_features(wildtype: str, mutants: Sequence[str], sequences: Sequence[str],
                           projection: np.ndarray) -> np.ndarray:
    """400 exact dipeptide differences plus a projected tripeptide difference.

    Both blocks are mutant-minus-wild-type counts over the complete sequence, so
    they express declared short-range arrangement rather than composition: a
    dipeptide difference is nonzero only where the mutation changes an adjacent
    residue pair, and the composition block ``S`` already carries the residue
    frequencies those pairs are built from.
    """
    if len(mutants) != len(sequences):
        raise ValueError("unaligned mutation strings and mutant sequences")
    if projection.shape[0] != len(AA20) ** 3:
        raise ValueError("local projection must consume 8,000 tripeptide coordinates")
    wild_codes = _codes(wildtype)
    rows = np.zeros((len(mutants), len(AA20) ** 2 + projection.shape[1]), dtype=np.float64)
    for index, (mutant, sequence) in enumerate(zip(mutants, sequences)):
        if len(sequence) != len(wildtype):
            raise ValueError("substitution variants must preserve wild-type length")
        table = substitution_table(wildtype, mutant, sequence)
        mutant_codes = _codes(sequence)
        positions = [row[0] for row in table]
        for kmer, value in _kmer_difference(wild_codes, mutant_codes, positions, 2).items():
            rows[index, kmer] += value
        projected = rows[index, len(AA20) ** 2:]
        for kmer, value in _kmer_difference(wild_codes, mutant_codes, positions, 3).items():
            projected += value * projection[kmer]
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite local-pattern features")
    return rows


def _column_entropy(frequencies: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats per profile column; an unsupported column is 0."""
    safe = np.where(frequencies > 0.0, frequencies, 1.0)
    return -(frequencies * np.log(safe)).sum(axis=1)


def profile_features(wildtype: str, mutants: Sequence[str], sequences: Sequence[str],
                     profile_scores: Sequence[float], frequencies: np.ndarray, background: np.ndarray,
                     record: dict, *, alpha: float) -> np.ndarray:
    """Mutation-local profile frequencies, column entropy and coverage.

    ``frequencies`` is the retained ``(length, 20)`` column-frequency array of
    this wild type's own corpus alignment, each row normalised over its own
    column weight, so a column supported by three effective sequences and one
    supported by three hundred are on the same scale. The support that
    normalisation divides out is therefore not recoverable from the array; what
    is recoverable is whether a column is supported at all, which is what the
    coverage coordinates report, together with the wild-type-level alignment
    depth and identity the profile record retains.
    """
    length = len(wildtype)
    if frequencies.shape != (length, len(AA20)):
        raise ValueError("profile frequencies are not (wild-type length, 20)")
    if background.shape != (len(AA20),) or not np.all(background > 0.0):
        raise ValueError("profile background must be 20 positive frequencies")
    if alpha <= 0:
        raise ValueError("the pseudocount weight must be positive")
    if len(mutants) != len(sequences) or len(mutants) != len(profile_scores):
        raise ValueError("unaligned profile inputs")
    entropy = _column_entropy(frequencies)
    supported = frequencies.sum(axis=1) > 0.0
    constant = np.array([
        float(supported.mean()),
        float(record["log10_neff"]),
        float(record["max_identity_over_query"]) / 100.0,
        float(entropy[supported].mean()) if supported.any() else 0.0,
    ])
    ranks = standardized_rank(np.asarray(profile_scores, dtype=np.float64))
    rows = np.empty((len(mutants), len(PROFILE_FEATURE_ORDER)), dtype=np.float64)
    for index, (mutant, sequence) in enumerate(zip(mutants, sequences)):
        table = substitution_table(wildtype, mutant, sequence)
        positions = np.array([row[0] for row in table])
        wild = np.array([row[1] for row in table])
        mutated = np.array([row[2] for row in table])
        wild_frequency = frequencies[positions, wild]
        mutant_frequency = frequencies[positions, mutated]
        local_entropy = entropy[positions]
        odds = np.log(mutant_frequency + alpha * background[mutated]) - np.log(
            wild_frequency + alpha * background[wild])
        rows[index] = np.r_[
            ranks[index],
            wild_frequency.mean(), mutant_frequency.mean(), mutant_frequency.min(), wild_frequency.max(),
            local_entropy.mean(), local_entropy.min(), local_entropy.max(),
            supported[positions].mean(), odds.mean(), constant]
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite profile features")
    return rows


def tokenisation_features(wildtype_tokens: Sequence[int], mutant_tokens: Sequence[Sequence[int]],
                          wildtype: str, sequences: Sequence[str], *, budget: int) -> np.ndarray:
    """Per-state pooled token counts, the token-count cycle and segmentation change.

    ``wildtype_tokens`` and each entry of ``mutant_tokens`` are the pooled
    residue-bearing token id spans the production extraction used, so the counted
    tokens are exactly the mean-pooling denominators. A residue tokenizer has no
    segmentation degree of freedom: both states carry one token per residue, the
    cycle is zero and the changed-token count is twice the number of
    substitutions, so on that stratum this block only restates the mutation
    positions the composition block already carries.
    """
    if len(mutant_tokens) != len(sequences):
        raise ValueError("unaligned mutant token spans and sequences")
    if budget < 1:
        raise ValueError("token budget must be positive")
    wild = list(wildtype_tokens)
    if not wild:
        raise ValueError("wild-type pooled token span is empty")
    rows = np.empty((len(sequences), len(TOKENISATION_FEATURE_ORDER)), dtype=np.float64)
    for index, (tokens, sequence) in enumerate(zip(mutant_tokens, sequences)):
        mutant = list(tokens)
        if not mutant:
            raise ValueError("mutant pooled token span is empty")
        limit = min(len(wild), len(mutant))
        prefix = 0
        while prefix < limit and wild[prefix] == mutant[prefix]:
            prefix += 1
        suffix = 0
        while suffix < limit - prefix and wild[-1 - suffix] == mutant[-1 - suffix]:
            suffix += 1
        rows[index] = (
            len(wild) / budget,
            len(mutant) / budget,
            len(mutant) - len(wild),
            prefix / len(wild),
            suffix / len(wild),
            len(wild) + len(mutant) - 2 * (prefix + suffix),
            len(wild) / len(wildtype),
            len(mutant) / len(sequence),
        )
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite tokenisation features")
    return rows


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def assemble_designs(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every control set with every addition, in the fixed block order.

    Column order follows :data:`BLOCK_ORDER` then the additions, so a control
    set's columns are a subset of every control set that extends it. Ridge is
    invariant to column order, which is why containment rather than a common
    prefix is the nesting condition.
    """
    missing = [name for name in (*BLOCK_ORDER, "M", "R") if name not in blocks]
    if missing:
        raise ValueError(f"missing feature blocks {missing}")
    return {control + addition: np.column_stack([blocks[name] for name in columns + ADDITIONS[addition]])
            for control, columns in CONTROL_SETS.items() for addition in ADDITIONS}


def design_names() -> list[str]:
    return [control + addition for control in CONTROL_SETS for addition in ADDITIONS]


def fold_membership(records: list[dict]) -> list[tuple]:
    """Outer and inner group membership only, without any tuning quantity.

    Inner rank-MSE losses and selected penalties differ between designs by
    construction, so fold identity has to be compared on membership alone.
    """
    return [(outer["fold"], tuple(outer["held_families"]), tuple(outer["training_families"]),
             tuple(tuple(inner["validation_families"]) for inner in outer["inner_folds"]))
            for outer in records]


def evaluate_crossed_controls(assays: list[dict], *, device: str = "cpu", seed: int = 20260923,
                              fold_seed: int = 20260923, bootstrap: int = 2000,
                              progress=None) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit every nested control set with and without M and R on one shared panel.

    Rows require ``assay``, ``cluster``, ``mutants``, ``measured``, ``P``, ``M``,
    ``S``, ``L``, ``P_block``, ``T`` and ``R``. Every design is fitted on the
    same rows in the same order with the same fold seed, so support, label budget
    and folds are identical across every compared fit by construction; the
    returned record states the realised fold map once and refuses a run in which
    any design's folds differ from it.
    """
    if len({r["assay"] for r in assays}) != len(assays):
        raise ValueError("duplicate assays")
    keys = ("measured", "P", "M", "S", "L", "P_block", "T", "R")
    for row in assays:
        n = len(row["mutants"])
        if n < 3 or len(set(row["mutants"])) != n:
            raise ValueError("too few or duplicate mutants")
        for key in keys:
            if len(row[key]) != n or not np.isfinite(row[key]).all():
                raise ValueError(f"unaligned or nonfinite {key}")
    y = np.concatenate([r["measured"] for r in assays])
    aid = np.concatenate([[r["assay"]] * len(r["mutants"]) for r in assays])
    family = np.concatenate([[r["cluster"]] * len(r["mutants"]) for r in assays])
    m = np.concatenate([standardized_rank(r["M"]) for r in assays])
    p = np.concatenate([standardized_rank(r["P"]) for r in assays])
    blocks = {name: np.concatenate([np.atleast_2d(np.asarray(r[key], dtype=np.float64)) for r in assays])
              for name, key in (("S", "S"), ("L", "L"), ("P", "P_block"), ("T", "T"), ("R", "R"))}
    blocks["M"] = m[:, None]
    designs = assemble_designs(blocks)
    predictions: dict[str, np.ndarray] = {}
    membership = None
    folds = None
    alphas: dict[str, list[float]] = {}
    for name, x in designs.items():
        if progress:
            progress(name)
        predictions[name], record = nested_predict(x, y, aid, family, seed=fold_seed, device=device)
        realised = fold_membership(record)
        if membership is None:
            membership, folds = realised, record
        elif realised != membership:
            raise ValueError(f"{name}: fold assignment differs from the shared fold map")
        alphas[name] = [outer["alpha"] for outer in record]
    rows = []
    for assay in assays:
        index = np.flatnonzero(aid == assay["assay"])
        target = standardized_rank(y[index])
        row = dict(assay=assay["assay"], cluster=assay["cluster"], n_variants=len(index))
        row["raw_M_spearman"] = correlation(rankdata(m[index]), target)
        row["raw_P_spearman"] = correlation(rankdata(p[index]), target)
        for name, prediction in predictions.items():
            row[name + "_spearman"] = correlation(rankdata(prediction[index]), target)
            row[name + "_rank_mse"] = float(np.mean((target - prediction[index]) ** 2))
        for control in CONTROL_SETS:
            for label, left, right in (("M", control + "+M", control),
                                       ("R", control + "+R", control),
                                       ("R_after_M", control + "+M+R", control + "+M")):
                high, low = row[left + "_spearman"], row[right + "_spearman"]
                row[f"increment_{label}_{control}"] = None if high is None or low is None else high - low
                row[f"rank_mse_reduction_{label}_{control}"] = row[right + "_rank_mse"] - row[left + "_rank_mse"]
        rows.append(row)
    metrics = [key for key in rows[0] if key not in ("assay", "cluster", "n_variants")]
    report = dict(
        n_assays=len(rows), n_families=len(set(family)), n_variants=len(y),
        fold_seed=fold_seed, bootstrap_seed=seed,
        control_sets={name: list(columns) for name, columns in CONTROL_SETS.items()},
        feature_dimensions={name: int(x.shape[1]) for name, x in designs.items()},
        block_dimensions={name: int(x.shape[1]) for name, x in blocks.items()},
        prediction_digests={name: _digest(values) for name, values in predictions.items()},
        folds=folds, folds_identical_across_designs=True, selected_alphas=alphas,
        summaries={key: summarize(rows, key, bootstrap=bootstrap, seed=seed) for key in metrics},
        assays=rows,
        protocol="admitted Readout recipe unchanged: 5 outer / 4 inner wild-type-cluster folds, "
                 "equal cluster and within-cluster assay weights, training-only weighted feature "
                 "scaling, within-assay standardized-rank targets, weighted rank-MSE ridge selection, "
                 "unpenalized intercept",
        primary_contrast="paired within-assay Spearman increment from adding M, and separately from "
                         "adding R, over each nested control set; R_after_M adds R over control+M",
        uncertainty="95% wild-type-cluster bootstrap conditional on the fitted cross-validation "
                    "predictions; training and tuning variation are not refitted and the intervals "
                    "are not adjusted across control sets, arms, endpoints or split seeds",
        limitation="A surviving increment is bounded by these declared controls, this experimental-label "
                   "budget and this compressed linear readout class. It does not locate a mechanism and "
                   "does not establish an irreducible biological rule. The tripeptide and representation "
                   "blocks are fixed random projections, so a null increment bounds the projected class "
                   "only.")
    return report, predictions
