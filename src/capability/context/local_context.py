"""Mutation-centred local-context controls, and the gate that admits one.

Why this module exists
======================

The crossed-controls study declared a local-pattern block ``L`` -- 400 exact
dipeptide count differences plus a 256-coordinate fixed Gaussian projection of
the 8,000 tripeptide count differences over the complete sequence -- and that
block failed as a control. It lowered the held-cluster Spearman of the control it
augmented by 0.04747 from C and by 0.01684 from C+P. A control that lowers its
own baseline licenses no conclusion, so the local-motif rung carries no verdict:
that is different from a representation having exceeded it.

The blocks here are built on the opposite design choice. ``L`` counted specific
k-mer identities over the whole sequence, which is high-dimensional, sparse and
family-specific: which particular dipeptide a substitution destroys is close to
unique to one wild type, so a coefficient fitted on training clusters has almost
nothing to transfer to a held-out cluster. These blocks instead describe the
*neighbourhood* of the mutated position in a coarsened, dense encoding at
declared radii: chemical-class composition, physicochemical window means, and
substitution-matrix compatibility between the substituted residue and its
neighbours. Every coordinate is an average over residues, so a coefficient
fitted on one family is a statement about local chemistry rather than about one
family's k-mer inventory.

The limited-receptive-field comparator
======================================

:func:`receptive_field_features` is the cleanest operationalisation of "local
information": a predictor that sees the bounded window around each mutated
position and nothing else. It receives no absolute position, no sequence length,
no global composition and no alignment quantity, and the test suite enforces
that by construction -- changing a residue outside every window, or moving the
same window content to a different position in a longer sequence, must leave its
rows unchanged.

Its pooling over a variant's substitutions is a sum rather than a mean, because
sum pooling is what a bounded-receptive-field encoder does and because the
substitution count is then recoverable inside the block (the wild-type half's
total mass is the number of in-sequence window positions) rather than having to
be supplied from outside the window.

The qualification gate
======================

No candidate may be used to measure anything before it has been shown to
generalise across held-out wild-type clusters itself. :func:`qualification_gate`
is that rule, applied to control-versus-control correlations only: a candidate
qualifies when adding it does not lower the held-cluster correlation of either
baseline it augments, at every declared split seed. Nothing in the gate reads a
model quantity, so a candidate cannot be tuned against the increment it is later
meant to absorb, which would invert the logic of a control.

Estimation discipline
=====================

Every estimation decision is imported unchanged from the admitted Readout
recipe, through the same call sites the crossed-controls study uses:
:func:`~.readout_analysis.sequence_features` supplies the composition control,
:func:`~.crossed_controls.profile_features` the mutation-local profile block,
:func:`~.readout_analysis.nested_predict` the nested held-cluster folds, equal
cluster and within-cluster assay weights, training-only weighted feature scaling
and the inner rank-mean-squared-error ridge selection, and
:func:`~.profile_increment.summarize` the cluster-bootstrap intervals. Nothing
here refits, reorders or reweights those choices, which is what makes these
numbers compose with the crossed-controls tables directly.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json

import numpy as np
from scipy.stats import rankdata

from ..core.amino_acids import AA20, BLOSUM62_ORDER, BLOSUM62_ROWS
from .crossed_controls import substitution_table
from ..core.protein_properties import AA_CLASSES, CLASS_NAMES, CLASS_OF_RESIDUE
from .profile_increment import correlation, standardized_rank, summarize
from ..readouts.readout_analysis import nested_predict

#: Declared context radii, in residues. A window at radius ``r`` around a
#: mutated position spans ``[i - r, i + r]`` clipped to the sequence; the
#: descriptor blocks read that window with the mutated position itself removed,
#: so what they express is the context of the substitution rather than the
#: substitution, which the composition control already carries.
RADII: tuple[int, ...] = (1, 2, 4, 8, 16)

#: Chou-Fasman conformational parameters, transcribed rather than fitted:
#: P(alpha), the helix propensity. Source: Chou P.Y. and Fasman G.D. (1978),
#: 'Empirical predictions of protein conformation', Adv. Enzymol. 47:45-148.
CHOU_FASMAN_HELIX: dict[str, float] = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13,
    "G": 0.57, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
    "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11, "R": 0.98,
    "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}

#: Chou-Fasman P(beta), the sheet propensity, from the same source.
CHOU_FASMAN_SHEET: dict[str, float] = {
    "A": 0.83, "C": 1.19, "D": 0.54, "E": 0.37, "F": 1.38,
    "G": 0.75, "H": 0.87, "I": 1.60, "K": 0.74, "L": 1.30,
    "M": 1.05, "N": 0.89, "P": 0.55, "Q": 1.10, "R": 0.93,
    "S": 0.75, "T": 1.19, "V": 1.70, "W": 1.37, "Y": 1.47,
}

CHOU_FASMAN_SOURCE = (
    "Chou P.Y. and Fasman G.D. (1978), 'Empirical predictions of protein conformation', "
    "Advances in Enzymology 47:45-148, conformational parameters P(alpha) and P(beta)"
)

#: Declared order of the six physicochemical scales. Hydropathy, charge and
#: volume are the repository's existing declared property basis; Grantham
#: polarity is the fourth axis the chemical-similarity declaration already uses;
#: the two Chou-Fasman propensities are added here because local secondary-
#: structure propensity is the one local-structural axis the other four do not
#: carry, and a local-motif control that omitted it would not express the thing
#: the rung is named after.
SCALE_ORDER: tuple[str, ...] = (
    "hydropathy", "charge", "volume", "polarity", "helix_propensity", "sheet_propensity",
)

#: Radii of the limited-receptive-field comparator. The primary is declared
#: before fitting; the other is a declared radius sensitivity, not a sweep to
#: select from after seeing an outcome.
RECEPTIVE_FIELD_RADII: tuple[int, ...] = (3, 7)
PRIMARY_RECEPTIVE_FIELD_RADIUS = 7

#: Candidate local controls, in declared order. ``wall`` is the union of the
#: three descriptor blocks; ``rf3`` and ``rf7`` are the limited-receptive-field
#: comparator at its two declared radii.
CANDIDATE_BLOCKS: tuple[str, ...] = ("wcomp", "wchem", "wsub", "wall", "rf3", "rf7")

#: Which candidates are limited-receptive-field comparators rather than
#: descriptor blocks. A comparator carries the extra qualification requirement
#: that it predicts held-out clusters above zero on its own.
RECEPTIVE_FIELD_CANDIDATES: tuple[str, ...] = ("rf3", "rf7")

#: The split seeds the gate is evaluated at, unchanged from the admitted Readout
#: recipe and from the crossed-controls study.
SPLIT_SEEDS: tuple[int, ...] = (20260923, 20260924, 20260925)

#: The two baselines every candidate has to leave no worse off.
QUALIFICATION_BASELINES: tuple[str, ...] = ("C", "C_P")

_CODE = {residue: index for index, residue in enumerate(AA20)}
_CLASS_INDEX = np.asarray([CLASS_NAMES.index(CLASS_OF_RESIDUE[residue]) for residue in AA20])


def _blosum62() -> np.ndarray:
    """BLOSUM62 half-bit scores reindexed onto :data:`AA20` order."""
    order = {residue: index for index, residue in enumerate(BLOSUM62_ORDER)}
    return np.asarray(
        [[float(BLOSUM62_ROWS[order[left]][order[right]]) for right in AA20] for left in AA20],
        dtype=np.float64)


def _scale_matrix() -> np.ndarray:
    """The six declared scales over :data:`AA20`, each z-scored over the alphabet.

    Z-scoring is over the twenty residues of the fixed published tables, so it
    depends on no cohort, no corpus and no label. It exists because the raw
    scales differ by two orders of magnitude -- side-chain volume runs 48 to 163
    cubic angstroms while formal charge runs -1 to +1 -- and the interaction
    coordinates below are products of two of them, which under one shared ridge
    penalty would otherwise be six blocks on six incomparable scales.
    """
    from ..core.protein_properties import CHARGE_PH7, SIDE_CHAIN_VOLUME
    from ..core.protein_properties import GRANTHAM_POLARITY
    from .profiles import KYTE_DOOLITTLE

    tables = {
        "hydropathy": KYTE_DOOLITTLE,
        "charge": CHARGE_PH7,
        "volume": SIDE_CHAIN_VOLUME,
        "polarity": GRANTHAM_POLARITY,
        "helix_propensity": CHOU_FASMAN_HELIX,
        "sheet_propensity": CHOU_FASMAN_SHEET,
    }
    if tuple(tables) != SCALE_ORDER:
        raise ValueError("scale tables do not follow the declared order")
    rows = []
    for name in SCALE_ORDER:
        values = np.asarray([float(tables[name][residue]) for residue in AA20], dtype=np.float64)
        spread = values.std()
        if spread <= 0.0:
            raise ValueError(f"scale {name} is constant over the alphabet")
        rows.append((values - values.mean()) / spread)
    return np.asarray(rows, dtype=np.float64)


def _codes(sequence: str) -> np.ndarray:
    try:
        return np.fromiter((_CODE[residue] for residue in sequence), dtype=np.int64,
                           count=len(sequence))
    except KeyError as error:
        raise ValueError(f"noncanonical residue {error.args[0]!r} in a cohort sequence") from error


def _context_indices(position: int, radius: int, length: int) -> np.ndarray:
    """In-sequence window positions at ``radius``, excluding the mutated one."""
    low = max(0, position - radius)
    high = min(length - 1, position + radius)
    indices = np.arange(low, high + 1)
    return indices[indices != position]


def _substitution_rows(wildtype: str, mutants: Sequence[str],
                       sequences: Sequence[str]) -> list[list[tuple[int, int, int]]]:
    """Validated substitution tables for every variant, refusing any mismatch.

    The grammar, the wild-type agreement check, the repeated-position check and
    the requirement that the retained mutant sequence equal the wild type with
    its mutation string applied all come from
    :func:`~.crossed_controls.substitution_table`, which is the single
    declaration of that validation in this package.
    """
    if len(mutants) != len(sequences):
        raise ValueError("unaligned mutation strings and mutant sequences")
    rows = []
    for mutant, sequence in zip(mutants, sequences):
        if len(sequence) != len(wildtype):
            raise ValueError("substitution variants must preserve wild-type length")
        rows.append(substitution_table(wildtype, mutant, sequence))
    return rows


def window_composition_feature_names() -> tuple[str, ...]:
    names = []
    for radius in RADII:
        names.extend(f"class_fraction_{name}_r{radius}" for name in CLASS_NAMES)
        names.append(f"class_entropy_nats_r{radius}")
        names.append(f"in_sequence_context_fraction_r{radius}")
    return tuple(names)


def window_composition_features(wildtype: str, mutants: Sequence[str],
                                sequences: Sequence[str]) -> np.ndarray:
    """Chemical-class composition of the mutation-centred wild-type window.

    Per declared radius: the four fractions of the repository's declared
    four-way chemical partition over the window with the mutated position
    removed, that fraction vector's Shannon entropy in nats, and the fraction of
    the nominal ``2r`` context positions that lie inside the sequence. A
    variant's row is the mean over its substitutions.

    The coarsening is what makes the block transferable: four dense fractions
    per radius are estimated from every window residue, where a dipeptide count
    difference is one sparse event tied to one wild type.
    """
    wild_classes = _CLASS_INDEX[_codes(wildtype)]
    length = len(wildtype)
    tables = _substitution_rows(wildtype, mutants, sequences)
    width = len(RADII) * (len(CLASS_NAMES) + 2)
    rows = np.zeros((len(mutants), width), dtype=np.float64)
    for index, table in enumerate(tables):
        accumulated = np.zeros(width, dtype=np.float64)
        for position, _, _ in table:
            offset = 0
            for radius in RADII:
                indices = _context_indices(position, radius, length)
                if indices.size == 0:
                    raise ValueError("a mutation-centred context window is empty")
                counts = np.bincount(wild_classes[indices], minlength=len(CLASS_NAMES))
                fractions = counts / counts.sum()
                safe = np.where(fractions > 0.0, fractions, 1.0)
                accumulated[offset:offset + len(CLASS_NAMES)] = fractions
                accumulated[offset + len(CLASS_NAMES)] = -(fractions * np.log(safe)).sum()
                accumulated[offset + len(CLASS_NAMES) + 1] = indices.size / (2 * radius)
                offset += len(CLASS_NAMES) + 2
            rows[index] += accumulated
        rows[index] /= len(table)
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite window-composition features")
    return rows


def window_chemistry_feature_names() -> tuple[str, ...]:
    names = [f"delta_{name}" for name in SCALE_ORDER]
    for radius in RADII:
        names.extend(f"context_mean_{name}_r{radius}" for name in SCALE_ORDER)
    for radius in RADII:
        names.extend(f"delta_times_context_mean_{name}_r{radius}" for name in SCALE_ORDER)
    return tuple(names)


def window_chemistry_features(wildtype: str, mutants: Sequence[str],
                              sequences: Sequence[str]) -> np.ndarray:
    """Physicochemical window means and their interaction with the substitution.

    Per variant, averaged over its substitutions: the substituted residue's own
    change on each of the six declared z-scored scales; the mean of each scale
    over the mutation-centred wild-type context window at each declared radius;
    and the product of the two. The product is the coordinate that expresses
    local context rather than either factor alone -- a hydrophobic-to-polar
    substitution inside a hydrophobic neighbourhood is a different event from
    the same substitution in a polar one -- and it is the one a linear readout
    cannot form from the composition control by itself.
    """
    scales = _scale_matrix()
    wild_codes = _codes(wildtype)
    length = len(wildtype)
    tables = _substitution_rows(wildtype, mutants, sequences)
    n_scales = len(SCALE_ORDER)
    width = n_scales * (1 + 2 * len(RADII))
    rows = np.zeros((len(mutants), width), dtype=np.float64)
    for index, table in enumerate(tables):
        accumulated = np.zeros(width, dtype=np.float64)
        for position, wild, mutated in table:
            delta = scales[:, mutated] - scales[:, wild]
            accumulated[:n_scales] += delta
            for order, radius in enumerate(RADII):
                indices = _context_indices(position, radius, length)
                if indices.size == 0:
                    raise ValueError("a mutation-centred context window is empty")
                means = scales[:, wild_codes[indices]].mean(axis=1)
                start = n_scales * (1 + order)
                accumulated[start:start + n_scales] += means
                start = n_scales * (1 + len(RADII) + order)
                accumulated[start:start + n_scales] += delta * means
        rows[index] = accumulated / len(table)
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite window-chemistry features")
    return rows


def window_substitution_feature_names() -> tuple[str, ...]:
    names = []
    for radius in RADII:
        names.append(f"mutant_context_blosum62_mean_r{radius}")
        names.append(f"wildtype_context_blosum62_mean_r{radius}")
        names.append(f"context_blosum62_mean_difference_r{radius}")
    return tuple(names)


def window_substitution_features(wildtype: str, mutants: Sequence[str],
                                 sequences: Sequence[str]) -> np.ndarray:
    """Substitution-matrix compatibility between each residue and its neighbours.

    Per declared radius: the mean BLOSUM62 half-bit score of the mutant residue
    against every residue of the mutation-centred wild-type context window, the
    same for the wild-type residue, and their difference. A variant's row is the
    mean over its substitutions.

    The position-independent BLOSUM62 score of the substitution itself is
    deliberately absent. It is a fixed function of the ordered residue pair, so
    it already lies in the linear span of the composition control's 400 directed
    substitution counts and adding it would widen the block without widening the
    information it carries. What is new here is the interaction with the
    neighbourhood. BLOSUM62 is estimated from aligned families, so this block is
    not free of evolutionary statistics; it is local in the sense of reading a
    bounded window, not in the sense of using sequence arrangement alone.
    """
    matrix = _blosum62()
    wild_codes = _codes(wildtype)
    length = len(wildtype)
    tables = _substitution_rows(wildtype, mutants, sequences)
    width = 3 * len(RADII)
    rows = np.zeros((len(mutants), width), dtype=np.float64)
    for index, table in enumerate(tables):
        accumulated = np.zeros(width, dtype=np.float64)
        for position, wild, mutated in table:
            for order, radius in enumerate(RADII):
                indices = _context_indices(position, radius, length)
                if indices.size == 0:
                    raise ValueError("a mutation-centred context window is empty")
                neighbours = wild_codes[indices]
                mutant_mean = matrix[mutated, neighbours].mean()
                wild_mean = matrix[wild, neighbours].mean()
                accumulated[3 * order] += mutant_mean
                accumulated[3 * order + 1] += wild_mean
                accumulated[3 * order + 2] += mutant_mean - wild_mean
        rows[index] = accumulated / len(table)
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite window-substitution features")
    return rows


def receptive_field_feature_names(radius: int) -> tuple[str, ...]:
    offsets = range(-radius, radius + 1)
    return tuple(
        [f"wildtype_offset{offset:+d}_{residue}" for offset in offsets for residue in AA20]
        + [f"difference_offset{offset:+d}_{residue}" for offset in offsets for residue in AA20])


def receptive_field_features(wildtype: str, mutants: Sequence[str], sequences: Sequence[str],
                             radius: int) -> np.ndarray:
    """The bounded window around each mutated position, and nothing else.

    Per substitution at position ``i``: a positional one-hot over the twenty
    residues of the wild-type sequence at relative offsets ``-radius`` through
    ``+radius``, concatenated with the mutant-minus-wild-type one-hot difference
    at the same offsets. Offsets falling outside the sequence contribute zero
    rows. A variant's row is the **sum** over its substitutions.

    What the predictor is denied is the point. No absolute position, no sequence
    length, no composition outside the windows, no alignment or profile quantity
    and no model output reaches these columns, so a correlation achieved from
    them is a correlation achieved from local sequence context alone. The two
    halves span the same space as the two states' one-hots; this basis is the
    declared one because the difference half is sparse -- for a single
    substitution it carries exactly two nonzero entries, at the centre offset --
    which separates the context the window supplies from the substitution events
    inside it, and because a ridge penalty is not invariant to that choice.

    Sum pooling rather than mean pooling keeps the substitution count inside the
    window description: the wild-type half's total mass is the number of
    in-sequence window positions, so a count the window itself determines does
    not have to be supplied from outside it.
    """
    if radius < 1:
        raise ValueError("receptive-field radius must be positive")
    wild_codes = _codes(wildtype)
    length = len(wildtype)
    span = 2 * radius + 1
    tables = _substitution_rows(wildtype, mutants, sequences)
    rows = np.zeros((len(mutants), 2 * span * len(AA20)), dtype=np.float64)
    half = span * len(AA20)
    for index, (table, sequence) in enumerate(zip(tables, sequences)):
        mutant_codes = _codes(sequence)
        for position, _, _ in table:
            for order, offset in enumerate(range(-radius, radius + 1)):
                site = position + offset
                if not 0 <= site < length:
                    continue
                wild = int(wild_codes[site])
                mutated = int(mutant_codes[site])
                rows[index, order * len(AA20) + wild] += 1.0
                rows[index, half + order * len(AA20) + mutated] += 1.0
                rows[index, half + order * len(AA20) + wild] -= 1.0
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite receptive-field features")
    return rows


def candidate_blocks(wildtype: str, mutants: Sequence[str],
                     sequences: Sequence[str]) -> dict[str, np.ndarray]:
    """Every declared candidate local control for one assay, in declared order."""
    composition = window_composition_features(wildtype, mutants, sequences)
    chemistry = window_chemistry_features(wildtype, mutants, sequences)
    substitution = window_substitution_features(wildtype, mutants, sequences)
    blocks = {
        "wcomp": composition,
        "wchem": chemistry,
        "wsub": substitution,
        "wall": np.column_stack([composition, chemistry, substitution]),
    }
    for radius in RECEPTIVE_FIELD_RADII:
        blocks[f"rf{radius}"] = receptive_field_features(wildtype, mutants, sequences, radius)
    if tuple(blocks) != CANDIDATE_BLOCKS:
        raise ValueError("assembled candidate blocks do not follow the declared order")
    return blocks


def candidate_feature_names() -> dict[str, tuple[str, ...]]:
    composition = window_composition_feature_names()
    chemistry = window_chemistry_feature_names()
    substitution = window_substitution_feature_names()
    names = {"wcomp": composition, "wchem": chemistry, "wsub": substitution,
             "wall": composition + chemistry + substitution}
    for radius in RECEPTIVE_FIELD_RADII:
        names[f"rf{radius}"] = receptive_field_feature_names(radius)
    return names


def declaration() -> dict:
    """The full declared block set, fixed before any fit is dispatched."""
    names = candidate_feature_names()
    return dict(
        schema_version="d1_local_context_declaration_v1",
        radii=list(RADII),
        chemical_partition={name: AA_CLASSES[name] for name in CLASS_NAMES},
        scale_order=list(SCALE_ORDER),
        chou_fasman_source=CHOU_FASMAN_SOURCE,
        chou_fasman_helix=CHOU_FASMAN_HELIX,
        chou_fasman_sheet=CHOU_FASMAN_SHEET,
        scale_standardisation="each scale z-scored over the twenty residues of its fixed "
                              "published table; no cohort, corpus or label enters",
        substitution_matrix="BLOSUM62 half-bit scores reindexed onto the alphabetical "
                            "twenty-residue order; the position-independent substitution score "
                            "is excluded as it lies in the span of the composition control",
        descriptor_window="mutation-centred wild-type window with the mutated position removed; "
                          "a variant's row is the mean over its substitutions",
        receptive_field_radii=list(RECEPTIVE_FIELD_RADII),
        primary_receptive_field_radius=PRIMARY_RECEPTIVE_FIELD_RADIUS,
        receptive_field_encoding="wild-type positional one-hot over the window concatenated with "
                                 "the mutant-minus-wild-type positional one-hot difference; "
                                 "out-of-sequence offsets contribute zero; summed over the "
                                 "variant's substitutions",
        candidates=list(CANDIDATE_BLOCKS),
        receptive_field_candidates=list(RECEPTIVE_FIELD_CANDIDATES),
        widths={name: len(columns) for name, columns in names.items()},
        feature_names={name: list(columns) for name, columns in names.items()},
        split_seeds=list(SPLIT_SEEDS),
        qualification_baselines=list(QUALIFICATION_BASELINES),
        qualification_rule="a candidate qualifies only if, at every declared split seed, adding "
                           "it does not lower the held-cluster Spearman of C or of C+P; a "
                           "receptive-field comparator must additionally predict held-out "
                           "clusters above zero on its own with a 95% cluster-bootstrap interval "
                           "excluding zero at every split seed",
        selection_rule="among qualified descriptor candidates the primary local control is the "
                       "one with the largest mean increase of the C baseline over the three "
                       "split seeds; a qualified receptive-field comparator is carried "
                       "separately because it is a different kind of object. Both quantities are "
                       "control-versus-control and read no model output",
    )


def declaration_sha256() -> str:
    return hashlib.sha256(
        json.dumps(declaration(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def control_sets(local_blocks: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Nested control sets over one or more declared local blocks.

    ``C`` is the admitted composition control and ``C+P`` adds the
    mutation-local profile block, both unchanged from the crossed-controls
    study, so the two columns are directly comparable with its tables. Each
    local block contributes the block alone, the block over composition, and the
    block over composition and profile.
    """
    if not local_blocks or len(set(local_blocks)) != len(local_blocks):
        raise ValueError("declare at least one distinct local block")
    unknown = [name for name in local_blocks if name not in CANDIDATE_BLOCKS]
    if unknown:
        raise ValueError(f"undeclared local blocks {unknown}")
    sets: dict[str, tuple[str, ...]] = {"C": ("S",), "C_P": ("S", "P")}
    for name in local_blocks:
        sets[name] = (name,)
        sets[f"C_{name}"] = ("S", name)
        sets[f"C_P_{name}"] = ("S", "P", name)
    return sets


#: Additions evaluated separately over every control set of a measurement cell.
ADDITIONS: dict[str, tuple[str, ...]] = {
    "": (), "+M": ("M",), "+R": ("R",), "+M+R": ("M", "R")}


def assemble_designs(blocks: dict[str, np.ndarray], sets: dict[str, tuple[str, ...]],
                     additions: dict[str, tuple[str, ...]]) -> dict[str, np.ndarray]:
    """Every control set with every addition, in one fixed column order.

    Columns follow the control set's declared block order and then the addition,
    so a control set's columns are a subset of every design that extends it.
    Ridge is invariant to column order, which is why containment rather than a
    shared prefix is the nesting condition the tests check.
    """
    required = sorted({name for columns in sets.values() for name in columns}
                      | {name for columns in additions.values() for name in columns})
    missing = [name for name in required if name not in blocks]
    if missing:
        raise ValueError(f"missing feature blocks {missing}")
    return {name + label: np.column_stack([blocks[block] for block in columns + extra])
            for name, columns in sets.items() for label, extra in additions.items()}


def fold_membership(records: list[dict]) -> list[tuple]:
    """Outer and inner group membership only, without any tuning quantity."""
    return [(outer["fold"], tuple(outer["held_families"]), tuple(outer["training_families"]),
             tuple(tuple(inner["validation_families"]) for inner in outer["inner_folds"]))
            for outer in records]


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def qualification_gate(measured: dict[str, dict[str, dict]], candidates: Sequence[str] = CANDIDATE_BLOCKS
                       ) -> dict:
    """Apply the declared gate to measured control-versus-control correlations.

    ``measured`` maps ``str(split seed)`` to that seed's summaries, each holding
    ``qualification_delta_over_<baseline>_<candidate>`` and, for a comparator,
    ``standalone_<candidate>``. Every entry is a ``summarize`` record with a
    ``point`` and an ``interval``.

    The gate is a refusal, not a score. A candidate that lowers either baseline's
    held-cluster correlation at any one of the declared split seeds is refused,
    however large its point estimate is elsewhere, because a control that makes
    its own baseline worse cannot bound anything. Nothing in this function reads
    a model quantity.
    """
    seeds = [str(seed) for seed in SPLIT_SEEDS]
    absent = [seed for seed in seeds if seed not in measured]
    if absent:
        raise ValueError(f"the gate requires every declared split seed; missing {absent}")
    verdicts = {}
    for candidate in candidates:
        if candidate not in CANDIDATE_BLOCKS:
            raise ValueError(f"undeclared candidate {candidate}")
        failures, per_seed = [], {}
        for seed in seeds:
            record = {}
            for baseline in QUALIFICATION_BASELINES:
                key = f"qualification_delta_over_{baseline}_{candidate}"
                if key not in measured[seed]:
                    raise ValueError(f"{seed}: {key} was not measured")
                entry = measured[seed][key]
                point = entry["point"]
                if point is None:
                    raise ValueError(f"{seed}: {key} has no point estimate")
                record[baseline] = dict(point=float(point), interval=entry["interval"])
                if float(point) < 0.0:
                    failures.append(f"lowers {baseline} by {-float(point):.5f} at split seed {seed}")
            if candidate in RECEPTIVE_FIELD_CANDIDATES:
                key = f"standalone_{candidate}"
                if key not in measured[seed]:
                    raise ValueError(f"{seed}: {key} was not measured")
                entry = measured[seed][key]
                record["standalone"] = dict(point=entry["point"], interval=entry["interval"])
                interval = entry["interval"]
                if entry["point"] is None or interval is None or float(interval[0]) <= 0.0:
                    failures.append(
                        f"held-cluster correlation on its own is not resolved above zero at "
                        f"split seed {seed}")
            per_seed[seed] = record
        points = [per_seed[seed]["C"]["point"] for seed in seeds]
        verdicts[candidate] = dict(
            qualified=not failures, refusals=failures, per_seed=per_seed,
            mean_increase_over_C=float(np.mean(points)),
            minimum_increase_over_C=float(np.min(points)),
            kind="receptive_field" if candidate in RECEPTIVE_FIELD_CANDIDATES else "descriptor")
    descriptors = [name for name, entry in verdicts.items()
                   if entry["qualified"] and entry["kind"] == "descriptor"]
    comparators = [name for name, entry in verdicts.items()
                   if entry["qualified"] and entry["kind"] == "receptive_field"]
    primary = max(descriptors, key=lambda name: verdicts[name]["mean_increase_over_C"]) \
        if descriptors else None
    comparator = max(comparators, key=lambda name: verdicts[name]["mean_increase_over_C"]) \
        if comparators else None
    return dict(
        schema_version="d1_local_context_gate_v1",
        split_seeds=list(SPLIT_SEEDS), candidates=list(candidates), verdicts=verdicts,
        qualified=[name for name, entry in verdicts.items() if entry["qualified"]],
        discarded=[name for name, entry in verdicts.items() if not entry["qualified"]],
        primary_local_control=primary, receptive_field_control=comparator,
        carried_forward=[name for name in (primary, comparator) if name is not None],
        rule=declaration()["qualification_rule"],
        selection=declaration()["selection_rule"],
        limitation="The gate admits a control that generalises across held-out wild-type "
                   "clusters under this readout class at this label budget. It does not "
                   "establish that the control exhausts local sequence information, and a "
                   "refusal is a statement about this declared block under this readout class "
                   "rather than about local information as such.")


def evaluate_local_context(assays: list[dict], *, local_blocks: Sequence[str],
                           additions: dict[str, tuple[str, ...]] | None = None,
                           device: str = "cpu", seed: int = 20260923, fold_seed: int = 20260923,
                           bootstrap: int = 2000, progress=None
                           ) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit every control set with every addition on one shared panel.

    Rows require ``assay``, ``cluster``, ``mutants``, ``measured``, ``P``, ``M``,
    ``S``, ``P_block``, ``R`` and one array per named local block. Every design
    is fitted on the same rows in the same order with the same fold seed, so
    support, label budget and folds are identical across every compared fit by
    construction; the returned record states the realised fold map once and
    refuses a run in which any design's folds differ from it.

    ``additions`` empty of everything but the no-addition entry makes this a
    qualification cell: no model quantity then reaches any fitted design, which
    is what lets the gate be applied before any increment is measured.
    """
    additions = dict(ADDITIONS) if additions is None else dict(additions)
    if "" not in additions or additions[""]:
        raise ValueError("the addition set must contain the empty addition")
    sets = control_sets(local_blocks)
    if len({r["assay"] for r in assays}) != len(assays):
        raise ValueError("duplicate assays")
    model_keys = sorted({name for columns in additions.values() for name in columns})
    required = ("measured", "P", "M", "S", "P_block", "R", *local_blocks)
    for row in assays:
        count = len(row["mutants"])
        if count < 3 or len(set(row["mutants"])) != count:
            raise ValueError("too few or duplicate mutants")
        for key in required:
            if len(row[key]) != count or not np.isfinite(row[key]).all():
                raise ValueError(f"unaligned or nonfinite {key}")
    y = np.concatenate([r["measured"] for r in assays])
    aid = np.concatenate([[r["assay"]] * len(r["mutants"]) for r in assays])
    family = np.concatenate([[r["cluster"]] * len(r["mutants"]) for r in assays])
    m = np.concatenate([standardized_rank(r["M"]) for r in assays])
    p = np.concatenate([standardized_rank(r["P"]) for r in assays])
    blocks = {name: np.concatenate([np.atleast_2d(np.asarray(r[key], dtype=np.float64))
                                    for r in assays])
              for name, key in [("S", "S"), ("P", "P_block"), ("R", "R")]
              + [(name, name) for name in local_blocks]}
    blocks["M"] = m[:, None]
    designs = assemble_designs(blocks, sets, additions)
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
        for candidate in local_blocks:
            row[f"standalone_{candidate}"] = row[f"{candidate}_spearman"]
            for baseline in QUALIFICATION_BASELINES:
                high = row[f"{baseline}_{candidate}_spearman"]
                low = row[f"{baseline}_spearman"]
                row[f"qualification_delta_over_{baseline}_{candidate}"] = (
                    None if high is None or low is None else high - low)
        for name in sets:
            for label, addition in (("M", "+M"), ("R", "+R")):
                if addition not in additions:
                    continue
                high, low = row[name + addition + "_spearman"], row[name + "_spearman"]
                row[f"increment_{label}_{name}"] = (
                    None if high is None or low is None else high - low)
                row[f"rank_mse_reduction_{label}_{name}"] = (
                    row[name + "_rank_mse"] - row[name + addition + "_rank_mse"])
            if "+M+R" in additions and "+M" in additions:
                high, low = row[name + "+M+R_spearman"], row[name + "+M_spearman"]
                row[f"increment_R_after_M_{name}"] = (
                    None if high is None or low is None else high - low)
                row[f"rank_mse_reduction_R_after_M_{name}"] = (
                    row[name + "+M_rank_mse"] - row[name + "+M+R_rank_mse"])
        rows.append(row)
    metrics = [key for key in rows[0] if key not in ("assay", "cluster", "n_variants")]
    report = dict(
        n_assays=len(rows), n_families=len(set(family)), n_variants=len(y),
        fold_seed=fold_seed, bootstrap_seed=seed,
        local_blocks=list(local_blocks), control_sets={k: list(v) for k, v in sets.items()},
        additions={k: list(v) for k, v in additions.items()},
        model_quantities_fitted=model_keys,
        feature_dimensions={name: int(x.shape[1]) for name, x in designs.items()},
        block_dimensions={name: int(x.shape[1]) for name, x in blocks.items()},
        prediction_digests={name: _digest(values) for name, values in predictions.items()},
        folds=folds, folds_identical_across_designs=True, selected_alphas=alphas,
        summaries={key: summarize(rows, key, bootstrap=bootstrap, seed=seed) for key in metrics},
        assays=rows,
        protocol="admitted Readout recipe unchanged: 5 outer / 4 inner wild-type-cluster folds, "
                 "equal cluster and within-cluster assay weights, training-only weighted feature "
                 "scaling, within-assay standardized-rank targets, weighted rank-MSE ridge "
                 "selection, unpenalized intercept",
        primary_contrast="qualification: the paired within-assay Spearman change of C and of C+P "
                         "from adding each candidate local block, and each comparator's own "
                         "held-cluster correlation. Measurement: the paired increment from "
                         "adding M, and separately R, over each control set",
        uncertainty="95% wild-type-cluster bootstrap conditional on the fitted cross-validation "
                    "predictions; training and tuning variation are not refitted and the "
                    "intervals are not adjusted across control sets, candidates, arms, endpoints "
                    "or split seeds",
        limitation="Every number is bounded by this declared control set, this experimental-label "
                   "budget and this compressed linear readout class. A qualified local control "
                   "bounds local information only as this block expresses it; a surviving "
                   "increment locates no mechanism and establishes no irreducible biological "
                   "rule.")
    return report, predictions
