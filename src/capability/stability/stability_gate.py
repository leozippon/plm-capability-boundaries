"""Single-mutant stability gate: endpoint, label-blind support, declared controls.

The gate's question is whether a frozen model quantity places resolvable
information about how a single substitution changes measured folding stability,
beyond controls that are competent held-out predictors in their own right.

The endpoint is the single-mutant stability change

    ddG = y_mut - y_WT

on the combined MegaScale ``dG_ML`` scale in kcal/mol, with the source's sign
convention: ``dG_ML`` is an unfolding free energy, larger is more stable, so
``ddG < 0`` is destabilising. It is a two-estimate difference, not the four-estimate
cycle that ``pairwise_epistasis`` targets, so it carries its own between-protease
agreement, its own noise floor and its own support accounting; nothing in the
cycle endpoint's admission transfers to it.

Three support rules are declared here rather than discovered:

* Insertion and deletion constructs have their ``aa_seq`` truncated to the wild
  type's length in the pinned files, so keying a state on ``aa_seq`` alone lets
  an indel row masquerade as a substitution variant. Every row whose ``mut_type``
  begins ``ins`` or ``del`` is excluded, with its accounting reported.
* Censored ``dG_ML`` strings (``<-1``, ``>5``, ``-``) are never coerced to numeric
  endpoints; they are counted and dropped.
* One background per final family group, chosen by alphabetical ``WT_name``, and
  at most :data:`VARIANT_CAP` single mutants per background drawn by a stable hash
  of the mutant sequence. Neither rule inspects a measurement or a model score.

Weighting, folds, the ridge recipe, the projection and the group bootstrap are
imported from the admitted Readout and pairwise recipes rather than restated. The
independent unit is the family group; inside a group the mutated site carries the
weight and the variants of one site share it, because the substitutions at one
position probe one local environment and every variant of a background shares that
background's single wild-type measurement.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..core.amino_acids import AA20, BLOSUM62_ORDER, BLOSUM62_ROWS
from ..core.protein_properties import CHARGE_PH7, SIDE_CHAIN_VOLUME
from ..interactions.pairwise_epistasis import (
    ALPHAS, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, FEATURE_BLOCKS, INNER_SPLITS,
    OUTER_SPLITS, PROFILE_PSEUDO_FREQUENCY, PROJECTION_DIM, PROJECTION_SEED,
    SPLIT_SEEDS, _select_alpha, group_errors, group_spearman, interval,
    projection_matrices)
from .pairwise_stability import QC_SOURCE, QC_WIDTH_COLUMNS, QC_WIDTH_KCAL_MOL
from ..context.profiles import KYTE_DOOLITTLE
from ..readouts.readout_analysis import family_folds, ridge_predict, row_weights

#: What the target is, in one line, so an artefact cannot be read against the
#: cycle endpoint by mistake.
ENDPOINT = ('ddG = y_mut - y_WT on the combined MegaScale dG_ML scale in kcal/mol; '
            'dG_ML is an unfolding free energy, so positive ddG is stabilising')

#: Rows whose ``mut_type`` begins with one of these prefixes carry a length-truncated
#: ``aa_seq`` and are not substitution variants.
INDEL_MUT_TYPE_PREFIXES = ('ins', 'del')

#: Censored ``dG_ML`` encodings. Counted, never converted to -1 or 5.
CENSORED_DG_ML = ('<-1', '>5', '-')

#: Per-background cap on sampled single mutants, and the stable-hash draw seed.
#: The cap is a compute budget and an equalising rule, not a quality boundary.
VARIANT_CAP = 256
DRAW_SEED = 20260925

#: Radii of the two mutation-centred windows the local-chemistry control reads.
CHEMISTRY_WINDOW_RADII = (2, 4)

#: Detected-similarity thresholds for the remote stratification, as
#: ``(label, identity floor in percent, mutual coverage floor in percent)``. A
#: training background whose local alignment to any held-out background reaches
#: both floors is purged from that fold's training set. The frozen grouping
#: contract already merged everything at 30% identity and 80% mutual coverage, so
#: both rules bite strictly below the contract's own edge rule: the first keeps
#: the contract's coverage requirement and lowers only identity, the second
#: lowers both and is the aggressive variant.
REMOTE_THRESHOLDS: tuple[tuple[str, float, float], ...] = (
    ('identity20_coverage80', 20.0, 80.0),
    ('identity20_coverage50', 20.0, 50.0),
)

#: Chou-Fasman conformational parameters. Declared here because no other module
#: carries them; hydropathy, formal charge and side-chain volume are imported.
HELIX_PROPENSITY: dict[str, float] = {
    'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
    'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
    'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
    'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69,
}
SHEET_PROPENSITY: dict[str, float] = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47,
}

#: The five residue scales the local-chemistry control reads, in order.
CHEMISTRY_SCALES: tuple[str, ...] = ('hydropathy', 'charge', 'volume', 'helix', 'sheet')

_SCALE_TABLES: dict[str, dict[str, float]] = {
    'hydropathy': dict(KYTE_DOOLITTLE),
    'charge': dict(CHARGE_PH7),
    'volume': dict(SIDE_CHAIN_VOLUME),
    'helix': dict(HELIX_PROPENSITY),
    'sheet': dict(SHEET_PROPENSITY),
}

_CODE = {residue: index for index, residue in enumerate(AA20)}


def _blosum_table() -> np.ndarray:
    order = [BLOSUM62_ORDER.index(residue) for residue in AA20]
    return np.array(BLOSUM62_ROWS, dtype=np.float64)[np.ix_(order, order)]


BLOSUM62 = _blosum_table()

_SCALE_VECTORS = {name: np.array([table[residue] for residue in AA20], dtype=np.float64)
                  for name, table in _SCALE_TABLES.items()}


# --------------------------------------------------------------------------- #
# Endpoint construction. One value per exact (background, sequence) state per
# channel, then the two-estimate difference against that background's wild type.
# --------------------------------------------------------------------------- #

def substitution_rows(mut_type) -> np.ndarray:
    """Boolean mask of rows that are substitution constructs.

    An insertion or deletion row carries an ``aa_seq`` truncated to the wild type's
    length, so it is length-matched and would otherwise be read as a substitution
    state. Excluding it is part of the support definition, not a cleanup step.
    """

    text = np.asarray(mut_type, dtype=object).astype(str)
    keep = np.ones(len(text), dtype=bool)
    for prefix in INDEL_MUT_TYPE_PREFIXES:
        keep &= ~np.char.startswith(text.astype(str), prefix)
    return keep


def single_substitution(wildtype: str, sequence: str) -> int | None:
    """Zero-based mutated position, or ``None`` when this is not a single mutant."""

    if len(sequence) != len(wildtype):
        return None
    sites = [index for index in range(len(wildtype)) if sequence[index] != wildtype[index]]
    return sites[0] if len(sites) == 1 else None


def ddg(mutant_value: float, wildtype_value: float) -> float:
    """The declared endpoint: mutant minus wild type on the combined dG_ML scale.

    ``dG_ML`` is an unfolding free energy in kcal/mol, so a destabilising
    substitution lowers it and this difference is negative.
    """

    return float(mutant_value) - float(wildtype_value)


def validate_plan_background(row: dict) -> None:
    """Re-derive every declared state of one plan background rather than trusting it.

    The extraction entry point calls this before a forward pass, so a plan whose
    state index, position or mutant residue disagrees with its own sequences is
    refused instead of silently scoring the wrong string.
    """

    wildtype, sequences = row['wildtype'], row['sequences']
    if sequences[0] != wildtype:
        raise ValueError(f"{row['name']}: the wild type is not the first declared state")
    if len(set(sequences)) != len(sequences):
        raise ValueError(f"{row['name']}: duplicated sequence in one background")
    for variant in row['variants']:
        sequence = sequences[variant['state']]
        site = variant['position'] - 1
        if len(sequence) != len(wildtype):
            raise ValueError(f"{row['name']}: variant state length differs from the wild type")
        differing = [i for i in range(len(wildtype)) if sequence[i] != wildtype[i]]
        if differing != [site] or sequence[site] != variant['mutant']:
            raise ValueError(f"{row['name']}: state {variant['state']} is not the declared "
                             f"single substitution at position {variant['position']}")


def draw_order(background: str, sequences, *, seed: int = DRAW_SEED) -> list[str]:
    """Stable-hash ordering of a background's mutant sequences.

    The key is the seed, the background name and the mutant sequence. It inspects
    no measurement, no confidence width and no model score, so a draw cannot
    favour large measured effects.
    """

    return sorted(sequences, key=lambda s: hashlib.sha256(
        f'{seed}:{background}:{s}'.encode()).digest())


def endpoint_digest(records) -> str:
    """Content digest over the endpoint's own values, in a stable order."""

    payload = '\n'.join(
        f"{row['background']}\t{row['sequence']}\t{row['position']}\t{row['ddg']:.10g}"
        for row in sorted(records, key=lambda r: (r['background'], r['sequence'])))
    return hashlib.sha256(payload.encode()).hexdigest()


#: Protease channels of the pinned files, and the columns that carry them.
CHANNELS = {'trypsin': 'deltaG_t', 'chymotrypsin': 'deltaG_c'}

#: Every column the endpoint and its qualification read.
SOURCE_COLUMNS = [
    'name', 'dna_seq', 'mut_type', 'WT_name', 'WT_cluster', 'aa_seq', 'dG_ML', 'ddG_ML',
    'deltaG', 'deltaG_95CI', 'deltaG_95CI_low', 'deltaG_95CI_high',
    'deltaG_t', 'deltaG_t_95CI', 'deltaG_t_95CI_low', 'deltaG_t_95CI_high',
    'deltaG_c', 'deltaG_c_95CI', 'deltaG_c_95CI_low', 'deltaG_c_95CI_high',
]

CHANNEL_FIT_BOUND_KCAL_MOL = 15.0


def load_source_frame(data_dir):
    """Read the pinned dataset2 parquet files and check every interval's width."""

    import pandas as pd
    from pathlib import Path

    parts = sorted(Path(data_dir).glob('*.parquet'))
    if not parts:
        raise ValueError(f'no parquet files under {data_dir}')
    frame = pd.concat([pd.read_parquet(p, columns=SOURCE_COLUMNS) for p in parts],
                      ignore_index=True)
    for prefix in ('deltaG', 'deltaG_t', 'deltaG_c'):
        width = frame[f'{prefix}_95CI']
        span = frame[f'{prefix}_95CI_high'] - frame[f'{prefix}_95CI_low']
        finite = np.isfinite(width) & np.isfinite(span)
        if not np.allclose(width[finite], span[finite], rtol=1e-6, atol=1e-8):
            raise ValueError(f'{prefix}_95CI differs from its high minus low bound')
    return frame


def accept_rows(frame) -> tuple[np.ndarray, dict]:
    """The admission rule for this endpoint, with the accounting it removes by.

    Finite numeric ``dG_ML``; each of the combined, trypsin and chymotrypsin 95%
    confidence widths in [0, 0.5] kcal/mol; substitution constructs only.
    """

    import pandas as pd

    numeric = pd.to_numeric(frame.dG_ML, errors='coerce')
    finite = np.isfinite(numeric)
    deviation = float(np.abs(numeric[finite] - frame.deltaG[finite]).max())
    if deviation > 1e-6:
        raise ValueError(f'numeric dG_ML departs from deltaG by {deviation} kcal/mol')
    accounting = {
        'rows_total': int(len(frame)),
        'rows_censored_dG_ML': {v: int((frame.dG_ML == v).sum()) for v in CENSORED_DG_ML},
        'censored_handling': 'counted and dropped; never coerced to a numeric endpoint',
        'rows_finite_numeric_dG_ML': int(finite.sum()),
    }
    mask = finite.to_numpy()
    for column in ('deltaG_95CI', 'deltaG_t_95CI', 'deltaG_c_95CI'):
        mask &= frame[column].between(0, QC_WIDTH_KCAL_MOL).to_numpy()
        accounting[f'rows_after_{column}_rule'] = int(mask.sum())
    substitutions = substitution_rows(frame.mut_type.to_numpy())
    accounting['rows_indel_construct'] = int((mask & ~substitutions).sum())
    accounting['indel_prefixes'] = list(INDEL_MUT_TYPE_PREFIXES)
    accounting['indel_rule'] = (
        'an insertion or deletion construct carries an aa_seq truncated to the wild '
        'type length, so it is length-matched and would enter a substitution support; '
        'excluded by mut_type prefix as part of the support definition')
    mask &= substitutions
    accounting['accepted_rows'] = int(mask.sum())
    accounting['accepted_rows_at_channel_fit_bound'] = {
        channel: int((frame.loc[mask, column].abs() == CHANNEL_FIT_BOUND_KCAL_MOL).sum())
        for channel, column in CHANNELS.items()}
    accounting['accepted_channel_range_kcal_mol'] = {
        channel: [float(frame.loc[mask, column].min()), float(frame.loc[mask, column].max())]
        for channel, column in CHANNELS.items()}
    accounting['accepted_combined_range_kcal_mol'] = [
        float(numeric[mask].min()), float(numeric[mask].max())]
    return mask, accounting


def aggregate_states(frame, mask: np.ndarray, *, statistic: str = 'median'):
    """One value per exact (WT_name, aa_seq) per channel, identical rows per channel."""

    import pandas as pd

    if statistic not in ('median', 'first'):
        raise ValueError('statistic must be median or first')
    rows = frame.loc[mask, ['WT_name', 'aa_seq', 'name', 'dna_seq', 'dG_ML',
                            *CHANNELS.values()]].copy()
    rows['combined'] = pd.to_numeric(rows.dG_ML, errors='coerce')
    grouped = rows.groupby(['WT_name', 'aa_seq'], sort=True)
    columns = [*CHANNELS.values(), 'combined']
    values = grouped[columns].median() if statistic == 'median' else grouped[columns].first()
    values['n_rows'] = grouped.size()
    values['n_dna'] = grouped['dna_seq'].nunique()
    return values.reset_index()


def build_endpoint(values, catalogue: dict):
    """Single-mutant ddG per channel and on the combined scale, one row per variant."""

    import pandas as pd

    frames, accounting = [], {'backgrounds_in_catalogue': len(catalogue),
                              'backgrounds_without_accepted_wt': 0,
                              'backgrounds_with_variants': 0}
    for name, part in values.groupby('WT_name', sort=True):
        entry = catalogue.get(name)
        if entry is None:
            continue
        wildtype = entry['sequence']
        part = part[part.aa_seq.str.len() == len(wildtype)]
        wt_row = part[part.aa_seq == wildtype]
        if wt_row.empty:
            accounting['backgrounds_without_accepted_wt'] += 1
            continue
        wt_row = wt_row.iloc[0]
        positions, mutants, keep = [], [], []
        for sequence in part.aa_seq:
            position = single_substitution(wildtype, sequence)
            if position is None or any(a not in AA20 for a in sequence):
                keep.append(False)
                continue
            keep.append(True)
            positions.append(position + 1)
            mutants.append(sequence[position])
        part = part[np.asarray(keep)]
        if part.empty:
            continue
        record = pd.DataFrame({
            'WT_name': name, 'kind': entry['kind'], 'cluster': entry['cluster'],
            'group': entry['group'], 'sequence': part.aa_seq.to_numpy(),
            'position': positions, 'mutant': mutants,
            'n_rows': part.n_rows.to_numpy(), 'n_dna': part.n_dna.to_numpy()})
        for channel, column in CHANNELS.items():
            record[channel] = part[column].to_numpy(float) - float(wt_row[column])
        record['combined'] = part.combined.to_numpy(float) - float(wt_row.combined)
        record['wildtype_combined'] = float(wt_row.combined)
        frames.append(record)
        accounting['backgrounds_with_variants'] += 1
    if not frames:
        raise ValueError('no single-mutant variant survived the accepted support')
    endpoint = pd.concat(frames, ignore_index=True)
    sample = endpoint.iloc[0]
    if abs(ddg(sample.combined + sample.wildtype_combined, sample.wildtype_combined)
           - sample.combined) > 1e-9:
        raise ValueError('endpoint construction disagrees with its declared definition')
    return endpoint, accounting


# --------------------------------------------------------------------------- #
# Declared control blocks. Every block is a function of sequences, frozen
# sequence-derived quantities or training-group labels only.
# --------------------------------------------------------------------------- #

#: The base every candidate control is qualified over: the directed substitution
#: identity and the mutated position's geometry. This is the level the capability
#: map records as settled, and it carries no chemistry, no composition and no
#: evolutionary statistic.
BASE_BLOCKS = ('ident', 'geom')

#: Candidate controls, in the order they are offered to the qualification rule.
#: Each is a competing explanation for a stability change, never a mechanism.
#:
#: ``prof2`` is a bounded restatement of the same independent-site profile
#: information, added to the ladder on 2026-09-24 after the first pass measured
#: ``prof`` failing the rule and the coordinate ranges showed why: on a
#: near-complete single-substitution scan against shallow alignments, the raw
#: mutant log frequency sits at the pseudo-count floor of log(1e-6) for the
#: majority of variants, so that coordinate carries almost no gradation, while
#: the column-weight coordinates span 0 to 5,986 effective sequences and do not
#: standardise across held-out groups. The amendment is a function of coordinate
#: ranges alone and was fixed while no model likelihood or representation had
#: been read at all, so it cannot be tuned to a model outcome. Both verdicts are
#: reported.
CANDIDATE_BLOCKS = ('comp', 'chem', 'prof', 'prof2', 'G')

#: Ordered names of the mutation-local profile summaries. The last coordinate is
#: the availability indicator: a background that retrieved no qualifying homolog
#: carries zeros before it and a zero here, a declared absence rather than an
#: imputed profile.
PROFILE_FEATURE_ORDER = (
    'log_frequency_wild', 'log_frequency_mutant', 'log_odds',
    'column_entropy_nats', 'column_weight',
    'window_mean_entropy_nats', 'window_mean_column_weight',
    'wildtype_log10_neff', 'wildtype_supported_column_fraction', 'profile_available',
)

#: Tokenisation interface descriptors for the two states. A representation
#: difference can be nonzero purely through segmentation, so these are offered to
#: each arm's own matched baseline under the same qualification rule.
TOKENISATION_FEATURE_ORDER = (
    'pooled_tokens_wild', 'pooled_tokens_mutant', 'pooled_token_count_difference',
    'common_prefix_tokens', 'common_suffix_tokens', 'segmentation_changed_tokens',
    'pooled_tokens_per_residue_wild', 'pooled_tokens_per_residue_mutant',
)

#: First-stage absolute-state descriptors for the nonlinear-additive response G.
G_STATE_BLOCKS = ('count', 'ident', 'comp', 'geom', 'chem', 'prof2')


def _codes(sequence: str) -> list[int]:
    try:
        return [_CODE[residue] for residue in sequence]
    except KeyError as error:
        raise ValueError(f'noncanonical residue {error.args[0]!r} in a cohort sequence') from error


def _composition(sequence: str) -> np.ndarray:
    return np.array([sequence.count(residue) for residue in AA20], dtype=float) / len(sequence)


def identity_block(wildtype: str, position: int, mutant: str) -> np.ndarray:
    """400 directed substitution-count coordinates for one single mutant."""

    block = np.zeros((len(AA20), len(AA20)))
    block[_CODE[wildtype[position]], _CODE[mutant]] = 1.0
    return block.ravel()


def geometry_block(position: int, length: int) -> np.ndarray:
    """Mutated-position geometry and sequence length, 6 coordinates."""

    relative = (position + 1) / length
    from_end = min(position + 1, length - position) / length
    return np.array([relative, from_end, np.log10(position + 1),
                     np.log10(max(length - position, 1)), length / 1024,
                     float(position == 0 or position == length - 1)])


def composition_block(wildtype: str, mutant_sequence: str) -> np.ndarray:
    """Wild-type and mutant amino-acid composition, 40 coordinates."""

    return np.concatenate([_composition(wildtype), _composition(mutant_sequence)])


def chemistry_block(wildtype: str, position: int, mutant: str) -> np.ndarray:
    """Local chemistry in mutation-centred windows.

    For each declared residue scale: the wild-type and mutant value at the site
    and their difference, and, at each declared window radius, the window mean of
    the wild-type and of the mutant sequence and their difference. Then the
    substitution-matrix score of the substitution itself and, in each window, the
    mean substitution-matrix score of the mutant residue and of the wild-type
    residue against the flanking wild-type residues, with their difference.

    These are competing explanations for a stability change, not a mechanism: a
    window mean is an aggregate of the same residues the identity block already
    names, read on a declared physical scale.
    """

    codes = np.asarray(_codes(wildtype))
    mutant_codes = codes.copy()
    mutant_codes[position] = _CODE[mutant]
    values = []
    for name in CHEMISTRY_SCALES:
        table = _SCALE_VECTORS[name]
        wild_site, mutant_site = table[codes[position]], table[_CODE[mutant]]
        values.extend([wild_site, mutant_site, mutant_site - wild_site])
        for radius in CHEMISTRY_WINDOW_RADII:
            low = max(position - radius, 0)
            high = min(position + radius + 1, len(codes))
            wild_window = float(table[codes[low:high]].mean())
            mutant_window = float(table[mutant_codes[low:high]].mean())
            values.extend([wild_window, mutant_window, mutant_window - wild_window])
    values.append(float(BLOSUM62[codes[position], _CODE[mutant]]))
    for radius in CHEMISTRY_WINDOW_RADII:
        low = max(position - radius, 0)
        high = min(position + radius + 1, len(codes))
        flank = np.concatenate([codes[low:position], codes[position + 1:high]])
        if flank.size:
            mutant_fit = float(BLOSUM62[_CODE[mutant], flank].mean())
            wild_fit = float(BLOSUM62[codes[position], flank].mean())
        else:
            mutant_fit = wild_fit = 0.0
        values.extend([wild_fit, mutant_fit, mutant_fit - wild_fit])
    return np.asarray(values, dtype=float)


def chemistry_width() -> int:
    return len(CHEMISTRY_SCALES) * (3 + 3 * len(CHEMISTRY_WINDOW_RADII)) + 1 + 3 * len(
        CHEMISTRY_WINDOW_RADII)


def profile_block(profile, wildtype: str, position: int, mutant: str,
                  *, radius: int = max(CHEMISTRY_WINDOW_RADII)) -> np.ndarray:
    """Mutation-local independent-site profile summaries, in the declared order."""

    values = np.zeros(len(PROFILE_FEATURE_ORDER))
    if profile is None:
        return values
    frequencies = np.clip(profile.frequencies, PROFILE_PSEUDO_FREQUENCY, None)
    entropy = -(frequencies * np.log(frequencies)).sum(axis=1)
    wild = float(np.log(frequencies[position, _CODE[wildtype[position]]]))
    mutated = float(np.log(frequencies[position, _CODE[mutant]]))
    low, high = max(position - radius, 0), min(position + radius + 1, profile.length)
    values[0], values[1], values[2] = wild, mutated, mutated - wild
    values[3] = float(entropy[position])
    values[4] = float(profile.column_weight[position])
    values[5] = float(entropy[low:high].mean())
    values[6] = float(profile.column_weight[low:high].mean())
    values[7] = float(profile.log10_neff)
    values[8] = float((profile.column_weight > 0).sum()) / profile.length
    values[9] = 1.0
    return values


#: Bounded restatement of the same mutation-local profile information. Every
#: coordinate lies in a declared range, and the coordinate the raw block saturates
#: is replaced by the indicator that actually carries its information: whether the
#: mutant residue was observed in the alignment column at all.
PROFILE_BOUNDED_FEATURE_ORDER = (
    'log_frequency_wild_clipped', 'log_odds_clipped', 'mutant_unobserved',
    'column_entropy_nats', 'window_mean_entropy_nats',
    'log10_1p_column_weight', 'log10_1p_window_mean_column_weight',
    'wildtype_log10_neff', 'wildtype_supported_column_fraction', 'profile_available',
)

#: Range the bounded profile coordinates clip log quantities to. A ratio beyond
#: exp(6) is already a 403-fold one and the raw coordinate is saturated there.
PROFILE_LOG_CLIP = 6.0


def profile_bounded_block(profile, wildtype: str, position: int, mutant: str,
                          *, radius: int = max(CHEMISTRY_WINDOW_RADII)) -> np.ndarray:
    """The bounded profile control, in the declared order."""

    values = np.zeros(len(PROFILE_BOUNDED_FEATURE_ORDER))
    if profile is None:
        return values
    frequencies = np.clip(profile.frequencies, PROFILE_PSEUDO_FREQUENCY, None)
    entropy = -(frequencies * np.log(frequencies)).sum(axis=1)
    wild = float(np.log(frequencies[position, _CODE[wildtype[position]]]))
    mutated = float(np.log(frequencies[position, _CODE[mutant]]))
    low, high = max(position - radius, 0), min(position + radius + 1, profile.length)
    values[0] = float(np.clip(wild, -PROFILE_LOG_CLIP, 0.0))
    values[1] = float(np.clip(mutated - wild, -PROFILE_LOG_CLIP, PROFILE_LOG_CLIP))
    values[2] = float(profile.frequencies[position, _CODE[mutant]] <= PROFILE_PSEUDO_FREQUENCY)
    values[3] = float(entropy[position])
    values[4] = float(entropy[low:high].mean())
    values[5] = float(np.log10(1.0 + profile.column_weight[position]))
    values[6] = float(np.log10(1.0 + profile.column_weight[low:high].mean()))
    values[7] = float(profile.log10_neff)
    values[8] = float((profile.column_weight > 0).sum()) / profile.length
    values[9] = 1.0
    return values


def state_features(wildtype: str, sequence: str, profile) -> np.ndarray:
    """Absolute-state descriptors for the nonlinear-additive response's first stage.

    A wild-type state carries a zero substitution count and zero mutation-local
    coordinates, which is what makes the wild type and its mutants one regression
    over absolute stability rather than two.
    """

    sites = [index for index in range(len(wildtype)) if sequence[index] != wildtype[index]]
    identity = np.zeros((len(AA20), len(AA20)))
    chemistry = np.zeros(chemistry_width())
    prof = np.zeros(len(PROFILE_BOUNDED_FEATURE_ORDER))
    geometry = np.zeros(6)
    if sites:
        position = sites[0]
        identity += identity_block(wildtype, position, sequence[position]).reshape(
            len(AA20), len(AA20))
        chemistry = chemistry_block(wildtype, position, sequence[position])
        prof = profile_bounded_block(profile, wildtype, position, sequence[position])
        geometry = geometry_block(position, len(wildtype))
    elif profile is not None:
        prof[7] = float(profile.log10_neff)
        prof[8] = float((profile.column_weight > 0).sum()) / profile.length
        prof[9] = 1.0
    return np.concatenate([[float(len(sites))], identity.ravel(),
                           composition_block(wildtype, sequence), geometry, chemistry, prof])


def tokenisation_block(pooled_counts: np.ndarray, token_ids: list[list[int]],
                       wild_index: int, mutant_index: int, length: int) -> np.ndarray:
    """Per-state token counts, their difference and the segmentation change."""

    reference, other = token_ids[wild_index], token_ids[mutant_index]
    prefix = 0
    while prefix < min(len(reference), len(other)) and reference[prefix] == other[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(reference), len(other)) - prefix
           and reference[-1 - suffix] == other[-1 - suffix]):
        suffix += 1
    changed = len(reference) + len(other) - 2 * (prefix + suffix)
    wild_count = float(pooled_counts[wild_index])
    mutant_count = float(pooled_counts[mutant_index])
    return np.array([wild_count, mutant_count, mutant_count - wild_count,
                     float(prefix), float(suffix), float(changed),
                     wild_count / length, mutant_count / length])


# --------------------------------------------------------------------------- #
# The nonlinear-additive global response, the qualification rule and the nested
# held-group comparison.
# --------------------------------------------------------------------------- #

def nuisance_response(states: dict, pair_states: np.ndarray, held: np.ndarray,
                      inner_partition: list, *, device='cpu') -> tuple[np.ndarray, dict]:
    """The nonlinear-additive global response G for one outer fold.

    A first-stage ridge predictor of absolute stability is fitted on the outer
    training groups only, its penalty selected on the same inner held-group
    partition the outer tuning uses. Training-state predictions are cross-fitted
    on that partition, an isotonic response is fitted on those cross-fitted
    predictions, and the response is applied to both states of every variant. The
    resulting calibrated difference lets a global monotone response on absolute
    stability account for part of the endpoint without any mutation-specific term
    beyond the first stage's own additive features. No held-out group label
    reaches the first stage, the penalty choice or the response.
    """

    from sklearn.isotonic import IsotonicRegression

    x, y = states['features'], states['y']
    groups, weights = states['group'], states['weight']
    train = np.flatnonzero(~np.isin(groups, held))
    test = np.flatnonzero(np.isin(groups, held))
    if not len(train) or not len(test):
        raise ValueError('nuisance fold has an empty side')
    losses = np.zeros(len(ALPHAS))
    crossfit = np.full((len(train), len(ALPHAS)), np.nan)
    position = {index: order for order, index in enumerate(train)}
    for validation in inner_partition:
        fit = train[~np.isin(groups[train], validation)]
        valid = train[np.isin(groups[train], validation)]
        if not len(fit) or not len(valid):
            raise ValueError('nuisance inner fold has an empty side')
        prediction = ridge_predict(x[fit], y[fit], weights[fit], x[valid], ALPHAS, device)
        crossfit[[position[index] for index in valid]] = prediction
        share = weights[valid] / weights[valid].sum()
        losses += len(validation) * (share[:, None] * (prediction - y[valid][:, None]) ** 2).sum(0)
    if not np.isfinite(crossfit).all():
        raise ValueError('incomplete cross-fitted first-stage predictions')
    index = _select_alpha(losses / len(set(groups[train])))
    alpha = ALPHAS[index]
    fitted = np.empty(len(y))
    fitted[train] = crossfit[:, index]
    fitted[test] = ridge_predict(x[train], y[train], weights[train], x[test], [alpha], device)[:, 0]
    response = IsotonicRegression(increasing=True, out_of_bounds='clip')
    response.fit(crossfit[:, index], y[train], sample_weight=weights[train])
    wild, mutant = pair_states[:, 0], pair_states[:, 1]
    calibrated = response.predict(fitted[mutant]) - response.predict(fitted[wild])
    block = np.column_stack([calibrated, fitted[mutant] - fitted[wild],
                             fitted[wild], fitted[mutant]])
    share = weights[test] / weights[test].sum()
    centre = float((share * y[test]).sum())
    residual = float((share * (y[test] - fitted[test]) ** 2).sum())
    variance = float((share * (y[test] - centre) ** 2).sum())
    low, high = float(crossfit[:, index].min()), float(crossfit[:, index].max())
    diagnostics = {
        'selected_alpha': alpha,
        'held_out_states': int(len(test)),
        'held_out_weighted_r2': None if variance <= 0 else 1.0 - residual / variance,
        'response_range_kcal_mol': float(np.diff(response.predict(np.array([low, high])))[0]),
        'first_stage_states': int(len(y)),
    }
    return block, diagnostics


def fold_predictions(panel: dict, blocks: dict, design: dict, *, seed: int,
                     device='cpu', purge=None) -> dict:
    """Held-group predictions for every named design, on one support and one seed.

    Every design sees the identical rows, the identical outer and inner group
    partitions and the identical label budget; only its declared column blocks
    differ. Centring and scaling are fitted on training rows inside the ridge
    recipe, so no held-out group label reaches feature scaling, penalty selection
    or the nonlinear response.

    ``purge`` optionally maps a held-out group to the training groups removed for
    that fold. The held-out rows are unchanged, so a purged fit is paired with an
    unpurged one on exactly the same evaluation rows.
    """

    groups = panel['group']
    weights = row_weights(panel['site'], groups)
    predictions = {name: np.full(len(groups), np.nan) for name in design}
    records, nuisance = [], []
    for outer, held in enumerate(family_folds(groups, OUTER_SPLITS, seed)):
        held = list(held)
        excluded = sorted({g for key in held for g in (purge or {}).get(key, ())} - set(held))
        train = np.flatnonzero(~np.isin(groups, held + excluded))
        test = np.flatnonzero(np.isin(groups, held))
        if not len(train) or not len(test):
            raise ValueError(f'fold {outer} has an empty side after the declared purge')
        remaining = len(set(groups[train]))
        if remaining < INNER_SPLITS:
            raise ValueError(
                f'fold {outer}: the declared purge of {len(excluded)} training groups leaves '
                f'{remaining} training groups, below the {INNER_SPLITS} the inner partition '
                'needs; the purge is not evaluable on this support and no weaker split is '
                'substituted')
        inner_partition = family_folds(groups[train], INNER_SPLITS, seed + 100 + outer)
        fold_blocks = dict(blocks)
        if any('G' in columns for columns in design.values()):
            block, diagnostics = nuisance_response(
                panel['states'], panel['pair_states'], np.asarray(held + excluded),
                inner_partition, device=device)
            diagnostics.update(fold=outer, held_groups=held, purged_training_groups=excluded)
            nuisance.append(diagnostics)
            fold_blocks['G'] = block
        record = {'fold': outer, 'held_groups': held, 'purged_training_groups': excluded,
                  'training_groups': sorted(set(groups[train])), 'alpha': {}, 'dimensions': {}}
        for name, columns in design.items():
            x = np.column_stack([fold_blocks[key] for key in columns])
            if not np.isfinite(x).all():
                raise ValueError(f'{name}: nonfinite design matrix')
            losses = np.zeros(len(ALPHAS))
            for validation in inner_partition:
                fit = train[~np.isin(groups[train], validation)]
                valid = train[np.isin(groups[train], validation)]
                predicted = ridge_predict(x[fit], panel['target'][fit], weights[fit],
                                          x[valid], ALPHAS, device)
                losses += len(validation) * (
                    weights[valid][:, None] * (predicted - panel['target'][valid][:, None]) ** 2).sum(0)
            alpha = ALPHAS[_select_alpha(losses / len(set(groups[train])))]
            predictions[name][test] = ridge_predict(
                x[train], panel['target'][train], weights[train], x[test], [alpha], device)[:, 0]
            record['alpha'][name] = alpha
            record['dimensions'][name] = int(x.shape[1])
        records.append(record)
    for name, value in predictions.items():
        if not np.isfinite(value).all():
            raise ValueError(f'{name}: incomplete held-out predictions')
    predictions['NO_EFFECT_NULL'] = np.zeros(len(groups))
    return {'predictions': predictions, 'folds': records, 'nuisance': nuisance,
            'row_weights': weights}


def paired_increment(panel: dict, predictions: dict, augmented: str, baseline: str,
                     *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired per-group reduction in mean squared error, in squared kcal/mol."""

    labels, base = group_errors(panel['target'], predictions[baseline],
                                panel['group'], panel['site'])
    _, better = group_errors(panel['target'], predictions[augmented],
                             panel['group'], panel['site'])
    record = interval(base - better, draws=draws, seed=seed)
    record.update(baseline=baseline, augmented=augmented,
                  baseline_mse=float(base.mean()), augmented_mse=float(better.mean()))
    return record


def spearman_increment(panel: dict, predictions: dict, augmented: str, baseline: str,
                       *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired per-group within-background rank-correlation increment."""

    _, base = group_spearman(panel['target'], predictions[baseline], panel['group'])
    _, better = group_spearman(panel['target'], predictions[augmented], panel['group'])
    paired = [None if a is None or b is None else b - a for a, b in zip(base, better)]
    return interval(paired, draws=draws, seed=seed)


def load_profiles(path, wildtypes: dict[str, str]) -> tuple[dict, dict]:
    """Reconstruct one independent-site profile per cohort background.

    A background that retrieved no qualifying homolog row is ``None`` and carries
    a zero availability indicator downstream; it is never given a pooled or
    nearest-background profile.
    """

    import json

    from ..context.profiles import Profile

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        profiles: dict[str, object] = {}
        for record in meta['backgrounds']:
            name = record['name']
            if name not in wildtypes:
                continue
            if record['status'] != 'present':
                profiles[name] = None
                continue
            scalars = data[f'{name}|scalars']
            profiles[name] = Profile(
                query_id=name, wildtype=wildtypes[name], n_hits=int(scalars[4]),
                n_sequences=int(scalars[3]), saturated=False,
                frequencies=data[f'{name}|frequencies'].astype(np.float64),
                column_weight=data[f'{name}|column_weight'].astype(np.float64),
                neff=float(scalars[0]), max_identity_over_query=float(scalars[2]))
    missing = sorted(set(wildtypes) - set(profiles))
    if missing:
        raise ValueError(f'profile file does not cover {len(missing)} cohort backgrounds')
    return profiles, meta


def build_panel(cohort: dict, profiles: dict) -> dict:
    """Rows, target, group and site labels, control blocks and the G first stage.

    The first stage's state table holds every background's wild type once followed
    by its sampled mutants, so a wild-type measurement enters the absolute-stability
    regression once and its difference against each mutant is formed by index.
    """

    target, group, site = [], [], []
    ident, geom, comp, chem, prof, prof2 = [], [], [], [], [], []
    state_rows, state_y, state_group, state_site, pair_states = [], [], [], [], []
    for row in cohort['backgrounds']:
        name, wildtype = row['name'], row['wildtype']
        profile = profiles[name]
        wild_state = len(state_rows)
        state_rows.append(state_features(wildtype, wildtype, profile))
        state_y.append(row['wildtype_combined_kcal_mol'])
        state_group.append(row['group'])
        state_site.append(f'{name}:wt')
        for variant in row['variants']:
            position = variant['position'] - 1
            mutant, sequence = variant['mutant'], variant['sequence']
            target.append(variant['ddg'])
            group.append(row['group'])
            site.append(f"{name}:{variant['position']}")
            ident.append(identity_block(wildtype, position, mutant))
            geom.append(geometry_block(position, len(wildtype)))
            comp.append(composition_block(wildtype, sequence))
            chem.append(chemistry_block(wildtype, position, mutant))
            prof.append(profile_block(profile, wildtype, position, mutant))
            prof2.append(profile_bounded_block(profile, wildtype, position, mutant))
            pair_states.append([wild_state, len(state_rows)])
            state_rows.append(state_features(wildtype, sequence, profile))
            state_y.append(row['wildtype_combined_kcal_mol'] + variant['ddg'])
            state_group.append(row['group'])
            state_site.append(f"{name}:{variant['position']}")
    states = {'features': np.asarray(state_rows, dtype=float),
              'y': np.asarray(state_y, dtype=float),
              'group': np.asarray(state_group),
              'weight': row_weights(np.asarray(state_site), np.asarray(state_group))}
    return {'target': np.asarray(target, dtype=float), 'group': np.asarray(group),
            'site': np.asarray(site), 'pair_states': np.asarray(pair_states, dtype=int),
            'states': states,
            'blocks': {'ident': np.asarray(ident, dtype=float),
                       'geom': np.asarray(geom, dtype=float),
                       'comp': np.asarray(comp, dtype=float),
                       'chem': np.asarray(chem, dtype=float),
                       'prof': np.asarray(prof, dtype=float),
                       'prof2': np.asarray(prof2, dtype=float)}}


#: A candidate control is qualified only if it is a competent held-out predictor
#: in its own right: adding it to the standing set must not lower that set's
#: group-equal held-out mean squared error at any prespecified split seed. A
#: candidate that lowers it is discarded and reported, never carried forward.
QUALIFICATION_SEEDS = SPLIT_SEEDS


#: A discarded candidate that restates a kept one's information is superseded by it,
#: so the two are never both in one design.
SUPERSEDED = {'prof': 'prof2'}


def secondary_control_set(controls: dict) -> tuple[tuple[str, ...], dict]:
    """The control set the correlation rule accepts, derived from the frozen ladder.

    The prespecified qualification rule is stated on group-equal held-out mean
    squared error, and the capability map's binding rule for a control is stated on
    held-out correlation. For the mutation-local profile block the two disagree on
    this endpoint: its within-background rank increment is above zero at every
    split seed while its squared-error increment is catastrophically below zero at
    one of them, because it transfers rank and does not transfer level. Rather than
    choose a metric after seeing that, both sets are carried: the primary set is
    the one the mean-squared-error rule qualified, and this secondary set adds every
    discarded candidate whose correlation increment is positive at all three seeds,
    in ladder order, with a superseded candidate replaced by its restatement. A
    squared-error increment over the secondary set is reported as a sensitivity and
    a correlation increment over the primary set likewise; each rule licenses the
    increments read on its own metric.
    """

    kept = list(controls['qualified_control_set'])
    added, reasons = [], {}
    for row in controls['ladder']:
        candidate = row['candidate']
        # A block that another block restates is judged only through that
        # restatement, so a superseded candidate never admits itself and never
        # admits its restatement on the strength of its own numbers.
        if row['qualified'] or candidate in SUPERSEDED or candidate in kept:
            continue
        points = [row['spearman_increment'][str(seed)]['point'] for seed in QUALIFICATION_SEEDS]
        if any(point is None or point <= 0 for point in points):
            continue
        added.append(candidate)
        reasons[candidate] = {'offered_as': candidate,
                              'per_seed_spearman_increment': points,
                              'per_seed_mse_increment_kcal2_mol2': list(
                                  row['per_seed_increment_kcal2_mol2'].values())}
    wanted, ordered = set(kept) | set(added), []
    for candidate in controls['candidate_order']:
        block = SUPERSEDED.get(candidate, candidate)
        if block in wanted and block not in ordered:
            ordered.append(block)
    secondary = tuple([*controls['base_blocks'], *ordered])
    return secondary, {'added_over_primary': added, 'rule': secondary_control_set.__doc__.strip(),
                       'evidence': reasons}


def qualify(increments: dict[int, float]) -> dict:
    """Apply the qualification rule to one candidate's per-seed increments."""

    if set(increments) != set(QUALIFICATION_SEEDS):
        raise ValueError('a candidate must be scored at every prespecified split seed')
    values = [float(increments[seed]) for seed in QUALIFICATION_SEEDS]
    return {'per_seed_increment_kcal2_mol2': {str(s): v for s, v in zip(QUALIFICATION_SEEDS, values)},
            'min_increment_kcal2_mol2': min(values),
            'qualified': bool(min(values) > 0.0)}


__all__ = [
    'AA20', 'BASE_BLOCKS', 'BLOSUM62', 'CANDIDATE_BLOCKS', 'CENSORED_DG_ML',
    'CHEMISTRY_SCALES', 'CHEMISTRY_WINDOW_RADII', 'DRAW_SEED', 'ENDPOINT',
    'FEATURE_BLOCKS', 'G_STATE_BLOCKS', 'HELIX_PROPENSITY',
    'INDEL_MUT_TYPE_PREFIXES', 'PROFILE_BOUNDED_FEATURE_ORDER', 'PROFILE_FEATURE_ORDER', 'PROJECTION_DIM',
    'PROJECTION_SEED', 'QC_SOURCE', 'QC_WIDTH_COLUMNS', 'QC_WIDTH_KCAL_MOL',
    'QUALIFICATION_SEEDS', 'REMOTE_THRESHOLDS', 'SUPERSEDED', 'secondary_control_set',
    'SHEET_PROPENSITY', 'SPLIT_SEEDS', 'TOKENISATION_FEATURE_ORDER', 'VARIANT_CAP',
    'chemistry_block', 'chemistry_width', 'composition_block', 'ddg', 'draw_order',
    'endpoint_digest', 'fold_predictions', 'geometry_block', 'identity_block',
    'interval', 'load_profiles', 'build_panel', 'profile_bounded_block', 'load_source_frame', 'accept_rows',
    'aggregate_states', 'build_endpoint', 'CHANNELS', 'group_errors', 'group_spearman',
    'nuisance_response', 'paired_increment', 'profile_block',
    'projection_matrices', 'qualify', 'single_substitution', 'spearman_increment',
    'state_features', 'substitution_rows', 'tokenisation_block',
    'validate_plan_background',
]
