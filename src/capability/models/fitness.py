"""Zero-shot DMS fitness scoring, and the free baseline it has to be read against.

**Why this module exists.** ProGenMech (arXiv:2606.16044) reports that a sparse
circuit over a cross-layer transcoder recovers "~95%" and "~80%" of ProGen3's
zero-shot fitness performance. Both figures are ratios of Spearman correlations
against a base of **0.29**, and neither the base nor the ratio carries a floor.
Standing rule 28 requires a selector to be scored against the trivial baseline
available from its own coordinates; the analogue for a fitness predictor is the
score computable from the mutation string alone, before any model exists. This
module supplies that baseline (BLOSUM62), the cohort machinery for the eight
assays their released artefacts name, and the scoring convention all arms share.

**One fact that removes a candidate explanation, recorded because it is easy to
get wrong.** A ProteinGym substitution assay has a single wild type, so the
mutant-minus-wildtype log-likelihood differs from the raw mutant log-likelihood
by a constant. Spearman is invariant to it: the two conventions give *bit
identical* correlations, measured (0.5021144742520806 both ways on 400 single
mutants of GRB2_HUMAN_Faure_2021). Sequence length is likewise constant within an
assay, so summing and averaging over positions also rank-tie. Neither choice can
explain a disagreement between two reported fitness numbers on the same assay,
and this module therefore does not expose them as options.

BLOSUM62 is free of the *model and the method*, which is what rule 28 asks for.
It is not free of biology -- it is estimated from aligned protein blocks -- and
a claim built on it must say so rather than calling it uninformed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.amino_acids import BLOSUM62_ORDER, BLOSUM62_ROWS
from ..core.sequence_metadata import PROTEINGYM_ROOT

#: The eight ProteinGym substitution assays ProGenMech evaluates on. Not taken
#: from the paper's prose: read from the directory names inside their released
#: ``ProGenMechData/functions.tar.gz``, which holds one circuit-discovery output
#: per assay, so this list is what their own artefacts were produced over.
PROGENMECH_ASSAYS: tuple[str, ...] = (
    "A4_HUMAN_Seuma_2022",
    "CAPSD_AAV2S_Sinai_2021",
    "F7YBW8_MESOW_Ding_2023",
    "GFP_AEQVI_Sarkisyan_2016",
    "GRB2_HUMAN_Faure_2021",
    "RASK_HUMAN_Weng_2022_abundance",
    "SPG1_STRSG_Olson_2014",
    "YAP1_HUMAN_Araya_2012",
)

#: Their ``function_circuit/discover_circuits.py`` defaults, so a run of this
#: module can be set to their sampling condition by name rather than by memory.
PROGENMECH_TEST_SEQUENCES = 1000

#: Their training-split size, removed from the pool before the test draw.
PROGENMECH_TRAIN_SEQUENCES = 256

#: ``seed = 42 + fold`` for five folds (``discover_circuits.py:200-202``).
PROGENMECH_FOLD_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

#: BLOSUM62, as ``(wild_type, mutant) -> score``. Pair scores come from the
#: NCBI table in :mod:`.amino_acids`; this dict is the fitness-scoring presentation.
BLOSUM62: dict[tuple[str, str], int] = {
    (left, right): score
    for left, row in zip(BLOSUM62_ORDER, BLOSUM62_ROWS)
    for right, score in zip(BLOSUM62_ORDER, row)
}


@dataclass(frozen=True)
class Assay:
    """One DMS assay, drawn to a declared size under a declared seed."""

    name: str
    wildtype: str
    mutants: list[str]
    sequences: list[str]
    scores: np.ndarray
    n_eligible: int
    seed: int
    sampling: str

    @property
    def blosum(self) -> np.ndarray:
        """Summed BLOSUM62 score of each variant's substitutions."""

        return np.array(
            [sum(BLOSUM62[(t[0], t[-1])] for t in m.split(":")) for m in self.mutants],
            dtype=np.float64,
        )

    @property
    def substitutions(self) -> list[tuple[tuple[str, int, str], ...]]:
        """Each variant's substitutions as ``(wild, 1-based position, mutant)``.

        The parse ``Assay.blosum`` performs inline, exposed once so that a
        second predictor over the same variants cannot disagree with the first
        about what a mutation string means (Appendix B rule 12).
        """

        return [parse_mutant(mutant) for mutant in self.mutants]

    @property
    def n_multi_mutant_drawn(self) -> int:
        """Drawn variants carrying more than one substitution.

        Over the drawn variants and not the eligible pool. It used to be counted
        while the file was read, which put a pool-wide count -- up to 535,917 --
        beside a drawn ``n_variants`` of 1000 in the same record, and the
        composition of the drawn cohort is exactly what the ProGenMech design
        comparison turns on. Derived from :func:`parse_mutant` so that the
        substitution grammar has one declaration (Appendix B rule 12).
        """

        return sum(1 for entry in self.substitutions if len(entry) > 1)

    def record(self) -> dict:
        return {
            "assay": self.name,
            "wildtype_length": len(self.wildtype),
            "n_variants": len(self.sequences),
            "n_eligible": self.n_eligible,
            "n_multi_mutant_drawn": self.n_multi_mutant_drawn,
            "seed": self.seed,
            "sampling": self.sampling,
        }


def parse_mutant(mutant: str) -> tuple[tuple[str, int, str], ...]:
    """``"A12G:C40W"`` to ``(("A", 12, "G"), ("C", 40, "W"))``.

    One declaration of the mutation-string grammar. ``_revert`` and
    ``Assay.blosum`` each carried their own slicing of it, and a third consumer
    -- a position-specific profile, which needs the position and both residues at
    once -- is the point at which two spellings become a disagreement about which
    residue a variant carries.
    """

    tokens = mutant.split(":")
    if not tokens or any(len(token) < 3 for token in tokens):
        raise ValueError(f"{mutant!r} is not a substitution string")
    parsed: list[tuple[str, int, str]] = []
    for token in tokens:
        position = token[1:-1]
        if not position.isdigit():
            raise ValueError(f"{mutant!r}: token {token!r} carries no position")
        parsed.append((token[0], int(position), token[-1]))
    return tuple(parsed)


def available_assays(directory: Path | None = None) -> tuple[str, ...]:
    """Every ProteinGym substitution assay staged on this host, sorted.

    Read from the directory rather than declared, because the benchmark is the
    population and a hand-maintained list is the shape Appendix B rule 12 warns
    about: EXP-R2-134's whole-benchmark result is a statement about *all* 217
    assays, and a list that drifts from the directory would quietly narrow it.
    """

    root = Path(directory) if directory is not None else PROTEINGYM_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist; set TRANSFER_PROTEINGYM_DIR to its location"
        )
    names = tuple(sorted(path.stem for path in root.glob("*.csv")))
    if not names:
        raise RuntimeError(f"no ProteinGym assay files under {root}")
    return names


def wildtype_of(name: str, directory: Path | None = None) -> str:
    """The wild type of one assay, from its first row alone.

    :func:`load_assay` also returns it, but only after reading and checking every
    row of a file that runs to hundreds of thousands of variants. Enumerating the
    benchmark's distinct wild types is a catalogue operation over 217 files and
    has to cost one row each; the single-wildtype invariant is still enforced,
    where it belongs, by :func:`load_assay` on the assays actually scored.
    """

    root = Path(directory) if directory is not None else PROTEINGYM_ROOT
    path = root / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no ProteinGym assay at {path}")
    with path.open() as handle:
        for row in csv.DictReader(handle):
            return _revert(row["mutated_sequence"], row["mutant"])
    raise RuntimeError(f"{name}: assay file holds no rows")


def _revert(sequence: str, mutant: str) -> str:
    """The wild type implied by one variant, reverting every substitution in it."""

    out = list(sequence)
    for wild, position, mutated in parse_mutant(mutant):
        if out[position - 1] != mutated:
            raise ValueError(
                f"{mutant}: variant carries {out[position - 1]!r} at position "
                f"{position}, but the mutation string says {mutated!r}"
            )
        out[position - 1] = wild
    return "".join(out)


def load_assay(
    name: str,
    *,
    n: int,
    seed: int,
    include_multi: bool = True,
    stratify_by_score_bin: bool = False,
    train_holdout: int = 0,
    directory: Path | None = None,
) -> Assay:
    """Draw ``n`` variants of one ProteinGym substitution assay under ``seed``.

    The draw is a seeded permutation of the eligible rows, never a prefix: a
    ProteinGym CSV is ordered by position, so the first ``n`` rows are the
    protein's N-terminus rather than a sample of it (Appendix B rule 1).

    ``stratify_by_score_bin`` reproduces **ProGenMech's sampling design**: an
    equal number of variants from each ``DMS_score_bin``, drawn from a pool with
    a ``train_holdout`` split -- itself equally stratified -- removed first
    (``function_circuit/prepare_data.py:7-36``). This is not cosmetic. Their
    reported ProGen3 base of 0.29 is measured on a class-balanced resample,
    where ProteinGym's own benchmark records 0.497 for the identical checkpoint
    on six of the same assays; a recovery ratio is only readable against the
    base its own sampling produced, so both draws have to be available here.

    **The design is matched; the exact rows are not.** Their draw goes through
    ``pandas.DataFrame.sample(random_state=...)``, and pandas is deliberately
    not a dependency of this package -- the measurement harness has to run in an
    offline pod on torch, transformers, numpy and scipy alone. Reproducing their
    row selection bit-for-bit would buy a false precision anyway: the statistic
    moves by 0.013-0.035 across their own five fold seeds, so what reproduces is
    the design and its spread, not a single draw.

    Every retained variant is checked to revert to the *same* wild type. An
    assay whose rows disagree is a different protein per row, and the
    single-wildtype invariant that makes the likelihood-ratio convention
    rank-equivalent would not hold.
    """

    root = Path(directory) if directory is not None else PROTEINGYM_ROOT
    path = root / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no ProteinGym assay at {path}")

    mutants: list[str] = []
    sequences: list[str] = []
    scores: list[float] = []
    bins: list[str] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            mutant = row["mutant"]
            tokens = mutant.split(":")
            if not include_multi and len(tokens) > 1:
                continue
            if any((t[0], t[-1]) not in BLOSUM62 for t in tokens):
                continue
            mutants.append(mutant)
            sequences.append(row["mutated_sequence"])
            scores.append(float(row["DMS_score"]))
            bins.append(row["DMS_score_bin"])

    if not mutants:
        raise RuntimeError(f"{name}: no eligible variants")
    eligible = len(mutants)
    wildtype = _revert(sequences[0], mutants[0])
    for mutant, sequence in zip(mutants, sequences):
        if _revert(sequence, mutant) != wildtype:
            raise RuntimeError(
                f"{name}: variant {mutant} reverts to a different wild type, so "
                "this assay is not a single-wildtype substitution set"
            )

    rng = np.random.default_rng(seed)
    if stratify_by_score_bin:
        strata: dict[str, list[int]] = {}
        for index, label in enumerate(bins):
            strata.setdefault(label, []).append(index)
        if len(strata) < 2:
            raise RuntimeError(
                f"{name}: DMS_score_bin has {len(strata)} level(s), so a "
                "class-balanced draw is not defined on this assay"
            )
        per_stratum = n // len(strata)
        holdout_per_stratum = train_holdout // len(strata)
        picked_list: list[int] = []
        for label in sorted(strata):
            shuffled = rng.permutation(strata[label])
            # The train split is removed from the pool first, exactly as their
            # design does, so the evaluated variants are disjoint from the ones
            # a circuit-selection step would have seen.
            pool = shuffled[holdout_per_stratum:]
            picked_list.extend(pool[:per_stratum].tolist())
        picked = np.array(sorted(picked_list), dtype=int)
    else:
        picked = np.sort(rng.permutation(eligible)[: min(n, eligible)])
    return Assay(
        name=name,
        wildtype=wildtype,
        mutants=[mutants[i] for i in picked],
        sequences=[sequences[i] for i in picked],
        scores=np.array([scores[i] for i in picked], dtype=np.float64),
        n_eligible=eligible,
        seed=seed,
        sampling=(
            f"score_bin_stratified, train_holdout={train_holdout} "
            "(ProGenMech's design, not their exact rows)"
            if stratify_by_score_bin
            else "seeded_uniform_permutation"
        ),
    )
