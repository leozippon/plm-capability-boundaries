#!/usr/bin/env python3
"""Build a composition-matched, fold-disjoint retrieval cohort -- or refuse to.

**Why this exists.** Three earlier audits each found a residue-statistics surrogate
matching or beating the model, and in one of them a 3-mer frequency surrogate
out-retrieved a 7B model by 2.6x on that model's own protein mode. The common
property of all three measurements is that they were read on *agreement* sets:
cohorts where evolutionary sequence statistics and biological structure predict
the same neighbour, so a model that replayed corpus statistics and a model that
learned structure are observationally identical. No amount of further measuring
on an agreement set separates them.

This script constructs the disagreement set. Its unit is a **triple**

    (anchor, sequence partner, structure partner)

where the sequence partner is matched to the anchor on amino-acid composition --
tightly enough that composition cannot tell them apart at the lengths involved --
while carrying a *different* fold, and the structure partner carries the anchor's
own fold while being as far from it in sequence space as the pool allows. A
completion of the anchor's prefix that follows the sequence partner is a
statistics result; one that follows the structure partner is a structure result.
The triple is admitted only when the two spaces genuinely disagree about which
target is nearest, so the two hypotheses make opposite predictions on every
admitted record. That is the contradiction the impasse requires, and building it
is the whole of this script's job: **it fits nothing, runs no model, and licenses
no claim about any model.**

**What "matched" means, and why it is not an absolute tolerance.** Composition is
estimated from a finite sequence, so at length L it cannot be measured more
finely than multinomial sampling noise. Two independent draws of the *same*
composition at L = 200 differ by a total-variation distance of about 0.17. An
absolute tolerance chosen without that reference is therefore either unattainable
or vacuous, which is the failure mode this programme has twice paid for. The
criterion here is a ratio against that null (:data:`COMPOSITION_MATCH_RATIO_MAX`):
an admitted pair must be *closer in composition than two independent samples of
one composition would be at its own two lengths*. The null is analytic and its
agreement with simulation is recorded in the manifest.

**Which k-mer, and why 2 rather than 3.** At 80-400 residues a tripeptide profile
holds L-2 counts spread over 8,000 cells, so its cosine is not a frequency
comparison at all: it is dominated by whichever few 3-mers two sequences happen
to share, which makes it a low-complexity and repeat detector. Measured on the
pool, the tripeptide cosine separates same-CATH-superfamily from
different-superfamily pairs at chance -- every manifest carries the figure as a
negative control -- and choosing a partner by it gives a composition match no
better than a random pair. The dipeptide profile holds L-1 counts over 400 cells
and is an estimate. Selection therefore runs on the dipeptide profile
(:data:`SEQUENCE_SPACE_KMER`) and the tripeptide profile is carried as a second
channel whose ordering is required and reported.

**The fold statistic is contact-map-derived and is calibrated before it is used.**
The obvious choice -- a coarse two-dimensional density of the contact map in
normalised sequence coordinates -- was tried first and is useless: it separates
CATH superfamilies at AUC 0.53, because it is dominated by the diagonal band
every protein shares. What does separate them is the **contact separation
spectrum**, the fraction of contacts falling in each log-spaced band of sequence
separation, concatenated with the CA-trace secondary-structure fractions this
repository already derives; the two together reach roughly 0.8-0.87 depending on
the pool, against a k-mer negative control at chance. CATH superfamily is an
externally curated structural label and is used both to calibrate the statistic
and, independently, as an admission criterion, so no triple rests on the proxy
alone. The calibration is recomputed and written into every manifest rather than
quoted from here, and ``--foldseek`` adds a per-pair TM-align check of what the
descriptor decided.

**Leakage is measured, not assumed.** Both partners are drawn only from records
that share no near-duplicate group with the anchor (the relation and threshold of
:mod:`src.capability.generation.near_duplicates`) and that DIAMOND does not align to it at all
at ``--evalue 1e-3``. L30 is why the record is not the unit: on a protein corpus
a record-level split is not a split. The realised identity of every triple leg is
reported at five boundaries off the same all-against-all alignment, beside the
threshold-free containment distribution, so the residual is a reading rather than
a claim.

**A refusal is a result.** If no threshold in the declared sweeps yields a cohort
that clears the floors, the manifest says so, carries the curves that show why,
and the run still completes: a measured "this cohort cannot support the estimand"
is the deliverable in that case, and is worth more than a cohort that fails a
gate after burning GPU hours. The verdict field is the place to read it.

Runs on the workstation and not in a pod: DIAMOND is the offline standard this
repository calibrates its alignment-free relations against and no aligner is
staged in a pod. Pass ``--diamond`` the path to an extracted
``external/tools/diamond-linux64-*.tar.gz``.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.arms import AA20, REPO, env_path, require_input_path  # noqa: E402
from src.capability.generation.structure_inputs import ALPHAFOLD_ROOT, alphafold_models, ca_secondary_structure, read_alphafold_model
from src.capability.core.sequence_metadata import PFAM_RESIDUE_TSV
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation.near_duplicates import (  # noqa: E402
    NEAR_DUPLICATE_CONTAINMENT,
    group_disjoint_split,
    near_duplicate_groups,
    shingles,
)
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

SCHEMA_VERSION = "r2_composition_matched_fold_set_v1"

CATH_SUPERFAMILY_TSV = env_path(
    "TRANSFER_CATH_SUPERFAMILY_TSV", REPO / "data/interpro/cath_superfamily.tsv"
)
CATH_TSV_HEADER = ("uniprot", "cath_superfamily")

# --------------------------------------------------------------- declared rules
#
# Every constant below is a choice. Each carries the reason it holds the value it
# holds, and `main` reports the cohort size over a sweep of it, because a
# threshold whose attainability was never checked is how three earlier audits each
# arrived at a number that could not have meant what it was read to mean.

#: Confidence below which a residue's coordinates are not evidence. 70 is
#: AlphaFold's own boundary between "confident" and "low"; contacts involving a
#: residue under it are excluded from every contact map here, so a predicted
#: structure contributes only where its predictor was confident.
RESIDUE_PLDDT_FLOOR = 70.0

#: Whole-model rules, applied on top of the residue rule. A model whose mean
#: confidence is under 70, or four fifths of whose residues are not confident, is
#: not a structure this cohort can reason about: its contact map would be mostly
#: absent and its fold descriptor would measure the missing region.
MODEL_MEAN_PLDDT_FLOOR = 70.0
MODEL_CONFIDENT_FRACTION_FLOOR = 0.80

#: Contact definition. 8 A between CA atoms is the contact-prediction standard;
#: the separation floor removes the backbone-adjacent contacts every chain has
#: regardless of fold.
CONTACT_RADIUS_ANGSTROM = 8.0
CONTACT_MIN_SEQUENCE_SEPARATION = 3

#: A model with no confident contact has no contact map, so it has no fold
#: descriptor and cannot enter any pairing. It is excluded by the residue
#: confidence rule rather than by a threshold of its own, and the count is
#: reported rather than left to be inferred from a missing row.
MODEL_MIN_CONTACTS = 1

#: Log-spaced sequence-separation bands. The fold descriptor is the fraction of a
#: model's contacts in each band, which is length-normalised by construction and
#: is what distinguishes a local-contact fold from a long-range one.
CONTACT_SEPARATION_EDGES = (3, 5, 8, 12, 18, 27, 40, 60, 90, 135)

#: Length band of the cohort. Below 80 residues a chain is too short for its
#: contact map to express a topology; above 400 it is usually multi-domain, and a
#: multi-domain chain has no single fold for the estimand to be about.
LENGTH_BAND = (80, 400)

#: Widest band any sweep may reach. The alignment, the near-duplicate grouping and
#: the profile matrices are built once over this universe so that every sweep is a
#: subset of one measurement rather than a separate one.
LENGTH_UNIVERSE = (60, 600)

#: Two partners of a triple may differ in length by at most this factor. Length is
#: itself a sequence statistic, so a partner that is twice as long is
#: distinguishable without reading a residue.
LENGTH_RATIO_MAX = 1.25

#: Composition match, as a ratio against the sampling-noise null described in the
#: module docstring. 0.7 means the pair is at least 30% closer in composition than
#: two independent draws of one composition at those two lengths -- attainable for
#: 880 of 1,880 pool members, and roughly the first percentile of the all-pairs
#: ratio distribution, so it is tight rather than nominal.
COMPOSITION_MATCH_RATIO_MAX = 0.70

#: k for the k-mer profile that selects the sequence partner. See the module
#: docstring: at these lengths k=3 is not a frequency estimate.
SEQUENCE_SPACE_KMER = 2

#: k for the second, reported k-mer channel -- the surrogate that out-retrieved
#: the model. Its ordering is required of every admitted triple even
#: though it does not select.
REPORTED_KMER = 3

#: Fold-descriptor cosine distance above which two structures count as different
#: folds. Only 5.3% of same-CATH-superfamily pairs in the pool reach 0.15, against
#: 49.4% of different-superfamily pairs, so a pair at or above it is more
#: dissimilar than 94.7% of the pairs CATH itself calls one fold.
FOLD_DISTANCE_MIN = 0.15

#: Fold-descriptor cosine at or above which the structure partner is confirmed to
#: carry the anchor's fold. It is a second, descriptor-side check on top of the
#: CATH superfamily identity the structure partner already has to satisfy: 71.6%
#: of same-superfamily pairs reach 0.95 against 19.7% of different-superfamily
#: pairs.
FOLD_SIMILARITY_MIN = 0.95

#: DIAMOND settings. The e-value is this repository's declared setting for the
#: relation the shingle threshold was calibrated against; a partner is admissible
#: only if DIAMOND produces no alignment with the anchor at all.
HOMOLOGY_EVALUE = "1e-3"

#: Identity boundaries the leakage reading is taken at, matching the retained
#: pool leakage readings under `archive/results/shared/transfer/pool_homology_leakage/`
#: so the two are comparable. The stage that produced those readings is not
#: retained; its outputs are.
IDENTITY_BOUNDARIES = (100.0, 95.0, 90.0, 70.0, 50.0)

#: Fraction of the anchor given to the model as the prefix in the measurement this
#: cohort exists to support. Half is the largest prefix that still leaves half a
#: sequence to be completed; the prefix's own statistics are measured against both
#: partners rather than assumed uninformative.
PREFIX_FRACTION = 0.5

#: Direction margins. Admission requires only that the two spaces disagree in
#: *direction* (ratio at or above 1.0); the size of the disagreement is reported
#: as a distribution and swept, because no attainable margin was known before this
#: measurement and gating on an unmeasured margin is the defect that voided both
#: of the earlier audit's adequacy criteria.
ORDERING_MARGIN = 1.0

#: The conventional boundary above which two structures are called the same fold
#: by TM-score. It is used only by the optional external verification below: the
#: admission rules of this script are contact-map-derived and do not consult a
#: structural aligner, so the cohort is reproducible without one and the aligner
#: is the standard those rules are checked against rather than a criterion.
SAME_FOLD_TM_SCORE = 0.5

#: Circular-permutant detection. A permutant of B aligns to B concatenated with
#: itself in one high-coverage HSP that crosses the junction, while its direct
#: alignment to B breaks into two blocks whose order is reversed between query and
#: subject. Both halves must be substantial and interior, or a terminal extension
#: counts as a permutation.
CP_DUPLICATE_COVERAGE_MIN = 0.70
CP_CROSSED_COVERAGE_MIN = 0.70
CP_BLOCK_FRACTION_MIN = 0.20
CP_INTERIOR_MARGIN = 20
CP_EVALUE = "1e-5"

#: Sweeps. Each holds every other rule at its declared value.
SWEEP_COMPOSITION_MATCH_RATIO = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00)
SWEEP_FOLD_DISTANCE_MIN = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
SWEEP_FOLD_SIMILARITY_MIN = (0.90, 0.92, 0.95, 0.98, 0.99)
SWEEP_LENGTH_RATIO_MAX = (1.05, 1.10, 1.25, 1.50, 2.00)
SWEEP_MODEL_MEAN_PLDDT_FLOOR = (50.0, 60.0, 70.0, 80.0, 90.0)
SWEEP_MODEL_CONFIDENT_FRACTION_FLOOR = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90)
SWEEP_LENGTH_BAND = ((60, 600), (80, 500), (80, 400), (100, 300), (120, 250))
SWEEP_ORDERING_MARGIN = (1.0, 1.1, 1.25, 1.5, 2.0)
SWEEP_RESIDUE_PLDDT_FLOOR = (50.0, 70.0, 90.0)

AA_INDEX = {residue: position for position, residue in enumerate(AA20)}


# ------------------------------------------------------------------- utilities


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    """The distribution of a quantity, never a bare mean.

    Returned for every distance this script reports, because "the achieved match
    quality -- a distribution, not a claim" is the only form in which a match
    quality can be checked by a reader.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    finite = np.isfinite(array)
    return {
        "n": int(array.size),
        "n_non_finite": int((~finite).sum()),
        "min": float(array[finite].min()) if finite.any() else None,
        "p5": float(np.percentile(array[finite], 5)) if finite.any() else None,
        "p25": float(np.percentile(array[finite], 25)) if finite.any() else None,
        "p50": float(np.percentile(array[finite], 50)) if finite.any() else None,
        "p75": float(np.percentile(array[finite], 75)) if finite.any() else None,
        "p95": float(np.percentile(array[finite], 95)) if finite.any() else None,
        "max": float(array[finite].max()) if finite.any() else None,
        "mean": float(array[finite].mean()) if finite.any() else None,
    }


def load_cath_superfamilies(path: Path) -> dict[str, set[str]]:
    """CATH superfamily assignments keyed on UniProt accession."""

    require_input_path(Path(path), "TRANSFER_CATH_SUPERFAMILY_TSV")
    assignments: dict[str, set[str]] = defaultdict(set)
    with Path(path).open(encoding="utf-8") as handle:
        header = tuple(next(handle).rstrip("\n").split("\t"))
        if header != CATH_TSV_HEADER:
            raise ValueError(f"{path}: expected columns {CATH_TSV_HEADER}, found {header}")
        for number, line in enumerate(handle, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{number}: expected 2 columns, found {len(fields)}")
            assignments[fields[0]].add(fields[1])
    if not assignments:
        raise RuntimeError(f"{path}: no CATH superfamily assignments were read")
    return dict(assignments)


def load_repeated_pfam_accessions(path: Path) -> set[str]:
    """Accessions carrying the same Pfam domain more than once.

    The circular-permutant detector's one confounder. A protein with n copies of
    one domain aligns to its own tandem duplicate across the junction and to any
    other member of its repeat family with crossed blocks, which is the
    permutation signature produced by a repeat shift rather than by a permutation.
    Reading the repeat off curated Pfam spans rather than off a self-alignment is
    deliberate: DIAMOND reports one HSP per pair by default and its self-hit is
    the diagonal, so a self-alignment screen silently returns nothing.
    """

    require_input_path(Path(path), "TRANSFER_PFAM_RESIDUE_TSV")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            accession, _, _, pfam = line.rstrip("\n").split("\t")
            counts[accession][pfam] += 1
    return {
        accession
        for accession, domains in counts.items()
        if domains and max(domains.values()) > 1
    }


def single_fragment_models(root: Path) -> dict[str, Path]:
    """AlphaFold PDB models of accessions covered by exactly one fragment.

    A multi-fragment accession is a chain AlphaFold could not model in one piece,
    so no fragment of it is the protein and the fold of the whole is not on disk.
    """

    fragments: dict[str, set[str]] = defaultdict(set)
    paths: dict[str, Path] = {}
    for path in alphafold_models(root):
        parts = path.name.split("-")
        fragments[parts[1]].add(parts[2])
        if parts[2] == "F1":
            paths[parts[1]] = path
    return {
        accession: path
        for accession, path in sorted(paths.items())
        if fragments[accession] == {"F1"}
    }


# ------------------------------------------------------- structural extraction


def _sequence_only(path: Path) -> tuple[str, str]:
    structure = read_alphafold_model(path)
    return structure.accession, structure.sequence


def _features(path: Path, *, residue_plddt_floor: float) -> dict[str, Any]:
    """Composition, confidence and the contact-map fold descriptor of one model."""

    structure = read_alphafold_model(path)
    length = len(structure)
    confident = structure.plddt >= residue_plddt_floor
    composition = np.zeros(len(AA20), dtype=np.float64)
    for residue in structure.sequence:
        composition[AA_INDEX[residue]] += 1.0

    rows, columns = np.triu_indices(length, CONTACT_MIN_SEQUENCE_SEPARATION)
    distance = np.linalg.norm(
        structure.ca[rows] - structure.ca[columns], axis=-1
    )
    in_contact = (
        (distance < CONTACT_RADIUS_ANGSTROM) & confident[rows] & confident[columns]
    )
    separation = (columns - rows)[in_contact]

    edges = np.asarray(CONTACT_SEPARATION_EDGES + (length + 1,), dtype=np.int64)
    bands = np.zeros(len(CONTACT_SEPARATION_EDGES), dtype=np.float64)
    if separation.size:
        for position in range(len(CONTACT_SEPARATION_EDGES)):
            lower, upper = edges[position], edges[position + 1]
            bands[position] = np.count_nonzero(
                (separation >= lower) & (separation < upper)
            ) / separation.size

    assignment = ca_secondary_structure(structure.ca)[confident]
    secondary = np.zeros(3, dtype=np.float64)
    if assignment.size:
        for state in range(3):
            secondary[state] = float(np.count_nonzero(assignment == state) / assignment.size)

    return {
        "accession": structure.accession,
        "sequence": structure.sequence,
        "length": length,
        "mean_plddt": float(structure.plddt.mean()),
        "confident_fraction": float(confident.mean()),
        "composition": composition,
        "bands": bands,
        "secondary": secondary,
        "n_contacts": int(separation.size),
        "relative_contact_order": (
            float(separation.mean() / length) if separation.size else float("nan")
        ),
    }


def read_features(
    paths: Sequence[Path], *, residue_plddt_floor: float, processes: int
) -> list[dict[str, Any]]:
    worker = functools.partial(_features, residue_plddt_floor=residue_plddt_floor)
    with Pool(processes) as pool:
        return pool.map(worker, list(paths), chunksize=16)


# -------------------------------------------------------------- profile spaces


def kmer_matrix(sequences: Sequence[str], k: int) -> np.ndarray:
    """L2-normalised k-mer count profiles, one row per sequence."""

    if k < 1:
        raise ValueError("k must be positive")
    width = len(AA20) ** k
    matrix = np.zeros((len(sequences), width), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for start in range(len(sequence) - k + 1):
            index = 0
            for residue in sequence[start : start + k]:
                index = index * len(AA20) + AA_INDEX[residue]
            matrix[row, index] += 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise RuntimeError("a sequence shorter than k carries no k-mer profile")
    return matrix / norms


def composition_distances(
    frequencies: np.ndarray, lengths: np.ndarray, *, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Observed total-variation distance and its sampling-noise null, pairwise.

    The null is the expected total-variation distance between two *independent*
    multinomial draws of the two observed lengths from the pair's pooled
    composition. Under that null each coordinate difference is asymptotically
    centred normal with variance ``p(1-p)(1/L_i + 1/L_j)``, so its mean absolute
    value is ``sqrt(2/pi)`` times its standard deviation and the expected
    total-variation distance is half their sum. `main` records the agreement
    between this expression and a direct simulation on sampled pairs, because a
    null this criterion rests on may not be asserted in prose.
    """

    n = frequencies.shape[0]
    observed = np.zeros((n, n), dtype=np.float32)
    expected = np.zeros((n, n), dtype=np.float32)
    inverse_length = 1.0 / lengths.astype(np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = frequencies[start:stop, None, :] - frequencies[None, :, :]
        observed[start:stop] = (0.5 * np.abs(block).sum(-1)).astype(np.float32)
        pooled = 0.5 * (frequencies[start:stop, None, :] + frequencies[None, :, :])
        scale = inverse_length[start:stop, None] + inverse_length[None, :]
        sigma = np.sqrt(pooled * (1.0 - pooled) * scale[:, :, None])
        expected[start:stop] = (
            0.5 * np.sqrt(2.0 / np.pi) * sigma.sum(-1)
        ).astype(np.float32)
    return observed, expected


def simulate_composition_null(
    frequencies: np.ndarray,
    lengths: np.ndarray,
    expected: np.ndarray,
    *,
    seed: int,
    n_pairs: int,
    n_draws: int,
) -> dict[str, Any]:
    """Direct simulation of the analytic null on seeded pairs."""

    rng = np.random.default_rng(seed)
    n = frequencies.shape[0]
    left = rng.integers(0, n, n_pairs)
    right = rng.integers(0, n, n_pairs)
    ratios = []
    for i, j in zip(left, right):
        if i == j:
            continue
        pooled = 0.5 * (frequencies[i] + frequencies[j])
        first = rng.multinomial(int(lengths[i]), pooled, size=n_draws) / lengths[i]
        second = rng.multinomial(int(lengths[j]), pooled, size=n_draws) / lengths[j]
        simulated = float(np.mean(0.5 * np.abs(first - second).sum(1)))
        ratios.append(simulated / float(expected[i, j]))
    if not ratios:
        raise RuntimeError("the composition-null simulation drew no distinct pair")
    return {
        "statistic": "simulated expected total variation divided by the analytic value",
        "n_pairs": len(ratios),
        "n_draws_per_pair": int(n_draws),
        "ratio": quantiles(ratios),
    }


def fold_descriptor(
    bands: np.ndarray, secondary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalised concatenation of the contact separation spectrum and CA-trace
    secondary-structure fractions, and which rows carry one at all.

    Cosine on these rows is the fold similarity every criterion in this script
    reads. A model whose confident residues support no contact and no secondary
    structure has an all-zero row; it is returned as invalid rather than
    normalised into a direction it does not have, and the caller excludes it.
    """

    joint = np.concatenate([bands, secondary], axis=1)
    norms = np.linalg.norm(joint, axis=1, keepdims=True)
    valid = norms[:, 0] > 0.0
    return (joint / np.maximum(norms, 1e-12)).astype(np.float32), valid


# -------------------------------------------------------------------- alignment


def write_fasta(path: Path, sequences: Sequence[str], *, duplicate: bool = False) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences):
            body = sequence + sequence if duplicate else sequence
            handle.write(f">r{index}\n{body}\n")


def diamond_blastp(
    diamond: Path,
    query: Path,
    subject: Path,
    work: Path,
    *,
    threads: int,
    evalue: str,
    columns: Sequence[str],
    tag: str,
    max_hsps: int | None = None,
) -> Path:
    """One DIAMOND all-against-all, at this repository's declared settings."""

    database = work / f"{tag}.dmnd"
    hits = work / f"{tag}_hits.tsv"
    subprocess.run(
        [str(diamond), "makedb", "--in", str(subject), "--db", str(database), "--quiet"],
        check=True,
    )
    command = [
        str(diamond), "blastp",
        "--query", str(query),
        "--db", str(database),
        "--out", str(hits),
        "--very-sensitive",
        "--masking", "0",
        "--evalue", evalue,
        "--threads", str(threads),
        "--max-target-seqs", "0",
        "--outfmt", "6", *columns,
        "--quiet",
    ]
    if max_hsps is not None:
        command += ["--max-hsps", str(max_hsps)]
    subprocess.run(command, check=True)
    return hits


def homology_relation(hits: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric "DIAMOND aligned these two" mask and identity over the shorter."""

    aligned = np.zeros((n, n), dtype=bool)
    identity = np.zeros((n, n), dtype=np.float32)
    with hits.open(encoding="utf-8") as handle:
        for line in handle:
            query, subject, nident, qlen, slen = line.split()
            i, j = int(query[1:]), int(subject[1:])
            aligned[i, j] = aligned[j, i] = True
            value = 100.0 * int(nident) / min(int(qlen), int(slen))
            identity[i, j] = max(identity[i, j], value)
            identity[j, i] = max(identity[j, i], value)
    return aligned, identity


# -------------------------------------------------------- selection and admission


def select_and_admit(
    *,
    keep: np.ndarray,
    same_fold_family: np.ndarray,
    length_ratio: np.ndarray,
    allowed: np.ndarray,
    composition_ratio: np.ndarray,
    composition_distance: np.ndarray,
    select_kmer: np.ndarray,
    report_kmer: np.ndarray,
    fold_similarity: np.ndarray,
    composition_match_ratio_max: float,
    fold_distance_min: float,
    fold_similarity_min: float,
    length_ratio_max: float,
    ordering_margin: float,
) -> dict[str, Any]:
    """Choose both partners for every anchor and apply the admission rules.

    The sequence partner is the *most k-mer-similar* record that is CATH-disjoint,
    length-matched, composition-matched and structurally different; the structure
    partner is the *least k-mer-similar* record that shares the anchor's CATH
    superfamily and is confirmed to share its fold by the descriptor. Both
    selections run only over records the anchor has no near-duplicate or alignment
    relation with. Selecting the structure partner away from sequence space is the
    construction, not a convenience: the estimand needs the structural neighbour
    to be one that sequence statistics would *not* propose.

    Returned counts are cumulative over the criteria in the order they are applied,
    so a sweep can be read as a curve of where the population is lost.
    """

    n = keep.size
    if int(keep.sum()) < 2:
        empty = np.zeros(n, dtype=bool)
        zeros = np.zeros(n, dtype=np.int64)
        return {
            "n_pool": int(keep.sum()),
            "n_admitted": 0,
            "attrition": {"pool": int(keep.sum()), "pool_too_small_to_pair": True},
            "admitted_mask": empty,
            "partner_sequence": zeros,
            "partner_structure": zeros,
            "global_composition_neighbour": zeros,
            "global_structure_neighbour": zeros,
            "global_kmer_neighbour": zeros,
            "margins": {
                "composition": np.zeros(n),
                f"kmer{SEQUENCE_SPACE_KMER}": np.zeros(n),
                f"kmer{REPORTED_KMER}": np.zeros(n),
                "fold": np.zeros(n),
            },
        }

    ratio_max = float(composition_match_ratio_max)
    fold_distance = 1.0 - fold_similarity
    in_pool = keep[:, None] & keep[None, :]

    sequence_candidate = (
        allowed
        & ~same_fold_family
        & in_pool
        & (length_ratio <= length_ratio_max)
        & (composition_ratio <= ratio_max)
        & (fold_distance >= fold_distance_min)
    )
    structure_candidate = (
        allowed
        & same_fold_family
        & in_pool
        & (fold_similarity >= fold_similarity_min)
    )

    has_sequence = sequence_candidate.any(axis=1)
    has_structure = structure_candidate.any(axis=1)
    partner_sequence = np.where(sequence_candidate, select_kmer, -2.0).argmax(axis=1)
    partner_structure = np.where(structure_candidate, select_kmer, 2.0).argmin(axis=1)

    # The neighbour-divergence check the estimand rests on, taken over the whole
    # allowed set rather than over the selected candidates: if one record is both
    # the composition-nearest and the structure-nearest, the two spaces agree
    # about this anchor and no completion of it can distinguish them.
    allowed_or = allowed & in_pool
    global_composition = np.where(allowed_or, -composition_distance, -9.0).argmax(axis=1)
    global_structure = np.where(allowed_or, fold_similarity, -2.0).argmax(axis=1)
    global_kmer = np.where(allowed_or, report_kmer, -2.0).argmax(axis=1)
    diverge = (
        allowed_or.any(axis=1)
        & (global_composition != global_structure)
        & (global_kmer != global_structure)
    )

    rows = np.arange(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        composition_margin = (
            composition_distance[rows, partner_structure]
            / np.maximum(composition_distance[rows, partner_sequence], 1e-12)
        )
        select_margin = (
            select_kmer[rows, partner_sequence]
            / np.maximum(select_kmer[rows, partner_structure], 1e-12)
        )
        report_margin = (
            report_kmer[rows, partner_sequence]
            / np.maximum(report_kmer[rows, partner_structure], 1e-12)
        )
        fold_margin = (
            fold_distance[rows, partner_sequence]
            / np.maximum(fold_distance[rows, partner_structure], 1e-12)
        )

    both = keep & has_sequence & has_structure
    distinct = both & (partner_sequence != partner_structure)
    ordered_composition = distinct & (composition_margin >= ordering_margin)
    ordered_select = ordered_composition & (select_margin >= ordering_margin)
    ordered_report = ordered_select & (report_margin >= ordering_margin)
    ordered_fold = ordered_report & (fold_margin >= ordering_margin)
    admitted = ordered_fold & diverge

    return {
        "n_pool": int(keep.sum()),
        "n_admitted": int(admitted.sum()),
        "attrition": {
            "pool": int(keep.sum()),
            "with_sequence_partner": int((keep & has_sequence).sum()),
            "with_structure_partner": int((keep & has_structure).sum()),
            "with_both_partners": int(both.sum()),
            "partners_distinct": int(distinct.sum()),
            "composition_ordering_holds": int(ordered_composition.sum()),
            f"kmer{SEQUENCE_SPACE_KMER}_ordering_holds": int(ordered_select.sum()),
            f"kmer{REPORTED_KMER}_ordering_holds": int(ordered_report.sum()),
            "fold_ordering_holds": int(ordered_fold.sum()),
            "global_neighbours_diverge": int(admitted.sum()),
            "n_excluded_by_neighbour_coincidence": int((ordered_fold & ~diverge).sum()),
        },
        "admitted_mask": admitted,
        "partner_sequence": partner_sequence,
        "partner_structure": partner_structure,
        "global_composition_neighbour": global_composition,
        "global_structure_neighbour": global_structure,
        "global_kmer_neighbour": global_kmer,
        "margins": {
            "composition": composition_margin,
            f"kmer{SEQUENCE_SPACE_KMER}": select_margin,
            f"kmer{REPORTED_KMER}": report_margin,
            "fold": fold_margin,
        },
    }


# ------------------------------------------- external structural verification


def tm_score_matrix(
    foldseek: Path,
    work: Path,
    *,
    paths: Sequence[Path],
    threads: int,
) -> np.ndarray:
    """All-against-all TM-align TM-scores over a set of AlphaFold models.

    The fold criteria this script admits on are read off contact maps, which is
    what makes them reproducible from the assets alone. That is also their weak
    point: a contact-separation spectrum is a fold *descriptor*, not a structural
    alignment. This runs the alignment, on the admitted members only, so the two
    can be compared pair by pair instead of on aggregate. The score kept is the
    smaller of the two length-normalised TM-scores -- normalisation by the longer
    chain -- because the question asked of it is whether the *whole* of one
    structure matches the other, and normalising by the shorter chain would let a
    small domain match a large protein containing something like it.

    A pair foldseek does not align at all is recorded as zero, which is the
    correct reading for "these two structures are not superposable" and is
    counted separately so a reader can see how much of the matrix it is.
    """

    structures = work / "structures"
    if structures.exists():
        shutil.rmtree(structures)
    structures.mkdir(parents=True)
    for position, path in enumerate(paths):
        shutil.copy(path, structures / f"m{position}{''.join(Path(path).suffixes)}")
    hits = work / "tmalign_hits.tsv"
    scratch = work / "foldseek_tmp"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    subprocess.run(
        [
            str(foldseek), "easy-search", str(structures), str(structures),
            str(hits), str(scratch),
            "--alignment-type", "1",
            "--tmscore-threshold", "0.0",
            "-e", "100",
            "--max-seqs", str(max(len(paths), 1000)),
            "--threads", str(threads),
            "--format-output", "query,target,qtmscore,ttmscore",
            "-v", "1",
        ],
        check=True,
    )
    n = len(paths)
    matrix = np.zeros((n, n), dtype=np.float32)
    with hits.open(encoding="utf-8") as handle:
        for line in handle:
            query, target, qtm, ttm = line.split()
            i = int(Path(query).name.split(".")[0][1:])
            j = int(Path(target).name.split(".")[0][1:])
            value = min(float(qtm), float(ttm))
            if value > matrix[i, j]:
                matrix[i, j] = value
                matrix[j, i] = max(matrix[j, i], value)
    np.fill_diagonal(matrix, 1.0)
    return matrix


# ------------------------------------------------------ circular-permutant census


def circular_permutant_census(
    diamond: Path,
    work: Path,
    *,
    accessions: Sequence[str],
    sequences: Sequence[str],
    repeated_pfam: set[str],
    threads: int,
) -> dict[str, Any]:
    """How many natural circular permutants the AlphaFold assets on disk contain.

    A circular permutant of B has B's composition exactly and B's k-mer multiset
    up to two junctions, while its contact *order* differs -- the tightest
    composition-matched, fold-different pair that exists. Detection is the
    standard tandem-duplication test, run in both required directions and then
    screened for the one thing that mimics it.
    """

    direct_columns = (
        "qseqid", "sseqid", "pident", "qstart", "qend", "sstart", "send", "qlen", "slen"
    )
    plain = work / "cp_pool.fasta"
    duplicated = work / "cp_pool_duplicated.fasta"
    write_fasta(plain, sequences)
    write_fasta(duplicated, sequences, duplicate=True)

    duplicate_hits = diamond_blastp(
        diamond, plain, duplicated, work,
        threads=threads, evalue=CP_EVALUE, columns=direct_columns, tag="cp_duplicate",
    )
    direct_hits = diamond_blastp(
        diamond, plain, plain, work,
        threads=threads, evalue=CP_EVALUE, columns=direct_columns, tag="cp_direct",
        max_hsps=0,
    )

    lengths = np.array([len(sequence) for sequence in sequences], dtype=np.int64)
    crossing: dict[tuple[int, int], tuple[float, float]] = {}
    with duplicate_hits.open(encoding="utf-8") as handle:
        for line in handle:
            q, s, pident, qstart, qend, sstart, send, qlen, _ = line.split()
            i, j = int(q[1:]), int(s[1:])
            if i == j:
                continue
            start, end = int(sstart), int(send)
            if not start <= lengths[j] < end:
                continue
            coverage = (int(qend) - int(qstart) + 1) / int(qlen)
            if coverage > crossing.get((i, j), (0.0, 0.0))[0]:
                crossing[(i, j)] = (coverage, float(pident))

    blocks: dict[tuple[int, int], list[tuple[int, int, int, int, float, int, int]]] = (
        defaultdict(list)
    )
    with direct_hits.open(encoding="utf-8") as handle:
        for line in handle:
            q, s, pident, qstart, qend, sstart, send, qlen, slen = line.split()
            i, j = int(q[1:]), int(s[1:])
            if i == j:
                continue
            blocks[(i, j)].append(
                (int(qstart), int(qend), int(sstart), int(send), float(pident),
                 int(qlen), int(slen))
            )

    candidates: list[dict[str, Any]] = []
    for (i, j), hsps in blocks.items():
        if len(hsps) < 2 or (i, j) not in crossing:
            continue
        best = None
        for first in hsps:
            for second in hsps:
                if first is second:
                    continue
                qs1, qe1, ss1, se1, id1, qlen, slen = first
                qs2, qe2, ss2, se2, id2, _, _ = second
                if not (qe1 < qs2 and se2 < ss1):
                    continue
                left, right = qe1 - qs1 + 1, qe2 - qs2 + 1
                if min(left, right) < CP_BLOCK_FRACTION_MIN * qlen:
                    continue
                if min(ss1, qs2) < CP_INTERIOR_MARGIN:
                    continue
                if slen - se2 < CP_INTERIOR_MARGIN:
                    continue
                coverage = (left + right) / qlen
                if best is None or coverage > best[0]:
                    best = (coverage, (left * id1 + right * id2) / (left + right))
        if best is None or best[0] < CP_CROSSED_COVERAGE_MIN:
            continue
        duplicate_coverage, duplicate_identity = crossing[(i, j)]
        if duplicate_coverage < CP_DUPLICATE_COVERAGE_MIN:
            continue
        candidates.append({
            "query": accessions[i],
            "subject": accessions[j],
            "query_length": int(lengths[i]),
            "subject_length": int(lengths[j]),
            "crossed_block_coverage": float(best[0]),
            "crossed_block_identity": float(best[1]),
            "duplicate_junction_coverage": float(duplicate_coverage),
            "duplicate_junction_identity": float(duplicate_identity),
            "query_has_repeated_pfam_domain": accessions[i] in repeated_pfam,
            "subject_has_repeated_pfam_domain": accessions[j] in repeated_pfam,
        })

    surviving = [
        candidate for candidate in candidates
        if not candidate["query_has_repeated_pfam_domain"]
        and not candidate["subject_has_repeated_pfam_domain"]
    ]
    unordered = {
        frozenset((candidate["query"], candidate["subject"])) for candidate in surviving
    }
    return {
        "universe": {
            "n_models": len(sequences),
            "description": "every single-fragment AlphaFold model on disk",
            "n_with_repeated_pfam_domain": int(
                sum(1 for accession in accessions if accession in repeated_pfam)
            ),
        },
        "criteria": {
            "duplicate_junction_query_coverage_min": CP_DUPLICATE_COVERAGE_MIN,
            "crossed_block_total_coverage_min": CP_CROSSED_COVERAGE_MIN,
            "crossed_block_fraction_min": CP_BLOCK_FRACTION_MIN,
            "interior_margin_residues": CP_INTERIOR_MARGIN,
            "diamond_evalue": CP_EVALUE,
        },
        "n_junction_crossing_ordered_pairs": len(crossing),
        "n_candidate_ordered_pairs": len(candidates),
        "n_after_repeated_pfam_screen": len(surviving),
        "n_unordered_pairs": len(unordered),
        "candidates": candidates,
        "note": (
            "a candidate whose query or subject carries the same Pfam domain more "
            "than once is a repeat shift and not a permutation: n copies of one "
            "domain produce a junction-crossing alignment against the tandem "
            "duplicate and crossed blocks against any other member of the family"
        ),
    }


# ------------------------------------------------------------------------ output


def triple_records(
    result: dict[str, Any],
    *,
    accessions: np.ndarray,
    sequences: Sequence[str],
    superfamily: np.ndarray,
    lengths: np.ndarray,
    composition: np.ndarray,
    composition_distance: np.ndarray,
    composition_expected: np.ndarray,
    select_kmer: np.ndarray,
    report_kmer: np.ndarray,
    fold_similarity: np.ndarray,
    relative_contact_order: np.ndarray,
    n_contacts: np.ndarray,
    mean_plddt: np.ndarray,
    confident_fraction: np.ndarray,
    identity: np.ndarray,
    groups: np.ndarray,
    shingle_sets: Sequence[frozenset[str]],
) -> list[dict[str, Any]]:
    """One JSONL record per admitted triple, carrying every distance it was
    admitted on plus the prefix the downstream measurement reads."""

    frequencies = composition / composition.sum(axis=1, keepdims=True)
    records: list[dict[str, Any]] = []
    partner_sequence = result["partner_sequence"]
    partner_structure = result["partner_structure"]
    for anchor in np.flatnonzero(result["admitted_mask"]):
        s_index = int(partner_sequence[anchor])
        t_index = int(partner_structure[anchor])
        prefix_length = max(1, int(round(PREFIX_FRACTION * lengths[anchor])))
        prefix = sequences[anchor][:prefix_length]
        prefix_composition = np.zeros(len(AA20), dtype=np.float64)
        for residue in prefix:
            prefix_composition[AA_INDEX[residue]] += 1.0
        prefix_composition /= prefix_length
        prefix_kmers = {
            prefix[start : start + REPORTED_KMER]
            for start in range(len(prefix) - REPORTED_KMER + 1)
        }

        def shared_kmer_share(other: int) -> float:
            other_kmers = {
                sequences[other][start : start + REPORTED_KMER]
                for start in range(len(sequences[other]) - REPORTED_KMER + 1)
            }
            return len(prefix_kmers & other_kmers) / len(prefix_kmers) if prefix_kmers else 0.0

        def containment(other: int) -> float:
            smaller = min(len(shingle_sets[anchor]), len(shingle_sets[other]))
            if not smaller:
                return 0.0
            return len(shingle_sets[anchor] & shingle_sets[other]) / smaller

        records.append({
            "anchor": str(accessions[anchor]),
            "sequence_partner": str(accessions[s_index]),
            "structure_partner": str(accessions[t_index]),
            "anchor_sequence": sequences[anchor],
            "sequence_partner_sequence": sequences[s_index],
            "structure_partner_sequence": sequences[t_index],
            "anchor_prefix": prefix,
            "prefix_fraction": PREFIX_FRACTION,
            "length": {
                "anchor": int(lengths[anchor]),
                "sequence_partner": int(lengths[s_index]),
                "structure_partner": int(lengths[t_index]),
                "ratio_anchor_sequence_partner": float(
                    max(lengths[anchor] / lengths[s_index], lengths[s_index] / lengths[anchor])
                ),
                "ratio_anchor_structure_partner": float(
                    max(lengths[anchor] / lengths[t_index], lengths[t_index] / lengths[anchor])
                ),
            },
            "cath": {
                "anchor": str(superfamily[anchor]),
                "sequence_partner": str(superfamily[s_index]),
                "structure_partner": str(superfamily[t_index]),
                "class_differs_anchor_to_sequence_partner": bool(
                    str(superfamily[anchor]).split(".")[0]
                    != str(superfamily[s_index]).split(".")[0]
                ),
            },
            "composition": {
                "total_variation_anchor_sequence_partner": float(
                    composition_distance[anchor, s_index]
                ),
                "total_variation_anchor_structure_partner": float(
                    composition_distance[anchor, t_index]
                ),
                "expected_under_identical_composition_sequence_partner": float(
                    composition_expected[anchor, s_index]
                ),
                "match_ratio_sequence_partner": float(
                    composition_distance[anchor, s_index]
                    / composition_expected[anchor, s_index]
                ),
                "match_ratio_structure_partner": float(
                    composition_distance[anchor, t_index]
                    / composition_expected[anchor, t_index]
                ),
            },
            "kmer": {
                f"k{SEQUENCE_SPACE_KMER}_cosine_sequence_partner": float(
                    select_kmer[anchor, s_index]
                ),
                f"k{SEQUENCE_SPACE_KMER}_cosine_structure_partner": float(
                    select_kmer[anchor, t_index]
                ),
                f"k{REPORTED_KMER}_cosine_sequence_partner": float(
                    report_kmer[anchor, s_index]
                ),
                f"k{REPORTED_KMER}_cosine_structure_partner": float(
                    report_kmer[anchor, t_index]
                ),
            },
            "fold": {
                "distance_anchor_sequence_partner": float(
                    1.0 - fold_similarity[anchor, s_index]
                ),
                "similarity_anchor_structure_partner": float(
                    fold_similarity[anchor, t_index]
                ),
                "relative_contact_order": {
                    "anchor": float(relative_contact_order[anchor]),
                    "sequence_partner": float(relative_contact_order[s_index]),
                    "structure_partner": float(relative_contact_order[t_index]),
                },
                "n_contacts": {
                    "anchor": int(n_contacts[anchor]),
                    "sequence_partner": int(n_contacts[s_index]),
                    "structure_partner": int(n_contacts[t_index]),
                },
            },
            "prefix_statistics": {
                "composition_total_variation_to_sequence_partner": float(
                    0.5 * np.abs(prefix_composition - frequencies[s_index]).sum()
                ),
                "composition_total_variation_to_structure_partner": float(
                    0.5 * np.abs(prefix_composition - frequencies[t_index]).sum()
                ),
                f"share_of_prefix_{REPORTED_KMER}mers_present_in_sequence_partner": (
                    shared_kmer_share(s_index)
                ),
                f"share_of_prefix_{REPORTED_KMER}mers_present_in_structure_partner": (
                    shared_kmer_share(t_index)
                ),
            },
            "confidence": {
                "mean_plddt": {
                    "anchor": float(mean_plddt[anchor]),
                    "sequence_partner": float(mean_plddt[s_index]),
                    "structure_partner": float(mean_plddt[t_index]),
                },
                "confident_residue_fraction": {
                    "anchor": float(confident_fraction[anchor]),
                    "sequence_partner": float(confident_fraction[s_index]),
                    "structure_partner": float(confident_fraction[t_index]),
                },
            },
            "leakage": {
                "diamond_identity_anchor_sequence_partner": float(identity[anchor, s_index]),
                "diamond_identity_anchor_structure_partner": float(identity[anchor, t_index]),
                "diamond_identity_between_partners": float(identity[s_index, t_index]),
                "shingle_containment_anchor_sequence_partner": containment(s_index),
                "shingle_containment_anchor_structure_partner": containment(t_index),
            },
            "near_duplicate_group": {
                "anchor": int(groups[anchor]),
                "sequence_partner": int(groups[s_index]),
                "structure_partner": int(groups[t_index]),
            },
        })
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_compact_json(record) + "\n")
    temporary.replace(path)
    return sha256_file(path)


def _compact_json(record: dict[str, Any]) -> str:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    )


# ---------------------------------------------------------------------- driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphafold-dir", type=Path, default=ALPHAFOLD_ROOT)
    parser.add_argument("--cath-tsv", type=Path, default=CATH_SUPERFAMILY_TSV)
    parser.add_argument("--pfam-tsv", type=Path, default=PFAM_RESIDUE_TSV)
    parser.add_argument("--diamond", type=Path, required=True,
                        help="path to an extracted DIAMOND binary")
    parser.add_argument("--foldseek", type=Path, default=None,
                        help="path to an extracted foldseek binary. Optional, and "
                             "it changes no admission rule: it runs TM-align over "
                             "the admitted members so that the contact-map fold "
                             "criteria can be checked pair by pair against a "
                             "structural alignment")
    parser.add_argument("--work", type=Path, required=True,
                        help="scratch directory for FASTA, databases and hit tables; "
                             "kept outside the repository")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--processes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--null-simulation-pairs", type=int, default=64)
    parser.add_argument("--null-simulation-draws", type=int, default=400)
    parser.add_argument(
        "--max-models", type=int, default=None,
        help="truncate the sorted model list. AlphaFold filenames are in accession "
             "order, so a truncation is a taxonomically clustered prefix rather "
             "than a sample: a smoke-test knob, never a reported cohort",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    # ---- the universe of structures ------------------------------------------
    models = single_fragment_models(args.alphafold_dir)
    if args.max_models is not None:
        models = dict(sorted(models.items())[: args.max_models])
    print(f"[models] {len(models)} single-fragment AlphaFold models")

    cath = load_cath_superfamilies(args.cath_tsv)
    single_family = {
        accession: path
        for accession, path in models.items()
        if len(cath.get(accession, ())) == 1
    }
    print(f"[cath] {len(single_family)} carry exactly one CATH superfamily")
    if not single_family:
        raise RuntimeError(
            "no AlphaFold model on disk carries exactly one CATH superfamily; the "
            "fold label this cohort is built on does not exist for this asset set"
        )

    features = read_features(
        list(single_family.values()),
        residue_plddt_floor=RESIDUE_PLDDT_FLOOR,
        processes=args.processes,
    )
    lengths_all = np.array([f["length"] for f in features], dtype=np.int64)
    contacts_all = np.array([f["n_contacts"] for f in features], dtype=np.int64)
    in_band = (lengths_all >= LENGTH_UNIVERSE[0]) & (lengths_all <= LENGTH_UNIVERSE[1])
    has_contacts = contacts_all >= MODEL_MIN_CONTACTS
    in_universe = in_band & has_contacts
    n_no_contacts = int((in_band & ~has_contacts).sum())
    universe_paths = [
        path for path, take in zip(single_family.values(), in_universe) if take
    ]
    features = [f for f, take in zip(features, in_universe) if take]
    if len(features) < 2:
        raise RuntimeError(
            f"only {len(features)} models fall in the length universe "
            f"{LENGTH_UNIVERSE} with at least {MODEL_MIN_CONTACTS} confident "
            "contact; no pairing is possible"
        )
    print(
        f"[universe] {len(features)} models in the length universe {LENGTH_UNIVERSE}; "
        f"{n_no_contacts} dropped for carrying no confident contact"
    )

    accessions = np.array([f["accession"] for f in features])
    sequences = [f["sequence"] for f in features]
    lengths = np.array([f["length"] for f in features], dtype=np.int64)
    mean_plddt = np.array([f["mean_plddt"] for f in features])
    confident_fraction = np.array([f["confident_fraction"] for f in features])
    composition = np.stack([f["composition"] for f in features])
    bands = np.stack([f["bands"] for f in features])
    secondary = np.stack([f["secondary"] for f in features])
    n_contacts = np.array([f["n_contacts"] for f in features], dtype=np.int64)
    relative_contact_order = np.array([f["relative_contact_order"] for f in features])
    superfamily = np.array([sorted(cath[accession])[0] for accession in accessions])
    _, superfamily_code = np.unique(superfamily, return_inverse=True)
    same_fold_family = superfamily_code[:, None] == superfamily_code[None, :]
    length_ratio = np.maximum(
        lengths[:, None] / lengths[None, :], lengths[None, :] / lengths[:, None]
    ).astype(np.float32)
    n = len(features)

    # ---- profile spaces -------------------------------------------------------
    frequencies = composition / composition.sum(axis=1, keepdims=True)
    composition_distance, composition_expected = composition_distances(frequencies, lengths)
    composition_ratio = composition_distance / composition_expected
    null_check = simulate_composition_null(
        frequencies, lengths, composition_expected,
        seed=args.seed, n_pairs=args.null_simulation_pairs,
        n_draws=args.null_simulation_draws,
    )
    print(
        "[null] simulated/analytic expected composition distance, median "
        f"{null_check['ratio']['p50']:.4f}"
    )

    select_profile = kmer_matrix(sequences, SEQUENCE_SPACE_KMER)
    report_profile = kmer_matrix(sequences, REPORTED_KMER)
    select_kmer = (select_profile @ select_profile.T).astype(np.float32)
    report_kmer = (report_profile @ report_profile.T).astype(np.float32)
    descriptor, descriptor_valid = fold_descriptor(bands, secondary)
    if not descriptor_valid.all():
        raise RuntimeError(
            f"{int((~descriptor_valid).sum())} universe models carry an all-zero fold "
            "descriptor despite holding a confident contact; the descriptor and the "
            "contact map disagree about what was measured"
        )
    fold_similarity = (descriptor @ descriptor.T).astype(np.float32)

    # ---- independence relations ----------------------------------------------
    groups, grouping = near_duplicate_groups(sequences, unit="residues")
    print(f"[groups] {grouping['n_groups']} near-duplicate groups over {n} records")
    fasta = args.work / "universe.fasta"
    write_fasta(fasta, sequences)
    hits = diamond_blastp(
        args.diamond, fasta, fasta, args.work,
        threads=args.threads, evalue=HOMOLOGY_EVALUE,
        columns=("qseqid", "sseqid", "nident", "qlen", "slen"), tag="universe",
    )
    aligned, identity = homology_relation(hits, n)
    same_group = groups[:, None] == groups[None, :]
    allowed = ~aligned & ~same_group
    np.fill_diagonal(allowed, False)
    print(
        f"[homology] DIAMOND aligns {int(aligned.sum() - n)} of {n * n - n} ordered "
        f"pairs; {float(allowed.mean()):.4f} of pairs are admissible partners"
    )

    # ---- calibration of the fold statistic -----------------------------------
    upper = np.triu_indices(n, 1)
    admissible = allowed[upper]
    same_family_pair = same_fold_family[upper] & admissible
    other_family_pair = ~same_fold_family[upper] & admissible

    def descriptor_auc(matrix: np.ndarray, valid: np.ndarray | None = None) -> float:
        """Separation of the CATH superfamily label by one similarity matrix."""

        same, other = same_family_pair, other_family_pair
        if valid is not None:
            usable = valid[upper[0]] & valid[upper[1]]
            same, other = same & usable, other & usable
        if not same.any() or not other.any():
            raise RuntimeError(
                "the fold statistic cannot be calibrated: one side of the CATH "
                "contrast has no admissible pair"
            )
        values = matrix[upper]
        labels = np.concatenate([np.ones(int(same.sum())), np.zeros(int(other.sum()))])
        return float(
            roc_auc_score(labels, np.concatenate([values[same], values[other]]))
        )

    def cosine(block: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        unit = block / np.maximum(norms, 1e-12)
        return (unit @ unit.T).astype(np.float32)

    calibration = {
        "standard": "CATH superfamily identity, over admissible pairs only",
        "n_same_superfamily_pairs": int(same_family_pair.sum()),
        "n_different_superfamily_pairs": int(other_family_pair.sum()),
        "auc": {
            "contact_separation_spectrum": descriptor_auc(cosine(bands)),
            "ca_secondary_structure_fractions": descriptor_auc(cosine(secondary + 1e-9)),
            "fold_descriptor": descriptor_auc(fold_similarity),
            f"kmer{REPORTED_KMER}_cosine_negative_control": descriptor_auc(report_kmer),
            f"kmer{SEQUENCE_SPACE_KMER}_cosine_negative_control": descriptor_auc(select_kmer),
        },
        "fold_distance_same_superfamily": quantiles(
            (1.0 - fold_similarity)[upper][same_family_pair]
        ),
        "fold_distance_different_superfamily": quantiles(
            (1.0 - fold_similarity)[upper][other_family_pair]
        ),
        "note": (
            "the k-mer rows are the negative control: a sequence statistic must not "
            "predict the structural label, and if it does the fold criterion is "
            "measuring homology"
        ),
    }
    print(f"[calibration] fold descriptor AUC {calibration['auc']['fold_descriptor']:.4f}")

    # ---- the cohort ----------------------------------------------------------
    def pool_mask(
        band: tuple[int, int], mean_floor: float, confident_floor: float
    ) -> np.ndarray:
        return (
            (lengths >= band[0]) & (lengths <= band[1])
            & (mean_plddt >= mean_floor)
            & (confident_fraction >= confident_floor)
        )

    keep = pool_mask(LENGTH_BAND, MODEL_MEAN_PLDDT_FLOOR, MODEL_CONFIDENT_FRACTION_FLOOR)
    print(f"[pool] {int(keep.sum())} models satisfy the declared pool rules")
    if int(keep.sum()) < 2:
        raise RuntimeError(
            f"only {int(keep.sum())} models satisfy the declared pool rules; no "
            "pairing is possible and no threshold here may be widened to change that"
        )

    selection_inputs = dict(
        same_fold_family=same_fold_family,
        length_ratio=length_ratio,
        allowed=allowed,
        composition_ratio=composition_ratio,
        composition_distance=composition_distance,
        select_kmer=select_kmer,
        report_kmer=report_kmer,
        fold_similarity=fold_similarity,
    )
    declared = dict(
        composition_match_ratio_max=COMPOSITION_MATCH_RATIO_MAX,
        fold_distance_min=FOLD_DISTANCE_MIN,
        fold_similarity_min=FOLD_SIMILARITY_MIN,
        length_ratio_max=LENGTH_RATIO_MAX,
        ordering_margin=ORDERING_MARGIN,
    )
    result = select_and_admit(keep=keep, **selection_inputs, **declared)
    print(f"[cohort] {result['n_admitted']} admitted triples")

    # ---- sensitivity ---------------------------------------------------------
    def sweep(name: str, values: Sequence[Any]) -> list[dict[str, Any]]:
        curve = []
        for value in values:
            if name == "length_band":
                mask = pool_mask(
                    tuple(value), MODEL_MEAN_PLDDT_FLOOR, MODEL_CONFIDENT_FRACTION_FLOOR
                )
                run = select_and_admit(keep=mask, **selection_inputs, **declared)
            elif name == "model_mean_plddt_floor":
                mask = pool_mask(LENGTH_BAND, float(value), MODEL_CONFIDENT_FRACTION_FLOOR)
                run = select_and_admit(keep=mask, **selection_inputs, **declared)
            elif name == "model_confident_fraction_floor":
                mask = pool_mask(LENGTH_BAND, MODEL_MEAN_PLDDT_FLOOR, float(value))
                run = select_and_admit(keep=mask, **selection_inputs, **declared)
            else:
                overrides = dict(declared)
                overrides[name] = float(value)
                run = select_and_admit(keep=keep, **selection_inputs, **overrides)
            curve.append({
                "value": list(value) if isinstance(value, tuple) else float(value),
                "n_pool": run["n_pool"],
                "n_admitted": run["n_admitted"],
                "attrition": run["attrition"],
            })
        return curve

    sensitivity = {
        "composition_match_ratio_max": sweep(
            "composition_match_ratio_max", SWEEP_COMPOSITION_MATCH_RATIO
        ),
        "fold_distance_min": sweep("fold_distance_min", SWEEP_FOLD_DISTANCE_MIN),
        "fold_similarity_min": sweep("fold_similarity_min", SWEEP_FOLD_SIMILARITY_MIN),
        "length_ratio_max": sweep("length_ratio_max", SWEEP_LENGTH_RATIO_MAX),
        "ordering_margin": sweep("ordering_margin", SWEEP_ORDERING_MARGIN),
        "length_band": sweep("length_band", SWEEP_LENGTH_BAND),
        "model_mean_plddt_floor": sweep(
            "model_mean_plddt_floor", SWEEP_MODEL_MEAN_PLDDT_FLOOR
        ),
        "model_confident_fraction_floor": sweep(
            "model_confident_fraction_floor", SWEEP_MODEL_CONFIDENT_FRACTION_FLOOR
        ),
    }

    # The residue floor changes the contact map itself, so its sweep re-reads every
    # structure rather than re-thresholding a matrix. Everything else is held.
    residue_curve = []
    for floor in SWEEP_RESIDUE_PLDDT_FLOOR:
        if floor == RESIDUE_PLDDT_FLOOR:
            residue_curve.append({
                "value": float(floor),
                "n_admitted": result["n_admitted"],
                "fold_descriptor_auc": calibration["auc"]["fold_descriptor"],
            })
            continue
        swept = read_features(
            list(single_family.values()),
            residue_plddt_floor=float(floor),
            processes=args.processes,
        )
        swept = [f for f, take in zip(swept, in_universe) if take]
        swept_descriptor, swept_valid = fold_descriptor(
            np.stack([f["bands"] for f in swept]), np.stack([f["secondary"] for f in swept])
        )
        swept_similarity = (swept_descriptor @ swept_descriptor.T).astype(np.float32)
        swept_confident = np.array([f["confident_fraction"] for f in swept])
        swept_contacts = np.array([f["n_contacts"] for f in swept], dtype=np.int64)
        swept_keep = (
            (lengths >= LENGTH_BAND[0]) & (lengths <= LENGTH_BAND[1])
            & (mean_plddt >= MODEL_MEAN_PLDDT_FLOOR)
            & (swept_confident >= MODEL_CONFIDENT_FRACTION_FLOOR)
            & (swept_contacts >= MODEL_MIN_CONTACTS)
            & swept_valid
        )
        swept_inputs = dict(selection_inputs)
        swept_inputs["fold_similarity"] = swept_similarity
        run = select_and_admit(keep=swept_keep, **swept_inputs, **declared)
        residue_curve.append({
            "value": float(floor),
            "n_pool": run["n_pool"],
            "n_admitted": run["n_admitted"],
            "n_models_without_a_descriptor": int((~swept_valid).sum()),
            "fold_descriptor_auc": descriptor_auc(swept_similarity, valid=swept_valid),
        })
    sensitivity["residue_plddt_floor"] = residue_curve

    # ---- realised distributions and leakage ----------------------------------
    admitted = np.flatnonzero(result["admitted_mask"])
    partner_sequence = result["partner_sequence"]
    partner_structure = result["partner_structure"]
    members = sorted(
        set(admitted.tolist())
        | {int(partner_sequence[i]) for i in admitted}
        | {int(partner_structure[i]) for i in admitted}
    )
    shingle_sets = [shingles(sequence, unit="residues") for sequence in sequences]

    records = triple_records(
        result,
        accessions=accessions, sequences=sequences, superfamily=superfamily,
        lengths=lengths, composition=composition,
        composition_distance=composition_distance,
        composition_expected=composition_expected,
        select_kmer=select_kmer, report_kmer=report_kmer,
        fold_similarity=fold_similarity,
        relative_contact_order=relative_contact_order, n_contacts=n_contacts,
        mean_plddt=mean_plddt, confident_fraction=confident_fraction,
        identity=identity, groups=groups, shingle_sets=shingle_sets,
    )

    verification: dict[str, Any] = {
        "verdict": "NOT_RUN",
        "reason": (
            "no --foldseek binary was given, so the contact-map fold criteria are "
            "supported by their CATH calibration alone and no per-pair structural "
            "alignment was taken"
        ),
    }
    if args.foldseek is not None and records:
        ordinal = {index: position for position, index in enumerate(members)}
        tm = tm_score_matrix(
            args.foldseek, args.work,
            paths=[universe_paths[index] for index in members],
            threads=args.threads,
        )
        sequence_leg = np.array([
            tm[ordinal[int(i)], ordinal[int(partner_sequence[i])]] for i in admitted
        ])
        structure_leg = np.array([
            tm[ordinal[int(i)], ordinal[int(partner_structure[i])]] for i in admitted
        ])
        for record, to_sequence, to_structure in zip(records, sequence_leg, structure_leg):
            record["fold"]["tm_score_anchor_sequence_partner"] = float(to_sequence)
            record["fold"]["tm_score_anchor_structure_partner"] = float(to_structure)
            record["fold"]["tm_score_ordering_holds"] = bool(to_structure > to_sequence)
            record["fold"]["structure_partner_at_or_above_same_fold_tm"] = bool(
                to_structure >= SAME_FOLD_TM_SCORE
            )
        member_upper = np.triu_indices(len(members), 1)
        member_same = (
            superfamily_code[np.array(members)][:, None]
            == superfamily_code[np.array(members)][None, :]
        )[member_upper]
        member_descriptor = fold_similarity[np.ix_(members, members)][member_upper]
        member_tm = tm[member_upper]
        verification = {
            "verdict": "VERIFIED_AGAINST_TM_ALIGN",
            "tool": "foldseek easy-search --alignment-type 1 (TM-align)",
            "score": (
                "the smaller of the two length-normalised TM-scores, i.e. "
                "normalisation by the longer chain; an unaligned pair reads zero"
            ),
            "same_fold_boundary": SAME_FOLD_TM_SCORE,
            "n_members_aligned": len(members),
            "n_member_pairs_unaligned": int((member_tm == 0.0).sum()),
            "descriptor_agreement": {
                "spearman_descriptor_vs_tm": float(
                    spearmanr(member_descriptor, member_tm).statistic
                ),
                "auc_descriptor_predicts_same_superfamily": float(
                    roc_auc_score(member_same, member_descriptor)
                ),
                "auc_tm_predicts_same_superfamily": float(
                    roc_auc_score(member_same, member_tm)
                ),
            },
            "tm_score_anchor_sequence_partner": quantiles(sequence_leg),
            "tm_score_anchor_structure_partner": quantiles(structure_leg),
            "n_sequence_partners_at_or_above_same_fold_tm": int(
                (sequence_leg >= SAME_FOLD_TM_SCORE).sum()
            ),
            "n_structure_partners_at_or_above_same_fold_tm": int(
                (structure_leg >= SAME_FOLD_TM_SCORE).sum()
            ),
            "n_triples_with_tm_ordering": int((structure_leg > sequence_leg).sum()),
            "n_triples_surviving_an_added_tm_criterion": int(
                (
                    (structure_leg >= SAME_FOLD_TM_SCORE)
                    & (sequence_leg < SAME_FOLD_TM_SCORE)
                ).sum()
            ),
            "note": (
                "this changes no admission rule. It is reported so that a "
                "downstream stage can take the stricter TM-verified subset from the "
                "per-record fields without rebuilding the cohort, and so that the "
                "share of structure partners a structural aligner does not place "
                "inside the conventional same-fold boundary is visible rather than "
                "inferred from an AUC"
            ),
        }
        print(
            f"[tm-align] structure partners at TM>={SAME_FOLD_TM_SCORE}: "
            f"{verification['n_structure_partners_at_or_above_same_fold_tm']}/"
            f"{len(records)}; sequence partners: "
            f"{verification['n_sequence_partners_at_or_above_same_fold_tm']}/{len(records)}"
        )

    leg_identity = np.array(
        [identity[i, partner_sequence[i]] for i in admitted]
        + [identity[i, partner_structure[i]] for i in admitted]
        + [identity[partner_sequence[i], partner_structure[i]] for i in admitted]
    )
    leg_containment = np.array([
        record["leakage"][key]
        for record in records
        for key in (
            "shingle_containment_anchor_sequence_partner",
            "shingle_containment_anchor_structure_partner",
        )
    ])
    leakage = {
        "relation": (
            "both partners are drawn only from records DIAMOND does not align to the "
            "anchor at --evalue " + HOMOLOGY_EVALUE + " and that share no "
            "near-duplicate group with it; L30 is why the near-duplicate group "
            "rather than the record is the unit"
        ),
        "n_triple_legs": int(leg_identity.size),
        "identity_at_or_above": {
            f"{boundary:g}": {
                "n": int((leg_identity >= boundary).sum()),
                "fraction": float((leg_identity >= boundary).mean()) if leg_identity.size else 0.0,
            }
            for boundary in IDENTITY_BOUNDARIES
        },
        "identity": quantiles(leg_identity),
        "shingle_containment": quantiles(leg_containment),
        "n_legs_at_or_above_near_duplicate_containment": int(
            (leg_containment >= NEAR_DUPLICATE_CONTAINMENT).sum()
        ),
        "unscreened_reference": {
            "statistic": (
                "per admitted anchor, its maximum DIAMOND identity over the shorter "
                "sequence against ANY other universe record, screened or not. It is "
                "what the anchor's closest relative in this asset set looks like, "
                "and the contrast with the screened rows above is the measurement -- "
                "a near-zero screened row means nothing without it"
            ),
            "max_identity_to_any_universe_record": quantiles([
                float(np.max(np.where(
                    np.arange(n) != i, identity[i], 0.0
                ))) for i in admitted
            ]),
        },
    }

    # ---- the split the downstream measurement resamples over ------------------
    if len(members) >= 2 * MINIMUM_BOOTSTRAP_UNITS:
        member_groups = groups[members]
        _, compacted = np.unique(member_groups, return_inverse=True)
        train, split_summary = group_disjoint_split(
            compacted, n_train=len(members) // 2, seed=args.seed
        )
    else:
        split_summary = {
            "verdict": "NOT_ATTEMPTED",
            "reason": (
                f"{len(members)} cohort members is below the {2 * MINIMUM_BOOTSTRAP_UNITS} "
                "needed for two sides at the bootstrap-unit floor"
            ),
        }

    distinct_groups = int(np.unique(groups[members]).size) if members else 0
    clears_floor = (
        split_summary.get("n_train_groups", 0) >= MINIMUM_BOOTSTRAP_UNITS
        and split_summary.get("n_eval_groups", 0) >= MINIMUM_BOOTSTRAP_UNITS
    )

    # ---- the circular-permutant census ---------------------------------------
    all_paths = list(models.values())
    with Pool(args.processes) as pool:
        all_sequences = pool.map(_sequence_only, all_paths, chunksize=16)
    repeated_pfam = load_repeated_pfam_accessions(args.pfam_tsv)
    census = circular_permutant_census(
        args.diamond, args.work,
        accessions=[accession for accession, _ in all_sequences],
        sequences=[sequence for _, sequence in all_sequences],
        repeated_pfam=repeated_pfam,
        threads=args.threads,
    )
    print(
        f"[permutants] {census['n_candidate_ordered_pairs']} candidates, "
        f"{census['n_unordered_pairs']} surviving the repeated-domain screen"
    )

    # ---- artefacts -----------------------------------------------------------
    jsonl = args.out / "composition_matched_fold_set.jsonl"
    digest = write_jsonl(jsonl, records)

    raw: dict[str, Any] = {}
    for name in ("universe_hits.tsv", "universe.fasta"):
        source = args.work / name
        destination = args.out / f"{name}.gz"
        with source.open("rb") as reader, gzip.open(destination, "wb") as writer:
            shutil.copyfileobj(reader, writer)
        raw[name] = {"path": destination.name, "sha256": sha256_file(destination)}

    def leg(key: str, field: str) -> list[float]:
        return [record[key][field] for record in records]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "declared_rules": {
            "residue_plddt_floor": RESIDUE_PLDDT_FLOOR,
            "model_mean_plddt_floor": MODEL_MEAN_PLDDT_FLOOR,
            "model_confident_fraction_floor": MODEL_CONFIDENT_FRACTION_FLOOR,
            "contact_radius_angstrom": CONTACT_RADIUS_ANGSTROM,
            "contact_min_sequence_separation": CONTACT_MIN_SEQUENCE_SEPARATION,
            "contact_separation_edges": list(CONTACT_SEPARATION_EDGES),
            "length_band": list(LENGTH_BAND),
            "length_universe": list(LENGTH_UNIVERSE),
            "length_ratio_max": LENGTH_RATIO_MAX,
            "composition_match_ratio_max": COMPOSITION_MATCH_RATIO_MAX,
            "sequence_space_kmer": SEQUENCE_SPACE_KMER,
            "reported_kmer": REPORTED_KMER,
            "fold_distance_min": FOLD_DISTANCE_MIN,
            "fold_similarity_min": FOLD_SIMILARITY_MIN,
            "homology_evalue": HOMOLOGY_EVALUE,
            "near_duplicate_containment": NEAR_DUPLICATE_CONTAINMENT,
            "ordering_margin": ORDERING_MARGIN,
            "prefix_fraction": PREFIX_FRACTION,
            "minimum_bootstrap_units": MINIMUM_BOOTSTRAP_UNITS,
        },
        "inputs": {
            "alphafold_dir": str(args.alphafold_dir),
            "cath_tsv": str(args.cath_tsv),
            "cath_tsv_sha256": sha256_file(Path(args.cath_tsv)),
            "pfam_tsv": str(args.pfam_tsv),
            "pfam_tsv_sha256": sha256_file(Path(args.pfam_tsv)),
            "n_single_fragment_models": len(models),
            "n_with_one_cath_superfamily": len(single_family),
            "n_in_length_universe": n,
            "n_dropped_for_no_confident_contact": n_no_contacts,
        },
        "pool": {
            "n_pool": int(keep.sum()),
            "n_distinct_superfamilies": int(np.unique(superfamily[keep]).size),
            "cath_class_counts": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(s.split(".")[0] for s in superfamily[keep]).items()
                )
            },
            "length": quantiles(lengths[keep]),
            "mean_plddt": quantiles(mean_plddt[keep]),
            "confident_residue_fraction": quantiles(confident_fraction[keep]),
            "n_contacts": quantiles(n_contacts[keep]),
            "relative_contact_order": quantiles(relative_contact_order[keep]),
            "near_duplicate_grouping": grouping,
        },
        "composition_null": null_check,
        "fold_descriptor_calibration": calibration,
        "cohort": {
            "n_admitted_triples": len(records),
            "n_distinct_members": len(members),
            "n_distinct_near_duplicate_groups": distinct_groups,
            "attrition": result["attrition"],
            "group_disjoint_split": split_summary,
            "clears_bootstrap_unit_floor_per_side": bool(clears_floor),
            "cath_class_differs_share": (
                float(np.mean([
                    record["cath"]["class_differs_anchor_to_sequence_partner"]
                    for record in records
                ])) if records else None
            ),
        },
        "distributions": {
            "composition_total_variation_to_sequence_partner": quantiles(
                leg("composition", "total_variation_anchor_sequence_partner")
            ),
            "composition_total_variation_to_structure_partner": quantiles(
                leg("composition", "total_variation_anchor_structure_partner")
            ),
            "composition_match_ratio_to_sequence_partner": quantiles(
                leg("composition", "match_ratio_sequence_partner")
            ),
            "composition_match_ratio_to_structure_partner": quantiles(
                leg("composition", "match_ratio_structure_partner")
            ),
            f"kmer{SEQUENCE_SPACE_KMER}_cosine_to_sequence_partner": quantiles(
                leg("kmer", f"k{SEQUENCE_SPACE_KMER}_cosine_sequence_partner")
            ),
            f"kmer{SEQUENCE_SPACE_KMER}_cosine_to_structure_partner": quantiles(
                leg("kmer", f"k{SEQUENCE_SPACE_KMER}_cosine_structure_partner")
            ),
            f"kmer{REPORTED_KMER}_cosine_to_sequence_partner": quantiles(
                leg("kmer", f"k{REPORTED_KMER}_cosine_sequence_partner")
            ),
            f"kmer{REPORTED_KMER}_cosine_to_structure_partner": quantiles(
                leg("kmer", f"k{REPORTED_KMER}_cosine_structure_partner")
            ),
            "fold_distance_to_sequence_partner": quantiles(
                leg("fold", "distance_anchor_sequence_partner")
            ),
            "fold_similarity_to_structure_partner": quantiles(
                leg("fold", "similarity_anchor_structure_partner")
            ),
            "length_ratio_to_sequence_partner": quantiles(
                leg("length", "ratio_anchor_sequence_partner")
            ),
            "ordering_margins": {
                "statistic": (
                    "per admitted triple, how many times further the structure "
                    "partner is in each sequence space than the sequence partner, "
                    "and how many times further the sequence partner is in fold "
                    "space. A denominator of exactly zero -- a structure partner "
                    "sharing no tripeptide with the anchor -- is floored at 1e-12, "
                    "so the upper tail of the k3 row is a division by that floor "
                    "and not a measured ratio"
                ),
                **{
                    name: quantiles(values[result["admitted_mask"]])
                    for name, values in result["margins"].items()
                },
            },
            "prefix": {
                "composition_total_variation_to_sequence_partner": quantiles(
                    leg("prefix_statistics",
                        "composition_total_variation_to_sequence_partner")
                ),
                "composition_total_variation_to_structure_partner": quantiles(
                    leg("prefix_statistics",
                        "composition_total_variation_to_structure_partner")
                ),
                f"share_of_prefix_{REPORTED_KMER}mers_in_sequence_partner": quantiles(
                    leg("prefix_statistics",
                        f"share_of_prefix_{REPORTED_KMER}mers_present_in_sequence_partner")
                ),
                f"share_of_prefix_{REPORTED_KMER}mers_in_structure_partner": quantiles(
                    leg("prefix_statistics",
                        f"share_of_prefix_{REPORTED_KMER}mers_present_in_structure_partner")
                ),
                "share_whose_prefix_still_favours_the_sequence_partner": (
                    float(np.mean([
                        record["prefix_statistics"][
                            "composition_total_variation_to_sequence_partner"
                        ] < record["prefix_statistics"][
                            "composition_total_variation_to_structure_partner"
                        ]
                        for record in records
                    ])) if records else None
                ),
            },
        },
        "leakage": leakage,
        "structural_alignment_verification": verification,
        "threshold_sensitivity": sensitivity,
        "circular_permutant_census": census,
        "artefacts": {
            "cohort_jsonl": {"path": jsonl.name, "sha256": digest, "n_records": len(records)},
            **raw,
        },
        "verdict": "COHORT_ADMITTED" if (records and clears_floor) else "COHORT_REFUSED",
        "limitations": [
            "The fold label is CATH superfamily and the fold statistic is a "
            "coordinate descriptor; neither is a structural alignment. The "
            "descriptor's separation of CATH superfamilies is reported as an AUC "
            "rather than assumed, and no triple rests on the descriptor alone "
            "because CATH disjointness is required independently.",
            "Structures are AlphaFold predictions. Residues below the declared "
            "confidence floor contribute no contacts and whole models below the "
            "model-level floors are excluded, but a confident prediction is still a "
            "prediction and not an experimental structure.",
            "The secondary-structure channel of the fold descriptor is the "
            "distance-only P-SEA approximation this repository already carries; its "
            "state fractions are not DSSP content and are used only as a "
            "coordinate-derived attribute.",
            "Composition is matched against a finite-length sampling null, so "
            "'matched' means 'closer than two independent samples of one "
            "composition at these lengths' and not 'identical'. The realised ratio "
            "distribution is reported and no tolerance was widened after seeing it.",
            "k-mer content is matched only in ordering. The tripeptide profile at "
            "these lengths is too sparse to be a frequency estimate, so the "
            "absolute share of a prefix's tripeptides present in either partner is "
            "small for both; the contrast this cohort supports is which partner a "
            "sequence statistic prefers, not that the partners share most of their "
            "k-mers.",
            "The structure partner is selected to be the *least* k-mer-similar "
            "record that carries the anchor's fold, which sharpens the contrast the "
            "estimand reads and also selects the most divergent member of the "
            "superfamily. Under TM-align a substantial minority of structure "
            "partners therefore falls below the conventional same-fold boundary of "
            "0.5, while sequence partners fall below it essentially always. The "
            "counts are in the structural-alignment verification block when a "
            "foldseek binary is given, and the per-record fields let a downstream "
            "stage take the TM-verified subset instead.",
            "The residual near-duplicate and homology leakage is measured on the "
            "realised triples rather than assumed absent, but DIAMOND cannot see "
            "what it does not align: a relationship below its detection limit is "
            "outside this measurement.",
            "The cohort is drawn from whatever proteomes the AlphaFold assets on "
            "disk cover, which the manifest's taxonomy-free counts do not record. "
            "A single-proteome asset set bounds both the circular-permutant census "
            "and the diversity of the cohort, and that bound travels with any "
            "result read off it.",
        ],
    }
    write_json(args.out / "composition_matched_fold_set_manifest.json", payload)

    print(f"[verdict] {payload['verdict']}")
    print(
        f"  {len(records)} triples, {len(members)} members, {distinct_groups} "
        f"near-duplicate groups, split "
        f"{split_summary.get('n_train_groups')}/{split_summary.get('n_eval_groups')} "
        f"groups against a floor of {MINIMUM_BOOTSTRAP_UNITS} per side"
    )
    print(f"[done] wrote {args.out / 'composition_matched_fold_set_manifest.json'}")


if __name__ == "__main__":
    main()
