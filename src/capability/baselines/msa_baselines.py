"""Homologous-support statistical baselines for the R1 capability comparison.

Two rungs are exported per mutant, both keyed by the frozen R1 ``sample_id``:

``profile_delta``
    The LOOKUP log ratio of the existing site-independent profile,
    ``sum_p log[f(m,p) + a*b(m)] - log[f(w,p) + a*b(w)]`` with the retained
    ``profiles.npz`` frequencies, the corpus background and ``a = PSEUDOCOUNT_ALPHA``.
    It is recomputed here rather than re-declared, and every value is checked
    against the frozen R5 cohort's own ``profile_scores`` before it is written.

``potts_delta``
    ``E(mutant) - E(wild type)`` under a plmc Potts model fitted on that wild
    type's own homologous support, where
    ``E(x) = sum_i h_i(x_i) + sum_{i<j} J_ij(x_i, x_j)``. The difference is taken
    on the *complete* sequence, so a multiple mutant carries every pair coupling
    among its changed sites and between those sites and the rest of the chain;
    single-mutation scores are never summed as a substitute.

Direction. Both rungs are log-odds in favour of the mutant: a higher value means
the model prefers the mutant to the wild type, which is the LOOKUP convention
this repository already publishes.

Support. Homolog rows, weights, coverage and identity floors are the ones the
retained profile store was built with (80% query coverage, highest bit score per
subject, at most 5000 sequences, 80% identity reweighting). They are rebuilt
through :func:`~..core.sequence_alignment.alignment_rows` and then verified
column by column against ``profiles.npz`` for every wild type before any fit.

The query row carries weight zero. It fixes the modelled columns and contributes
no evidence, so the Potts fit sees exactly the weights the profile saw. A real
plmc run in the accompanying tests shows that changing the zero-weight focus's
residues leaves the fitted parameters bit-identical, and every fit record checks
plmc's own reported effective sample count against the homolog weight sum.

Regularization is chosen on held-out homolog sequences only: the gap-ignoring
conditional log likelihood of held-out rows, which is the objective plmc itself
optimizes. No phenotype, no measured mutation and no model output is read
anywhere in this module.

Convergence. plmc's ``-m`` is a cap, not a target, so a run that stops at it has
not reached plmc's own gradient criterion. Every fit therefore escalates the cap
in the declared order (200, 1000, 5000) on the identical alignment, weights and
regularization, and each attempt is kept in the record. A value chosen from a
likelihood evaluated where the optimizer stopped would not be a cross-validated
choice, so selection requires the whole grid to have converged: one unconverged
or failed grid value refuses the background instead of being dropped from the
comparison. A background whose support is too thin for the held-out split is
fitted at the pinned external example value ``lambda_E = 16`` and labelled
``fixed_low_depth``; it is kept, not deleted, and it does not claim cross
validation. A background is scored only when a delivered model exists, and it is
complete only when every fit behind that model converged.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import subprocess

import numpy as np

from ..context.profiles import (
    AA20,
    GAP_CODE,
    NEFF_IDENTITY_FLOOR,
    PROFILE_COVERAGE_FLOOR,
    PSEUDOCOUNT_ALPHA,
    REWEIGHT_IDENTITY_FLOOR,
    Profile,
    lookup_score,
    sequence_weights,
    substitution_codes,
)
from ..core.sequence_alignment import alignment_rows, verify_rows_against_profile
from ..models.fitness import parse_mutant

#: Schema tags. One per artefact, so a reader can refuse a document it does not
#: know rather than guessing at its columns.
SCHEMA_MANIFEST = "r1_msa_baseline_manifest_v1"
SCHEMA_EXECUTION = "r1_msa_potts_execution_manifest_v1"
SCHEMA_SUPPORT = "r1_msa_support_v1"
SCHEMA_SCORES = "r1_msa_baseline_scores_v1"

#: Support protocol. Restates the settled values of the retained profile store
#: and the archived pairwise fit; ``tests/baselines/test_msa_baselines.py`` holds
#: the restated objective and split helpers against the archived implementation.
MAX_SEQUENCES = 5000
COVERAGE_FLOOR = PROFILE_COVERAGE_FLOOR
REWEIGHT_IDENTITY = REWEIGHT_IDENTITY_FLOOR
NEFF_IDENTITY = NEFF_IDENTITY_FLOOR
#: The query row locates columns and carries no weight (see the module docstring).
FOCUS_WEIGHT = 0.0

#: Regularization protocol, fixed before any new phenotype result is read.
LAMBDA_E_GRID: tuple[float, ...] = (0.16, 0.8, 4.0, 16.0, 80.0)
LAMBDA_H = 0.01
#: plmc's ``-m`` is a cap, not a target: a run stopped by it has not met plmc's
#: own gradient criterion. One fit therefore escalates the cap until the run
#: converges, and records every attempt it took to get there.
MAX_ITER_CAPS: tuple[int, ...] = (200, 1000, 5000)
MAX_ITER = MAX_ITER_CAPS[0]
#: plmc's own example value, used verbatim for backgrounds too thin to hold out.
FIXED_LOW_DEPTH_LAMBDA_E = 16.0
HELD_OUT_FRACTION = 0.2
SPLIT_SEED = 20260924
MIN_TRAIN_ROWS = 20
MIN_HELD_OUT_ROWS = 5
MIN_PRESENT_PER_ROW = 2
#: Independent-site pseudocount of the selection objective only; it is not the
#: profile channel's pseudocount and the two must never be confused.
INDEPENDENT_SITE_PSEUDOCOUNT = 0.5

#: A rebuilt alignment must reduce to the retained float32 frequencies this
#: closely. It is a float32 serialization tolerance, not a modelling tolerance.
PROFILE_FREQUENCY_TOLERANCE = 1e-6

CONVERGED_STATUS = "Minimization success"
CAP_HIT_MARK = "MAXIMUMITERATION"

POLICY_CROSS_VALIDATED = "cv_selected"
POLICY_FIXED_LOW_DEPTH = "fixed_low_depth"

STATUS_FITTED = "fitted"
#: Every fit converged, but the one delivered model did not: the scores exist and
#: are reported, and the background is not complete.
STATUS_FINAL_NOT_CONVERGED = "final_not_converged_after_caps"
#: At least one regularization never converged, so no value could be selected. No
#: final model is fitted and the background carries no score.
STATUS_CV_NOT_CONVERGED = "cv_not_converged_after_caps"
STATUS_PLMC_FAILED = "plmc_failed"
STATUS_ALL_REGULARIZATIONS_FAILED = "all_regularizations_failed"
STATUS_NO_SUPPORT = "no_homolog_support"
STATUS_NOT_FITTED = "not_fitted"
STATUS_BACKGROUND_FAILED = "background_failed"
#: One regularization's own outcome, before selection.
STATUS_ATTEMPT_CONVERGED = "converged"
STATUS_ATTEMPT_UNCONVERGED = "unconverged_after_caps"
#: A background is scoreable when a delivered model exists; it is complete only
#: when that model, and every fit behind its selection, converged.
SCOREABLE = (STATUS_FITTED, STATUS_FINAL_NOT_CONVERGED)
COMPLETE = (STATUS_FITTED,)

PROFILE_SCORE_DEFINITION = (
    "LOOKUP log ratio sum_p log[f(m,p)+alpha*b(m)] - log[f(w,p)+alpha*b(w)] over the "
    "retained profiles.npz column frequencies, corpus amino-acid background and "
    f"alpha=PSEUDOCOUNT_ALPHA={PSEUDOCOUNT_ALPHA}; unchanged from the repository's own "
    "profile channel"
)
POTTS_SCORE_DEFINITION = (
    "E(mutant)-E(wild type) with E=sum_i h_i(x_i)+sum_{i<j} J_ij(x_i,x_j) over the "
    "complete sequence, all pair couplings included; params from plmc -g with the query "
    "row at weight zero; higher favours the mutant"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name, parts = None, []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts)))
                name, parts = line[1:].strip(), []
            elif line.strip():
                parts.append(line.strip())
    if name is not None:
        records.append((name, "".join(parts)))
    return records


def encode_sequence(sequence: str) -> np.ndarray:
    """Residue codes of one aligned row, the gap character included as a gap."""

    table = np.full(256, GAP_CODE, dtype=np.int64)
    for code, residue in enumerate(AA20):
        table[ord(residue)] = code
    return table[np.frombuffer(sequence.encode(), dtype=np.uint8)]


# ---------------------------------------------------------------- the variants


def parse_substitutions(mutation: str, wildtype: str) -> tuple[tuple[str, int, str], ...]:
    """Validated ``(wild residue, 1-based position, mutant residue)`` substitutions.

    The grammar comes from :func:`~..models.fitness.parse_mutant`, the package's
    single declaration of it. What is added here is the part a baseline needs and
    the grammar does not promise: the position lies inside this wild type, the
    stated wild residue is the residue that wild type carries, no position repeats
    inside one variant, and nothing is called a substitution without changing a
    residue.
    """

    substitutions = parse_mutant(mutation)
    seen: set[int] = set()
    for wild, position, mutated in substitutions:
        if wild not in AA20 or mutated not in AA20:
            raise ValueError(f"{mutation!r} leaves the canonical amino-acid alphabet")
        if mutated == wild:
            raise ValueError(f"{mutation!r} calls {wild}{position}{mutated} a substitution")
        if not 1 <= position <= len(wildtype):
            raise ValueError(
                f"{mutation!r}: position {position} is outside a {len(wildtype)}-residue wild type")
        if wildtype[position - 1] != wild:
            raise ValueError(
                f"{mutation!r}: mutation string says {wild!r} at position {position}, the wild "
                f"type carries {wildtype[position - 1]!r}")
        if position in seen:
            raise ValueError(f"{mutation!r} substitutes position {position} twice")
        seen.add(position)
    return substitutions


def mutant_codes(wildtype: str, substitutions: Sequence[tuple[str, int, str]]) -> np.ndarray:
    """Full-length residue codes of one variant, the wild type elsewhere."""

    codes = np.array([AA20.index(residue) for residue in wildtype], dtype=np.int64)
    for _, position, mutated in substitutions:
        codes[position - 1] = AA20.index(mutated)
    return codes


def build_variant_table(
    samples: Sequence[dict],
    *,
    wildtype_of_assay: dict[str, str],
    sequences: dict[str, str],
    cohort: dict | None = None,
    sample_id_of: Callable[[str, str], str] | None = None,
) -> list[dict]:
    """Validate the frozen sample table and restate it as a variant table.

    Every check here is a refusal, not a repair: a missing key, a duplicate
    ``sample_id``, a duplicate ``(assay, mutation)`` pair, a non-dense
    ``variant_index``, a broken source order, or a mutation string that disagrees
    with the wild type stops the export instead of yielding a table that would
    silently score a different variant under a retained identifier. ``cohort``,
    when supplied, is the measured cohort the samples were drawn from; it is used
    to require the same mutation order, the same retained mutant sequence and the
    same wild type, which is what ties a sample_id to a measured variant.
    """

    required = {"assay", "cluster", "mutation", "variant_index", "original_index", "sample_id"}
    rows: list[dict] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, sample in enumerate(samples):
        missing = required - set(sample)
        if missing:
            raise ValueError(f"sample {index} is missing {sorted(missing)}")
        assay, mutation = str(sample["assay"]), str(sample["mutation"])
        sample_id = str(sample["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id {sample_id} at row {index}")
        seen_ids.add(sample_id)
        if (assay, mutation) in seen_pairs:
            raise ValueError(f"{assay}: duplicate variant {mutation!r} at row {index}")
        seen_pairs.add((assay, mutation))
        if int(sample["original_index"]) != index:
            raise ValueError(
                f"{sample_id}: original_index {sample['original_index']} is not row {index}; the "
                "frozen sample order is not preserved")
        if sample_id_of is not None and sample_id_of(assay, mutation) != sample_id:
            raise ValueError(f"{sample_id}: sample_id is not the frozen identifier of {assay}/{mutation}")
        if assay not in wildtype_of_assay:
            raise ValueError(f"{assay}: no wild type is assigned to this assay")
        wildtype_id = wildtype_of_assay[assay]
        wildtype = sequences[wildtype_id]
        substitutions = parse_substitutions(mutation, wildtype)
        row = {
            "sample_id": sample_id,
            "assay": assay,
            "cluster": int(sample["cluster"]),
            "wildtype_id": wildtype_id,
            "mutation": mutation,
            "variant_index": int(sample["variant_index"]),
            "original_index": index,
            "substitutions": [[wild, position, mutated]
                              for wild, position, mutated in substitutions],
            "n_substitutions": len(substitutions),
        }
        if cohort is not None:
            _check_against_cohort(row, wildtype, cohort)
        rows.append(row)
    _check_assay_blocks(rows)
    return rows


def _check_against_cohort(row: dict, wildtype: str, cohort: dict) -> None:
    record = cohort["assays"].get(row["assay"])
    if record is None:
        raise ValueError(f"{row['assay']}: no measured cohort record for this assay")
    if record["wildtype"] != wildtype:
        raise ValueError(f"{row['assay']}: cohort wild type differs from the retained support's")
    if record["wildtype_id"] != row["wildtype_id"]:
        raise ValueError(f"{row['assay']}: cohort wild-type identifier disagrees")
    index = row["variant_index"]
    if index >= len(record["mutants"]):
        raise ValueError(f"{row['assay']}: variant_index {index} exceeds the cohort's variants")
    if record["mutants"][index] != row["mutation"]:
        raise ValueError(
            f"{row['assay']}: cohort variant {index} is {record['mutants'][index]!r}, the sample "
            f"table says {row['mutation']!r}")
    rebuilt = list(wildtype)
    for wild, position, mutated in parse_mutant(row["mutation"]):
        rebuilt[position - 1] = mutated
    if "".join(rebuilt) != record["sequences"][index]:
        raise ValueError(f"{row['assay']}: {row['mutation']} does not rebuild the cohort's sequence")


def _check_assay_blocks(rows: Sequence[dict]) -> None:
    """Each assay is one contiguous block with a dense, zero-based variant index."""

    previous, expected = None, 0
    for row in rows:
        if row["assay"] != previous:
            if previous is not None and expected == 0:
                raise ValueError(f"{previous}: empty assay block")
            previous, expected = row["assay"], 0
        if row["variant_index"] != expected:
            raise ValueError(
                f"{row['assay']}: variant_index {row['variant_index']} where {expected} is due")
        expected += 1


def substitution_counts(rows: Iterable[dict]) -> dict[str, int]:
    single = sum(1 for row in rows if row["n_substitutions"] == 1)
    return {"samples": sum(1 for row in rows), "single_substitution": single,
            "multiple_substitution": sum(1 for row in rows if row["n_substitutions"] > 1)}


# ----------------------------------------------------------------- the support


def stored_profile(query_id: str, wildtype: str, record: dict, frequencies: np.ndarray) -> Profile:
    """The retained profile of one wild type, read back from its stored frequencies.

    Rebuilt exactly as the retained store's own consumer rebuilds it, so the
    column weights a verification compares are the ones the store implies.
    """

    table = np.asarray(frequencies, dtype=np.float64)
    return Profile(query_id=query_id, wildtype=wildtype, n_hits=int(record["n_hits"]),
                   n_sequences=int(record["n_profile_sequences"]),
                   saturated=bool(record["hit_list_saturated"]), frequencies=table,
                   column_weight=table.sum(axis=1), neff=float(record["neff"]),
                   max_identity_over_query=float(record["max_identity_over_query"]))


def reconstructed_frequencies(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    frequencies = np.zeros((rows.shape[1], len(AA20)), dtype=np.float64)
    for code in range(len(AA20)):
        frequencies[:, code] = (weights[:, None] * (rows == code)).sum(axis=0)
    column_weight = frequencies.sum(axis=1)
    positive = column_weight > 0
    frequencies[positive] /= column_weight[positive, None]
    return frequencies


def reconstruct_support(wildtype: str, query_id: str, hits: Sequence, *, frequencies: np.ndarray,
                        record: dict) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Rebuild one wild type's homolog rows and prove they are the retained ones.

    ``verify_rows_against_profile`` is the archive's own cross-reader guard; the
    explicit deviation, row count and Neff comparison beside it is what makes the
    claim per wild type rather than per example, because the stored frequencies
    are float32 and a silent difference of one row would otherwise still pass a
    loose tolerance.
    """

    rows, weights, neff = alignment_rows(
        wildtype, query_id, hits, max_sequences=MAX_SEQUENCES, coverage_floor=COVERAGE_FLOOR,
        reweight_identity=REWEIGHT_IDENTITY)
    profile = stored_profile(query_id, wildtype, record, frequencies)
    verify_rows_against_profile(rows, weights, profile, tolerance=PROFILE_FREQUENCY_TOLERANCE)
    rebuilt = reconstructed_frequencies(rows, weights)
    deviation = float(np.abs(rebuilt - profile.frequencies).max()) if len(rows) else 0.0
    if len(rows) != int(record["n_profile_sequences"]):
        raise RuntimeError(
            f"{query_id}: rebuilt {len(rows)} homolog rows where the retained profile holds "
            f"{record['n_profile_sequences']}")
    if not np.isclose(neff, float(record["neff"]), rtol=1e-9, atol=1e-9):
        raise RuntimeError(
            f"{query_id}: rebuilt Neff {neff!r} where the retained profile records {record['neff']!r}")
    return rows, weights, neff, {
        "rows": int(len(rows)),
        "neff_at_30_percent_identity": float(neff),
        "weights_sum": float(weights.sum()),
        "effective_depth_per_site": float(neff) / len(wildtype),
        "frequencies_max_absolute_deviation": deviation,
        "frequencies_verified": deviation <= PROFILE_FREQUENCY_TOLERANCE,
    }


def write_query_first_alignment(fasta: Path, weights: Path, wildtype: str,
                                rows: np.ndarray, row_weights: np.ndarray) -> None:
    """The exported alignment: the wild type first at weight zero, then the homologs.

    Homolog rows keep the order and the weights the profile was built from, and
    the weights file is written at six decimals because the archived fit writes
    six decimals; the fit stage reads this file back, so what plmc was given and
    what the record reports cannot drift apart.
    """

    letters = np.array(list(AA20) + ["-"])
    with Path(fasta).open("w", encoding="utf-8") as handle:
        handle.write(f">focus\n{wildtype}\n")
        for index, row in enumerate(rows):
            handle.write(f">h{index:06d}\n{''.join(letters[row])}\n")
    lines = [f"{FOCUS_WEIGHT:.6f}"] + [f"{weight:.6f}" for weight in row_weights]
    Path(weights).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_exported_alignment(fasta: Path, weights_path: Path, wildtype: str
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Read back a query-first export, refusing anything the export cannot mean."""

    records = read_fasta(fasta)
    if not records:
        raise ValueError(f"{fasta} holds no records")
    if records[0][0] != "focus" or records[0][1] != wildtype:
        raise ValueError(f"{fasta}: first row is not this wild type at weight zero")
    sequences = [sequence for _, sequence in records[1:]]
    if any(len(sequence) != len(wildtype) for sequence in sequences):
        raise ValueError(f"{fasta}: an alignment row is not {len(wildtype)} columns wide")
    rows = encode_sequence("".join(sequences)).reshape(len(sequences), len(wildtype))
    weights = [float(line) for line in Path(weights_path).read_text().split() if line.strip()]
    if len(weights) != len(records):
        raise ValueError(
            f"{weights_path}: {len(weights)} weights for {len(records)} alignment rows")
    if weights[0] != FOCUS_WEIGHT:
        raise ValueError(f"{weights_path}: the query row does not carry weight {FOCUS_WEIGHT}")
    return rows, np.asarray(weights[1:], dtype=np.float64)


# ------------------------------------------------- the selection objective
# Restated from scripts/capability/interactions/fit_pairwise_sequence_baseline.py:
# a library module cannot import an entry point. The restatement is not trusted,
# it is checked -- tests/baselines/test_msa_baselines.py compares every function
# below against the archived implementation it restates.


def held_out_mask(rows: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """Held-out mask from a stable hash of each aligned row's own residues.

    The row is hashed as residue codes, never as its storage dtype: the archive
    hashed ``int8`` rows and a reader that hands over ``int64`` must land on the
    same split, or the regularization choice would depend on an array type.
    """

    boundary = int(HELD_OUT_FRACTION * (1 << 32))
    return np.array([
        int.from_bytes(hashlib.sha256(
            f"{seed}:".encode() + np.asarray(row, dtype=np.int8).tobytes()).digest()[:4],
            "big") < boundary for row in rows], dtype=bool)


def scored_support(rows: np.ndarray) -> np.ndarray:
    """Positions both models are scored on: non-gap, in rows with two or more."""

    present = rows != GAP_CODE
    return present & (present.sum(axis=1) >= MIN_PRESENT_PER_ROW)[:, None]


def independent_site(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted per-column amino-acid log probabilities over non-gap observations."""

    counts = np.full((rows.shape[1], len(AA20)), INDEPENDENT_SITE_PSEUDOCOUNT, dtype=np.float64)
    for code in range(len(AA20)):
        counts[:, code] += (weights[:, None] * (rows == code)).sum(axis=0)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def profile_conditional(log_profile: np.ndarray, rows: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-site log likelihood of ``mask`` positions under a site profile."""

    if not mask.any():
        raise ValueError("no held-out position to score")
    columns = np.broadcast_to(np.arange(rows.shape[1]), rows.shape)
    return float(log_profile[columns[mask], rows[mask]].mean())


def coupling_tensor(couplings: np.ndarray, length: int) -> np.ndarray:
    """Upper-triangle pair blocks as a symmetric ``(L*q, L*q)`` matrix."""

    q = couplings.shape[1]
    table = np.zeros((length, q, length, q), dtype=np.float64)
    k = 0
    for i in range(length):
        for j in range(i + 1, length):
            block = couplings[k].astype(np.float64)
            table[i, :, j, :] = block
            table[j, :, i, :] = block.T
            k += 1
    return table.reshape(length * q, length * q)


def plmc_conditional(fields: np.ndarray, couplings: np.ndarray, rows: np.ndarray,
                     mask: np.ndarray, *, block: int = 512) -> float:
    """Mean gap-ignoring per-site conditional log likelihood under h and J."""

    if not mask.any():
        raise ValueError("no held-out position to score")
    length, q = fields.shape
    table = coupling_tensor(couplings, length)
    present = rows != GAP_CODE
    total, count = 0.0, 0
    for start in range(0, len(rows), block):
        stop = min(start + block, len(rows))
        chunk, chunk_mask = rows[start:stop], mask[start:stop]
        if not chunk_mask.any():
            continue
        onehot = np.zeros((len(chunk), length, q), dtype=np.float64)
        rowsel, colsel = np.nonzero(present[start:stop])
        onehot[rowsel, colsel, chunk[rowsel, colsel]] = 1.0
        energy = (onehot.reshape(len(chunk), -1) @ table.T).reshape(len(chunk), length, q)
        energy += fields.astype(np.float64)[None]
        partition = np.logaddexp.reduce(energy, axis=2)
        gathered = np.where(chunk == GAP_CODE, 0, chunk).astype(np.int64)
        observed = np.take_along_axis(energy, gathered[:, :, None], axis=2)[:, :, 0]
        total += float((observed - partition)[chunk_mask].sum())
        count += int(chunk_mask.sum())
    return total / count


# ------------------------------------------------------------------- the model


@dataclass(frozen=True)
class PottsScorer:
    """Full-sequence Potts energy differences for one wild type.

    The energy is the plmc log weight, so a mutation whose mutant energy is
    higher is a mutation the model prefers. ``delta`` evaluates the complete
    difference: a term per changed position and a term per pair touching a
    changed position, which is exactly ``E(mutant) - E(wild type)`` without
    materializing an ``(L*q, L*q)`` tensor, because a pair with both endpoints at
    their wild-type residues cancels. Pairs with both endpoints changed are
    reached from either endpoint and are subtracted once, so no coupling is
    counted twice and none is omitted.
    """

    wildtype: str
    fields: np.ndarray
    couplings: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.wildtype)
        if self.fields.shape != (length, len(AA20)):
            raise ValueError("fields are not (length, 20)")
        pairs = length * (length - 1) // 2
        if self.couplings.shape != (pairs, len(AA20), len(AA20)):
            raise ValueError("couplings are not one upper-triangle block per site pair")
        if not np.isfinite(self.fields).all() or not np.isfinite(self.couplings).all():
            raise ValueError("nonfinite Potts parameters")

    @property
    def wild_codes(self) -> np.ndarray:
        return np.array([AA20.index(residue) for residue in self.wildtype], dtype=np.int64)

    def offsets(self) -> np.ndarray:
        """Row-major start index of each position's upper-triangle block."""

        length = len(self.wildtype)
        first = np.arange(length, dtype=np.int64)
        return first * length - first * (first + 1) // 2

    def delta(self, codes: np.ndarray) -> float:
        codes = np.asarray(codes, dtype=np.int64)
        length = len(self.wildtype)
        if codes.shape != (length,):
            raise ValueError(f"a mutant of a {length}-residue wild type needs {length} codes")
        if codes.min() < 0 or codes.max() >= len(AA20):
            raise ValueError("mutant codes leave the canonical amino-acid alphabet")
        wild = self.wild_codes
        changed = np.flatnonzero(codes != wild)
        if not len(changed):
            return 0.0
        total = float(self.fields[changed, codes[changed]].sum(dtype=np.float64)
                      - self.fields[changed, wild[changed]].sum(dtype=np.float64))
        offsets = self.offsets()
        for i in changed:
            i = int(i)
            if i + 1 < length:
                start, stop = int(offsets[i]), int(offsets[i]) + length - 1 - i
                block = self.couplings[start:stop]
                rows = np.arange(stop - start)
                js = np.arange(i + 1, length)
                total += float(
                    block[rows, codes[i], codes[js]].sum(dtype=np.float64)
                    - block[rows, wild[i], wild[js]].sum(dtype=np.float64))
            if i:
                js = np.arange(i)
                keys = offsets[js] + i - js - 1
                total += float(
                    self.couplings[keys, codes[js], codes[i]].sum(dtype=np.float64)
                    - self.couplings[keys, wild[js], wild[i]].sum(dtype=np.float64))
        if len(changed) > 1:
            rowsel, colsel = np.triu_indices(len(changed), k=1)
            i2, j2 = changed[rowsel], changed[colsel]
            keys = offsets[i2] + j2 - i2 - 1
            total -= float(
                self.couplings[keys, codes[i2], codes[j2]].sum(dtype=np.float64)
                - self.couplings[keys, wild[i2], wild[j2]].sum(dtype=np.float64))
        return total


def potts_scorer(params, wildtype: str) -> PottsScorer:
    """Wrap a read plmc parameter set as a full-sequence scorer of one wild type."""

    if params.target != wildtype:
        raise ValueError(
            f"plmc focus {params.target!r} is not the wild type; the model's columns do not "
            "correspond to this protein")
    if params.fields.shape[0] != len(wildtype):
        raise ValueError("plmc model length differs from the wild type")
    if set(params.positions.tolist()) != set(range(1, len(wildtype) + 1)):
        raise ValueError("plmc focus positions are not the wild-type columns one to one")
    return PottsScorer(wildtype, params.fields.astype(np.float64), params.couplings)


def profile_deltas(frequencies: np.ndarray, background: np.ndarray,
                   variants: Sequence[Sequence[tuple[str, int, str]]]) -> np.ndarray:
    """LOOKUP scores of variants of the wild type these frequencies belong to."""

    return np.array([
        lookup_score(frequencies, background, substitution_codes(substitutions),
                     alpha=PSEUDOCOUNT_ALPHA)
        for substitutions in variants], dtype=np.float64)


def shuffle_sign_check(scorer: PottsScorer, draws: int = 3, seed: int = SPLIT_SEED) -> dict:
    """The wild type must outrank composition-matched shuffles.

    This holds the exported direction: if ``delta`` were the negative log weight,
    the wild type would lose to a shuffle. Recorded per background rather than
    asserted globally, so one bad fit cannot hide inside an aggregate.
    """

    rng = np.random.default_rng(seed)
    codes = np.array([AA20.index(residue) for residue in scorer.wildtype], dtype=np.int64)
    deltas = [scorer.delta(rng.permutation(codes)) for _ in range(draws)]
    record = {"draws": draws, "shuffle_deltas": [round(value, 6) for value in deltas],
              "fraction_shuffles_below_wildtype": float(np.mean([value < 0.0 for value in deltas]))}
    if record["fraction_shuffles_below_wildtype"] < 1.0:
        raise ValueError(
            f"{scorer.wildtype}: the wild type does not outscore every shuffle under the exported "
            f"convention ({record['shuffle_deltas']}); the score's sign is suspect")
    return record


# ------------------------------------------------------------- the plmc fitting


class PlmcFailure(RuntimeError):
    """plmc exited unsuccessfully or wrote no parameters."""

    def __init__(self, message: str, log_tail: Sequence[str]):
        super().__init__(message + (f" | last lines: {list(log_tail)}" if log_tail else ""))
        self.log_tail = list(log_tail)


_ITERATION = re.compile(r"^\d+\t\d")


def run_plmc(executable: Path, alignment: Path, weights: Path, params: Path, log: Path, *,
             lambda_e: float, lambda_h: float, max_iter: int, threads: int) -> dict:
    """One plmc run, with its own terminal state parsed rather than assumed."""

    command = [str(executable), "-o", str(params), "-f", "focus", "-le", str(lambda_e),
               "-lh", str(lambda_h), "-m", str(max_iter), "-g", "-n", str(threads),
               "-w", str(weights), str(alignment)]
    completed = subprocess.run(command, capture_output=True, text=True)
    text = completed.stdout + completed.stderr
    Path(log).write_text(text, encoding="utf-8")
    if completed.returncode != 0 or not Path(params).exists():
        raise PlmcFailure(f"plmc exited {completed.returncode} for {alignment}",
                          text.strip().splitlines()[-3:])
    statuses = re.findall(r"Gradient optimization: (.*)", text)
    if not statuses:
        raise PlmcFailure(f"plmc printed no gradient status for {alignment}",
                          text.strip().splitlines()[-3:])
    status = statuses[-1].strip()
    iterations = [line.split("\t") for line in text.splitlines() if _ITERATION.match(line)]
    last = iterations[-1] if iterations else None
    effective = re.findall(r"Effective number of samples \(to 1 decimal place\): (\d+(?:\.\d+)?)",
                           text)
    return {
        "command": command,
        "status": status,
        "converged": status == CONVERGED_STATUS,
        "cap_hit": CAP_HIT_MARK in status,
        "iterations": int(last[0]) if last else None,
        "seconds": float(last[1]) if last else None,
        "terminal_condition": float(last[2]) if last else None,
        "final_objective": float(last[3]) if last else None,
        "effective_number_of_samples": float(effective[-1]) if effective else None,
        "lambda_e": lambda_e,
        "lambda_h": lambda_h,
        "max_iter": max_iter,
        "threads": threads,
    }


def regularization_policy(train_rows: int, held_out_rows: int, *, scorable: bool) -> str:
    """Which branch this background's support admits, decided before any fit."""

    if scorable and train_rows >= MIN_TRAIN_ROWS and held_out_rows >= MIN_HELD_OUT_ROWS:
        return POLICY_CROSS_VALIDATED
    return POLICY_FIXED_LOW_DEPTH


def escalating_fit(runner: Callable[..., dict], *, alignment: Path, weights: Path,
                   params: Path, log: Path, lambda_e: float, threads: int,
                   caps: Sequence[int] = MAX_ITER_CAPS) -> dict:
    """Run one regularization until plmc's own criterion is met, escalating the cap.

    The same focus, alignment, weights and ``lambda`` are handed to every attempt;
    only the iteration cap changes, in the declared order. Every attempt is kept
    in the record -- its cap, status, iterations and seconds -- so a reader can see
    that the delivered parameters came from a converged run and how many attempts
    that took. A run that stops at the highest cap is reported as unconverged; it
    is never promoted because a later fit elsewhere converged.

    ``runner`` is :func:`run_plmc` bound to the executable in production, and a
    stub in the tests that hold the escalation and selection rules.
    """

    attempts = []
    for cap in caps:
        attempt = runner(alignment, weights, params, log, lambda_e=lambda_e,
                         lambda_h=LAMBDA_H, max_iter=cap, threads=threads)
        attempts.append(attempt)
        if attempt["converged"]:
            break
    last = attempts[-1]
    return {
        "lambda_e": lambda_e,
        "lambda_h": LAMBDA_H,
        "status": STATUS_ATTEMPT_CONVERGED if last["converged"] else STATUS_ATTEMPT_UNCONVERGED,
        "converged": bool(last["converged"]),
        "caps_offered": [int(cap) for cap in caps],
        "attempts": attempts,
        "max_iter": last["max_iter"],
        "iterations": last["iterations"],
        "seconds": float(sum(attempt.get("seconds") or 0.0 for attempt in attempts)),
        "status_text": last["status"],
        "effective_number_of_samples": last.get("effective_number_of_samples"),
        "terminal_condition": last.get("terminal_condition"),
        "final_objective": last.get("final_objective"),
    }


def select_cross_validated(sweep: Sequence[dict]) -> dict:
    """Select ``lambda_E`` on held-out homolog sequences, or refuse.

    Selection requires the whole grid to have converged. A likelihood evaluated at
    a point the optimizer had not reached is not a cross-validated quantity, so a
    run that stopped at a cap is a refusal rather than a usable entry -- excluding
    it and choosing among the rest would let the grid's own best be decided by
    where the optimizer ran out of iterations.
    """

    failed = [entry for entry in sweep if entry["status"] == STATUS_PLMC_FAILED]
    unconverged = [entry["lambda_e"] for entry in sweep if not entry.get("converged")]
    present = {entry["lambda_e"] for entry in sweep}
    missing = [value for value in LAMBDA_E_GRID if value not in present]
    if failed or unconverged or missing:
        return {
            "status": STATUS_PLMC_FAILED if failed else STATUS_CV_NOT_CONVERGED,
            "selected_lambda_e": None,
            "failed_lambda_e": [entry["lambda_e"] for entry in failed],
            "unconverged_lambda_e": unconverged,
            "missing_lambda_e": missing,
            "failure": (
                "a cross-validated choice needs every grid value fitted and converged: "
                f"{len(failed)} failed to run, {len(unconverged)} stopped at a cap, "
                f"{len(missing)} absent"),
        }
    best = max(sweep, key=lambda entry: entry["held_out_conditional"])
    return {
        "status": STATUS_FITTED,
        "selected_lambda_e": best["lambda_e"],
        "selected_held_out_conditional": best["held_out_conditional"],
        "selected_at_grid_edge": best["lambda_e"] in (LAMBDA_E_GRID[0], LAMBDA_E_GRID[-1]),
        "selected_max_iter": best["max_iter"],
        "selected_attempts": len(best["attempts"]),
        "total_sweep_attempts": sum(len(entry["attempts"]) for entry in sweep),
        "failed_lambda_e": [],
        "unconverged_lambda_e": [],
        "missing_lambda_e": [],
    }


def fit_background(job: dict) -> dict:
    """Fit, select and score one wild type; the unit a resume checkpoints on.

    The job carries the exported alignment's own bytes, so the fit is reproducible
    from the export alone: the rows and the six-decimal weights are the ones the
    file holds, and the record names the file's digest.
    """

    from functools import partial

    from ..stability.pairwise_stability import read_plmc

    wildtype_id, wildtype = job["wildtype_id"], job["wildtype"]
    runner = partial(run_plmc, Path(job["plmc"]))
    work = Path(job["work"])
    work.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "wildtype_id": wildtype_id,
        "length": len(wildtype),
        "status": STATUS_FITTED,
        "support": job["support"],
        "regularization_policy": None,
        "cv_qualified": False,
        "selected_lambda_e": None,
        "lambda_h": LAMBDA_H,
        "max_iter_caps": [int(cap) for cap in MAX_ITER_CAPS],
        "threads": job["threads"],
        "plmc": job["plmc_record"],
    }
    rows, weights = read_exported_alignment(Path(job["msa"]), Path(job["weights"]), wildtype)
    record["alignment"] = {"msa": job["msa"], "msa_sha256": job["msa_sha256"],
                           "weights_sha256": job["weights_sha256"],
                           "rows": int(len(rows)), "weights_sum_on_disk": float(weights.sum())}
    if len(rows) < 2:
        record["status"] = STATUS_NO_SUPPORT
        record["failure"] = f"{len(rows)} homolog rows cannot carry a Potts fit"
        return record

    held = held_out_mask(rows)
    train, test = rows[~held], rows[held]
    test_mask = scored_support(test) if len(test) else np.zeros((0, rows.shape[1]), dtype=bool)
    record["split"] = {"seed": SPLIT_SEED, "held_out_fraction": HELD_OUT_FRACTION,
                       "train_rows": int(len(train)), "held_out_rows": int(len(test)),
                       "held_out_positions_scored": int(test_mask.sum()),
                       "min_train_rows": MIN_TRAIN_ROWS, "min_held_out_rows": MIN_HELD_OUT_ROWS}
    policy = regularization_policy(len(train), len(test), scorable=bool(test_mask.any()))
    record["regularization_policy"] = policy
    record["cv_qualified"] = policy == POLICY_CROSS_VALIDATED

    if policy == POLICY_CROSS_VALIDATED:
        train_weights = sequence_weights(train, identity=REWEIGHT_IDENTITY)
        site = independent_site(train, train_weights)
        record["train_neff_at_80_percent_identity"] = float(train_weights.sum())
        record["independent_site_held_out_conditional"] = profile_conditional(site, test, test_mask)
        sweep = []
        for lambda_e in LAMBDA_E_GRID:
            tag = f"train_le{lambda_e}"
            fasta, weight_path = work / f"{tag}.fa", work / f"{tag}.weights"
            params, log = work / f"{tag}.params", work / f"{tag}.log"
            write_query_first_alignment(fasta, weight_path, wildtype, train, train_weights)
            try:
                entry = escalating_fit(runner, alignment=fasta, weights=weight_path, params=params,
                                       log=log, lambda_e=lambda_e, threads=job["threads"])
            except PlmcFailure as failure:
                sweep.append({"lambda_e": lambda_e, "status": STATUS_PLMC_FAILED,
                              "converged": False, "attempts": [],
                              "failure": str(failure)})
                continue
            fitted = read_plmc(params)
            entry["held_out_conditional"] = plmc_conditional(
                fitted.fields, fitted.couplings, test, test_mask)
            sweep.append(entry)
            for path in (params, fasta, weight_path):
                path.unlink()
        record["sweep"] = sweep
        decision = select_cross_validated(sweep)
        record["selection"] = decision
        if decision["status"] != STATUS_FITTED:
            record["status"] = decision["status"]
            record["failure"] = decision["failure"]
            return record
        record["selected_lambda_e"] = decision["selected_lambda_e"]
        record["selected_held_out_conditional"] = decision["selected_held_out_conditional"]
        record["held_out_gain_over_independent_site"] = (
            decision["selected_held_out_conditional"]
            - record["independent_site_held_out_conditional"])
        record["selected_at_grid_edge"] = decision["selected_at_grid_edge"]
        selected = float(decision["selected_lambda_e"])
    else:
        record["selected_lambda_e"] = FIXED_LOW_DEPTH_LAMBDA_E
        record["selection_note"] = (
            "support below the held-out floor; the pinned external plmc example value is used "
            "and no cross-validation qualification is claimed")
        selected = FIXED_LOW_DEPTH_LAMBDA_E

    fasta, weight_path = work / "full.fa", work / "full.weights"
    params, log = work / "full.params", work / "full.log"
    write_query_first_alignment(fasta, weight_path, wildtype, rows, weights)
    try:
        final = escalating_fit(runner, alignment=fasta, weights=weight_path, params=params,
                               log=log, lambda_e=selected, threads=job["threads"])
    except PlmcFailure as failure:
        record["status"] = STATUS_PLMC_FAILED
        record["failure"] = str(failure)
        return record
    record["final"] = final
    record["final_seconds"] = final["seconds"]
    if final["effective_number_of_samples"] is not None:
        reported = final["effective_number_of_samples"]
        expected = float(weights.sum())
        record["effective_samples_check"] = {
            "plmc_reported": reported, "weights_sum_on_disk": round(expected, 6),
            "absolute_difference": round(abs(reported - expected), 6),
            "within_rounding": abs(reported - expected) <= 0.05 + 1e-6 * expected}
        if not record["effective_samples_check"]["within_rounding"]:
            raise RuntimeError(
                f"{wildtype_id}: plmc counted {reported} effective sequences where the "
                f"zero-weight query row implies {expected:.6f}; the focus is not weightless")

    retained = Path(job["params"])
    retained.parent.mkdir(parents=True, exist_ok=True)
    retained.write_bytes(params.read_bytes())
    record["params"] = {"path": str(retained), "sha256": sha256_file(retained),
                        "bytes": retained.stat().st_size}
    fitted = read_plmc(retained)
    scorer = potts_scorer(fitted, wildtype)
    record["sign_check"] = shuffle_sign_check(scorer)
    if not final["converged"]:
        record["status"] = STATUS_FINAL_NOT_CONVERGED
    record["sample_scores"] = {
        row["sample_id"]: scorer.delta(mutant_codes(wildtype, row["substitutions"]))
        for row in job["variants"]}
    distinct = len(set(round(value, 9) for value in record["sample_scores"].values()))
    record["score_variation"] = {
        "variants": len(record["sample_scores"]),
        "distinct_values": distinct,
        "note": "a thin alignment can yield one score for every variant; reported, not repaired"}
    for path in (params, fasta, weight_path):
        if path.exists():
            path.unlink()
    return record


def background_status(record: dict | None) -> str:
    if record is None:
        return STATUS_NOT_FITTED
    if record["status"] in SCOREABLE:
        return record["status"]
    return STATUS_BACKGROUND_FAILED


def assemble_scores(variant_table: Sequence[dict], records: dict[str, dict], *,
                    wildtype_ids: Sequence[str], protocol: dict, inputs: dict) -> tuple[dict, dict]:
    """The unified per-sample score table, in the frozen sample order.

    A sample whose background is absent or failed is retained with its explicit
    status and a ``NaN`` Potts value. Nothing is dropped, duplicated or padded
    with a zero, because a zero here would be a legitimate score.
    """

    samples = []
    potts = np.full(len(variant_table), np.nan, dtype=np.float64)
    status = np.empty(len(variant_table), dtype=object)
    for index, row in enumerate(variant_table):
        record = records.get(row["wildtype_id"])
        state = background_status(record)
        status[index] = state
        if record is not None and state in SCOREABLE:
            potts[index] = record["sample_scores"][row["sample_id"]]
        samples.append({
            "sample_id": row["sample_id"],
            "assay": row["assay"],
            "wildtype_id": row["wildtype_id"],
            "mutation": row["mutation"],
            "n_substitutions": row["n_substitutions"],
            "profile_delta": row["profile_delta"],
            "potts_delta": None if np.isnan(potts[index]) else float(potts[index]),
            "status": state,
        })
    resolved = sorted(name for name in wildtype_ids if name in records)
    summaries = {name: _background_summary(records[name]) for name in resolved}
    counts = {
        "samples": len(samples),
        "samples_with_potts": int(np.isfinite(potts).sum()),
        "backgrounds_expected": len(wildtype_ids),
        "backgrounds_resolved": len(resolved),
        "backgrounds_fitted": sum(records[n]["status"] in COMPLETE for n in resolved),
        "backgrounds_final_not_converged": sum(
            records[n]["status"] == STATUS_FINAL_NOT_CONVERGED for n in resolved),
        "backgrounds_cv_not_converged": sum(
            records[n]["status"] == STATUS_CV_NOT_CONVERGED for n in resolved),
        "backgrounds_failed": sum(
            records[n]["status"] not in SCOREABLE for n in resolved),
        "backgrounds_pending": len(wildtype_ids) - len(resolved),
        "total_fit_attempts": sum(
            len(entry.get("attempts", [])) for name in resolved
            for entry in list(records[name].get("sweep", []))
            + ([records[name]["final"]] if "final" in records[name] else [])),
        "maximum_cap_reached": max(
            (entry.get("max_iter") or 0 for name in resolved
             for entry in list(records[name].get("sweep", []))
             + ([records[name]["final"]] if "final" in records[name] else [])),
            default=None),
        "policy_counts": {
            policy: sum(records[n]["regularization_policy"] == policy for n in resolved)
            for policy in (POLICY_CROSS_VALIDATED, POLICY_FIXED_LOW_DEPTH)},
    }
    counts["complete"] = (counts["backgrounds_pending"] == 0
                          and counts["backgrounds_failed"] == 0
                          and counts["backgrounds_final_not_converged"] == 0
                          and counts["backgrounds_cv_not_converged"] == 0)
    document = {
        "schema": SCHEMA_SCORES,
        "protocol": protocol,
        "inputs": inputs,
        "counts": counts,
        "backgrounds": summaries,
        "samples": samples,
    }
    arrays = {
        "sample_id": np.array([row["sample_id"] for row in samples]),
        "assay": np.array([row["assay"] for row in samples]),
        "wildtype_id": np.array([row["wildtype_id"] for row in samples]),
        "mutation": np.array([row["mutation"] for row in samples]),
        "profile_delta": np.array([row["profile_delta"] for row in samples], dtype=np.float64),
        "potts_delta": potts,
        "status": np.array([str(value) for value in status]),
    }
    return document, arrays


def _background_summary(record: dict) -> dict:
    keep = ("status", "length", "regularization_policy", "cv_qualified", "selected_lambda_e",
            "held_out_gain_over_independent_site", "selected_at_grid_edge", "selection",
            "sweep", "final", "params", "sign_check", "effective_samples_check",
            "score_variation", "max_iter_caps", "failure", "support")
    return {key: record.get(key) for key in keep if key in record}


def manifest_protocol(*, plmc_record: dict) -> dict:
    return {
        "max_sequences": MAX_SEQUENCES,
        "coverage_floor": COVERAGE_FLOOR,
        "reweight_identity": REWEIGHT_IDENTITY,
        "neff_identity": NEFF_IDENTITY,
        "focus_weight": FOCUS_WEIGHT,
        "lambda_e_grid": list(LAMBDA_E_GRID),
        "lambda_h": LAMBDA_H,
        "max_iter_default": MAX_ITER,
        "fixed_low_depth_lambda_e": FIXED_LOW_DEPTH_LAMBDA_E,
        "held_out_fraction": HELD_OUT_FRACTION,
        "split_seed": SPLIT_SEED,
        "min_train_rows": MIN_TRAIN_ROWS,
        "min_held_out_rows": MIN_HELD_OUT_ROWS,
        "min_present_per_row": MIN_PRESENT_PER_ROW,
        "independent_site_pseudocount": INDEPENDENT_SITE_PSEUDOCOUNT,
        "selection_criterion": (
            "mean gap-ignoring per-site conditional log likelihood of held-out homolog rows; "
            "no phenotype, measured mutation or model output is read"),
        "msa_layout": (
            "query-first FASTA, the wild type as row 1 at weight 0.000000 in the sibling "
            ".weights file, homolog rows ordered by descending bit score; the same file is the "
            "shared alignment of the GEMME baseline, whose method includes the query as a "
            "homolog -- a declared difference in method, not in support"),
        "profile_pseudocount": PSEUDOCOUNT_ALPHA,
        "profile_score": PROFILE_SCORE_DEFINITION,
        "potts_score": POTTS_SCORE_DEFINITION,
        "plmc": plmc_record,
    }


def read_manifest(out_dir: Path) -> dict:
    path = Path(out_dir) / "inputs" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is absent; the export stage must run before anything is fitted")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA_MANIFEST:
        raise ValueError(f"{path}: schema {manifest.get('schema')!r} is not {SCHEMA_MANIFEST!r}")
    return manifest


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def verify_inputs(out_dir: Path, *, root: Path) -> dict:
    """Re-hash every byte the frozen export declares, and refuse a mismatch.

    The export's own manifest mixes the code that generated the inputs with the
    inputs themselves. Only the inputs are re-verified here: the generating code
    is provenance and is bound by the manifest's digest rather than compared with
    today's code. Nothing is skipped -- the source files and all 174 alignments
    and weight files are hashed on every stage entry.
    """

    out_dir = Path(out_dir)
    manifest_path = out_dir / "inputs" / "manifest.json"
    manifest = read_manifest(out_dir)
    source, drift = {}, []
    for label, entry in manifest["inputs"].items():
        path = _resolve(root, entry["path"])
        if not path.is_file():
            drift.append(f"source {label}: {path} is missing")
        elif sha256_file(path) != entry["sha256"]:
            drift.append(f"source {label}: {path} changed")
        source[label] = {"path": entry["path"], "sha256": entry["sha256"]}
    support_path = out_dir / "inputs" / "support.json"
    table_path = out_dir / "inputs" / "variant_table.json"
    support = json.loads(support_path.read_text())
    table = json.loads(table_path.read_text())
    if table["source_sha256"] != source["samples"]["sha256"]:
        drift.append("variant_table: its source digest is not the manifest's sample table")
    msa_weights = {}
    for name, entry in support["wildtypes"].items():
        for label, field in (("msa", "msa"), ("weights", "weights")):
            path = _resolve(root, entry[field])
            if not path.is_file():
                drift.append(f"{label} {name}: {path} is missing")
            elif sha256_file(path) != entry[f"{label}_sha256"]:
                drift.append(f"{label} {name}: {path} changed")
        msa_weights[name] = {"msa_sha256": entry["msa_sha256"],
                             "weights_sha256": entry["weights_sha256"]}
    if drift:
        raise RuntimeError(
            "the frozen inputs no longer match their manifest; nothing may be fitted from them -- "
            + "; ".join(drift))
    return {
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "source": source,
        "derived": {
            "variant_table": {"path": str(table_path), "sha256": sha256_file(table_path)},
            "support": {"path": str(support_path), "sha256": sha256_file(support_path)},
            "msa_weights": msa_weights,
        },
    }


def optimization_protocol(*, plmc_record: dict) -> dict:
    """The pinned numerical protocol of the Potts fits, independent of the export."""

    return {
        "engine": "plmc",
        "lambda_e_grid": list(LAMBDA_E_GRID),
        "lambda_h": LAMBDA_H,
        "max_iter_caps": [int(cap) for cap in MAX_ITER_CAPS],
        "cap_order": "declared order; each cap retries the same focus, alignment, weights and lambda",
        "convergence": (
            "plmc's own terminal status 'Minimization success'; a run stopped by -m is "
            "unconverged and is not usable"),
        "selection_criterion": (
            "mean gap-ignoring per-site conditional log likelihood of held-out homolog rows; "
            "no phenotype, measured mutation or model output is read"),
        "selection_requires": (
            "every grid value fitted and converged; an unconverged or failed value refuses the "
            "background rather than being excluded from the choice"),
        "fixed_low_depth_lambda_e": FIXED_LOW_DEPTH_LAMBDA_E,
        "held_out_fraction": HELD_OUT_FRACTION,
        "split_seed": SPLIT_SEED,
        "min_train_rows": MIN_TRAIN_ROWS,
        "min_held_out_rows": MIN_HELD_OUT_ROWS,
        "min_present_per_row": MIN_PRESENT_PER_ROW,
        "independent_site_pseudocount": INDEPENDENT_SITE_PSEUDOCOUNT,
        "focus_weight": FOCUS_WEIGHT,
        "gap_ignoring": True,
        "potts_score": POTTS_SCORE_DEFINITION,
        "plmc": plmc_record,
    }


def verify_execution(execution: dict, *, inputs: dict, code_paths: dict[str, Path],
                     plmc_record: dict) -> None:
    """The fit stage continues only on the same inputs, the same code, the same protocol."""

    if execution.get("schema") != SCHEMA_EXECUTION:
        raise RuntimeError(
            f"execution manifest schema {execution.get('schema')!r} is not {SCHEMA_EXECUTION!r}")
    drift = []
    if execution["inputs"] != inputs:
        drift.append("the frozen inputs or a derived alignment differ from the ones this "
                     "execution started from")
    for label, path in code_paths.items():
        recorded = execution["code"].get(label)
        if recorded is None:
            drift.append(f"code {label}: not recorded")
        elif recorded != sha256_file(path):
            drift.append(f"code {label}: changed since this execution started")
    if execution["optimization"] != optimization_protocol(plmc_record=plmc_record):
        drift.append("the optimization protocol differs from the one this execution started with")
    if drift:
        raise RuntimeError(
            "refusing to continue this fit tree -- " + "; ".join(drift))


def load_or_create_execution(fit_out: Path, *, inputs: dict, code_paths: dict[str, Path],
                             plmc_record: dict) -> dict:
    """Create the fit tree's own manifest, or verify the one it already has.

    The frozen export's manifest is bound by digest and its input-generation code
    is kept there as provenance; this manifest records the code and the numerical
    protocol that are actually executing, so fixing the execution no longer
    invalidates the export that another baseline is reading.
    """

    from datetime import datetime, timezone

    path = Path(fit_out) / "execution_manifest.json"
    if path.is_file():
        execution = json.loads(path.read_text())
        verify_execution(execution, inputs=inputs, code_paths=code_paths,
                         plmc_record=plmc_record)
        return execution
    execution = {
        "schema": SCHEMA_EXECUTION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "input_generation_code_provenance": (
            "inputs/manifest.json#code, bound by the inputs manifest digest above and not "
            "compared with the execution code"),
        "code": {label: sha256_file(path) for label, path in code_paths.items()},
        "optimization": optimization_protocol(plmc_record=plmc_record),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(execution, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)
    return execution
