"""Predicted absolute folding free energy for small protein domains, and its validation.

What this is for
================

E14 could report structure and sequence-level family recognition for generated
proteins but not stability, because no stability predictor existed anywhere in
this project and predicted structural *confidence* is not stability. This module
builds the missing instrument for the one regime where enough measured data
exists to build and check it: small single domains.

**The validation is the deliverable, not the predictor.** A regression head on
frozen protein-language-model embeddings is easy to fit and easy to fool. What
decides whether it may touch a generated sequence is whether it still ranks
free energy on a dataset it was not trained on, and whether it beats a
composition-and-length baseline there. Those two conditions are declared in
:data:`GATE_CONDITIONS` before any number is computed, and
:func:`evaluate_gate` is the only thing licensed to open the door.

Two quantities that must never be substituted for one another
=============================================================

Folding confidence and folding free energy are different quantities with
different units, and the whole reason this module exists is that one was being
asked to stand in for the other. So they are not merely documented as distinct:
:data:`QUANTITIES` declares each quantity's unit and kind, and
:func:`require_comparable` **raises** when a confidence quantity and a free-energy
quantity are brought into the same comparison. Substituting one for the other is
a crash, not a judgement call.

What the predictor is and is not
================================

It predicts the *absolute* folding free energy of a whole small domain, in
kcal/mol, from sequence alone. It is not a ddG predictor: it takes no wild type
and no mutation, which is exactly why it applies to a generated sequence that has
no wild type. It is not a measurement. And it is licensed only inside the length
band its training split actually covers -- :func:`require_in_band` refuses
anything else, because a two-state folding free energy is not even well defined
for a chain several times longer than any domain in the training data.

The independence of the two validation sets, stated honestly
============================================================

The training data and the cross-dataset check come from the same measurement
technology: both are cDNA-display proteolysis assays reporting a free energy
derived from trypsin and chymotrypsin K50 values, and this repository's own notes
record that they share an author and a technology and therefore cannot serve as
each other's independent confirmation. So cross-dataset transfer here tests
generalisation across *sequence populations* -- MGnify-derived domains versus
natural and de novo designed domains -- and not across measurement technology.
Cross-technology corroboration, where it is available at all, comes from a
third-party checkpoint that was trained by other people on other data and that
must pass the same gate before its agreement counts for anything.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from ..core.statistics import bootstrap_unit_floor, paired_group_bootstrap
from .generated_phenotype import AA20

SCHEMA_VERSION = "d1_domain_stability_v1"

#: Every quantity this experiment reports, declared along three axes that are
#: deliberately kept apart.
#:
#: ``kind`` is the *physical* quantity, and it is what :func:`require_comparable`
#: enforces. ``source`` says whether a reading was predicted or measured, and it
#: is **not** enforced, because comparing a prediction with the measurement of the
#: same physical quantity is the validation this experiment exists to do. What
#: must never happen is comparing or exchanging two different physical quantities:
#: a folding confidence with a free energy, or an undocumented thermostability
#: score with either.
QUANTITIES: dict[str, dict[str, str]] = {
    "esmfold2_mean_ca_plddt": {
        "unit": "plddt_0_100",
        "kind": "structure_confidence",
        "source": "predicted",
        "is_not": "a free energy, a melting temperature or any thermodynamic quantity",
    },
    "esmfold2_ptm": {
        "unit": "dimensionless_0_1",
        "kind": "structure_confidence",
        "source": "predicted",
        "is_not": "a free energy, a melting temperature or any thermodynamic quantity",
    },
    "predicted_delta_g": {
        "unit": "kcal_per_mol",
        "kind": "folding_free_energy",
        "source": "predicted",
        "is_not": (
            "a measurement, a ddG, or a statement about any chain longer than the "
            "licensed band"
        ),
    },
    "measured_delta_g": {
        "unit": "kcal_per_mol",
        "kind": "folding_free_energy",
        "source": "measured",
        "is_not": "a prediction",
    },
    "prime_value_head": {
        "unit": "undeclared_by_model_card",
        "kind": "thermostability_proxy",
        "source": "predicted",
        "is_not": (
            "a free energy in kcal/mol. The checkpoint's model card is an empty "
            "template and declares no units, so this score is used for rank "
            "agreement only, is never converted to kcal/mol and is never averaged "
            "with a free energy"
        ),
    },
}

#: The conditions a candidate instrument must satisfy before it is allowed near a
#: generated sequence. Declared here, before any of them is evaluated, so that the
#: threshold cannot be chosen after seeing the result.
GATE_CONDITIONS: dict[str, str] = {
    "G1_transfers": (
        "on the cross-dataset check, the instrument's Spearman correlation with "
        "measured free energy must be positive with a 95% group-bootstrap interval "
        "excluding zero. An instrument that fits its own test fold and does not rank "
        "a second dataset has learned its assay's sequence population, not stability"
    ),
    "G2_beats_the_cheap_baseline": (
        "on the same cross-dataset check, the instrument must exceed a "
        "composition-and-length-only ridge trained on the identical split, with a "
        "95% paired interval on the difference excluding zero. Without this, any "
        "apparent skill could be residue composition, which is free"
    ),
    "both_required": (
        "G1 and G2 must both hold. If either fails, the instrument is reported as "
        "unusable and no generated sequence is scored with it"
    ),
    "unit_of_resampling": (
        "the wild-type cluster of the cross-dataset check, so that thousands of "
        "variants of one domain do not read as thousands of independent units"
    ),
}


# --------------------------------------------------------------- the two guards


def require_comparable(left: str, right: str) -> None:
    """Refuse to compare two different physical quantities.

    This is the mechanism, not the advice. The substitution this whole experiment
    exists to prevent is exchanging a folding confidence for a folding free
    energy, so it raises here rather than being left to a reader to notice in a
    table. Predicted against measured is *allowed* when the physical quantity and
    unit agree: that comparison is the validation.
    """

    unknown = sorted({left, right} - set(QUANTITIES))
    if unknown:
        raise KeyError(f"undeclared quantity {unknown}; declare it in QUANTITIES first")
    left_kind = QUANTITIES[left]["kind"]
    right_kind = QUANTITIES[right]["kind"]
    if left_kind != right_kind:
        raise ValueError(
            f"{left!r} is a {left_kind} in {QUANTITIES[left]['unit']} and {right!r} is a "
            f"{right_kind} in {QUANTITIES[right]['unit']}. These are different physical "
            f"quantities and must not be compared or substituted: {left!r} is not "
            f"{QUANTITIES[left]['is_not']}"
        )
    if QUANTITIES[left]["unit"] != QUANTITIES[right]["unit"]:
        raise ValueError(
            f"{left!r} and {right!r} are both {left_kind} but are expressed in "
            f"{QUANTITIES[left]['unit']} and {QUANTITIES[right]['unit']}; one would have "
            "to be converted before they could be compared, and no conversion is declared"
        )


def require_in_band(lengths: Sequence[int], band: Sequence[int], *, label: str) -> None:
    """Refuse to score sequences outside the predictor's licensed length band."""

    low, high = int(band[0]), int(band[1])
    array = np.asarray(lengths, dtype=np.int64)
    if array.size < 1:
        raise ValueError(f"{label}: nothing to score")
    outside = array[(array < low) | (array > high)]
    if outside.size:
        raise ValueError(
            f"{label}: {outside.size} sequences fall outside the licensed band "
            f"[{low}, {high}] (lengths {sorted(set(outside.tolist()))[:6]}). The "
            "predictor was fitted only inside that band and a folding free energy is "
            "not defined for a chain far outside it, so these are refused rather than "
            "extrapolated"
        )


def licensed_band(lengths: Iterable[int]) -> tuple[int, int]:
    """The band a training set actually covers, which is what licenses the predictor."""

    array = np.asarray(list(lengths), dtype=np.int64)
    if array.size < 1:
        raise ValueError("a licensed band cannot be derived from an empty training set")
    return int(array.min()), int(array.max())


# ----------------------------------------------------------- the cheap baseline

#: The composition-and-length baseline's feature names, in order. Twenty residue
#: fractions plus length and its logarithm: everything a reader would reasonably
#: suspect of explaining a stability prediction for free.
BASELINE_FEATURES: tuple[str, ...] = tuple(f"fraction_{residue}" for residue in AA20) + (
    "length",
    "log_length",
)


def composition_features(sequence: str) -> np.ndarray:
    """The cheap feature vector of one sequence, in :data:`BASELINE_FEATURES` order."""

    if not sequence:
        raise ValueError("a feature vector needs a sequence")
    unknown = sorted(set(sequence) - set(AA20))
    if unknown:
        raise ValueError(f"non-canonical residue(s) {unknown} in a modelling sequence")
    counts = np.zeros(len(AA20) + 2, dtype=np.float64)
    for index, residue in enumerate(AA20):
        counts[index] = sequence.count(residue) / len(sequence)
    counts[len(AA20)] = float(len(sequence))
    counts[len(AA20) + 1] = math.log(len(sequence))
    return counts


# ------------------------------------------------------------------ the fitting


class RidgeGram:
    """Ridge regression accumulated in chunks, so no design matrix is materialised.

    A million rows by thirteen hundred features is five gigabytes of float32 that
    this never needs to hold at once, and the pod's memory is not ours to assume.
    The normal equations are the same either way: features are standardised with
    statistics accumulated in the same pass, the intercept is carried by centring
    the target, and the penalty is applied to the standardised coefficients so one
    alpha means the same thing for every feature.
    """

    def __init__(self, n_features: int) -> None:
        if n_features < 1:
            raise ValueError("a design needs at least one feature")
        self.n_features = int(n_features)
        self._n = 0
        self._sum = np.zeros(n_features, dtype=np.float64)
        self._sum_sq = np.zeros(n_features, dtype=np.float64)
        self._gram = np.zeros((n_features, n_features), dtype=np.float64)
        self._xty = np.zeros(n_features, dtype=np.float64)
        self._sum_y = 0.0
        self._sum_y_sq = 0.0
        self._finalised = False

    def add(self, features: np.ndarray, targets: np.ndarray) -> None:
        if self._finalised:
            raise RuntimeError("this accumulator has already been finalised")
        block = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if block.ndim != 2 or block.shape[1] != self.n_features or block.shape[0] != y.size:
            raise ValueError("feature block and target vector shapes disagree")
        if not np.isfinite(block).all() or not np.isfinite(y).all():
            raise ValueError("a non-finite value reached the ridge accumulator")
        self._n += y.size
        self._sum += block.sum(axis=0)
        self._sum_sq += np.einsum("ij,ij->j", block, block)
        self._gram += block.T @ block
        self._xty += block.T @ y
        self._sum_y += float(y.sum())
        self._sum_y_sq += float(y @ y)

    def finalise(self) -> None:
        if self._n <= self.n_features:
            raise ValueError(
                f"{self._n} rows cannot identify {self.n_features} coefficients; the "
                "ridge solution would be determined by the penalty rather than the data"
            )
        self._mean = self._sum / self._n
        variance = self._sum_sq / self._n - self._mean**2
        # A constant feature carries no information and would divide by zero. It is
        # kept with unit scale so the coefficient is simply penalised to nothing,
        # rather than silently dropped and changing the feature indexing.
        self._scale = np.sqrt(np.maximum(variance, 0.0))
        self._constant = self._scale <= 1e-12
        self._scale[self._constant] = 1.0
        self._y_mean = self._sum_y / self._n
        # Gram and cross-product of the standardised, centred design.
        outer = np.outer(self._mean, self._mean) * self._n
        centred = (self._gram - outer) / np.outer(self._scale, self._scale)
        self._centred_gram = centred
        self._centred_xty = (self._xty - self._mean * self._sum_y) / self._scale
        self._finalised = True

    @property
    def n_rows(self) -> int:
        return self._n

    @property
    def n_constant_features(self) -> int:
        self._require_final()
        return int(self._constant.sum())

    def _require_final(self) -> None:
        if not self._finalised:
            raise RuntimeError("call finalise() before solving or transforming")

    def solve(self, alpha: float) -> dict[str, Any]:
        """Coefficients on the standardised design, plus the intercept."""

        self._require_final()
        if alpha <= 0.0:
            raise ValueError("the ridge penalty must be positive")
        matrix = self._centred_gram + alpha * np.eye(self.n_features)
        coefficients = np.linalg.solve(matrix, self._centred_xty)
        return {
            "alpha": float(alpha),
            "coefficients": coefficients,
            "intercept": float(self._y_mean),
            "mean": self._mean.copy(),
            "scale": self._scale.copy(),
        }

    def predict(self, model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
        block = np.asarray(features, dtype=np.float64)
        if block.ndim != 2 or block.shape[1] != self.n_features:
            raise ValueError("feature block does not match the fitted design")
        standardised = (block - model["mean"]) / model["scale"]
        return standardised @ model["coefficients"] + model["intercept"]


def predict_with(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    """Apply a solved ridge model to a feature block, without the accumulator."""

    block = np.asarray(features, dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != coefficients.size:
        raise ValueError("feature block does not match the fitted design")
    standardised = (block - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return standardised @ coefficients + float(model["intercept"])


# --------------------------------------------------------------- the validation


def _spearman_metric(truth: np.ndarray, prediction: np.ndarray) -> float:
    if np.ptp(truth) == 0.0 or np.ptp(prediction) == 0.0:
        return float("nan")
    return float(stats.spearmanr(truth, prediction).statistic)


def _left_score(left: float, _right: float) -> float:
    return left


def regression_report(
    truth: Sequence[float], prediction: Sequence[float], *, unit: str
) -> dict[str, Any]:
    """Point accuracy of a free-energy prediction, in its own units.

    Reported without an interval; the intervals live in :func:`transfer_contrast`,
    where the resampling unit is declared. ``rmse`` and ``mae`` are meaningful only
    when the prediction is in the same units as the measurement, so a prediction
    whose units are undeclared reports rank agreement and nothing else.
    """

    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size < 2:
        raise ValueError("truth and prediction must be one-dimensional and aligned")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("a non-finite value reached the regression report")
    report: dict[str, Any] = {
        "n": int(y.size),
        "spearman": _spearman_metric(y, p),
        "pearson": float(stats.pearsonr(y, p).statistic) if np.ptp(p) else None,
        "unit": unit,
    }
    if unit == QUANTITIES["measured_delta_g"]["unit"]:
        residual = p - y
        report["rmse"] = float(np.sqrt(np.mean(residual**2)))
        report["mae"] = float(np.mean(np.abs(residual)))
        report["bias"] = float(np.mean(residual))
    else:
        report["rmse"] = None
        report["mae"] = None
        report["bias"] = None
        report["error_not_reported_because"] = (
            f"the prediction is in {unit!r}, not kcal/mol, so an error against a "
            "measured free energy would have no units. Rank agreement only"
        )
    return report


def transfer_contrast(
    truth: Sequence[float],
    instrument: Sequence[float],
    baseline: Sequence[float],
    groups: Sequence[Any],
    *,
    seed: int,
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """The two gate quantities in one group bootstrap over the declared unit.

    ``derived`` carries the instrument's own Spearman and its interval (condition
    G1); ``difference`` carries the instrument minus the composition-and-length
    baseline and its interval (condition G2). Both come from the same resampled
    units, so the two conditions are consistent with each other draw by draw.
    """

    y = np.asarray(truth, dtype=np.float64)
    left = np.asarray(instrument, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if not (y.shape == left.shape == right.shape) or y.ndim != 1:
        raise ValueError("truth, instrument and baseline vectors must align")
    unit_ids = [str(group) for group in groups]
    if len(unit_ids) != y.size:
        raise ValueError("the resampling units must align with the rows")
    floor = bootstrap_unit_floor(len({*unit_ids}))
    if floor["degenerate"]:
        return {
            "resolved": False,
            "unit_floor": floor,
            "reason": floor["degenerate_reason"],
            "instrument_spearman": _spearman_metric(y, left),
            "baseline_spearman": _spearman_metric(y, right),
        }
    bootstrap = paired_group_bootstrap(
        y,
        left,
        right,
        unit_ids,
        _spearman_metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
        derived_statistic=_left_score,
    )
    return {
        "resolved": True,
        "n": int(y.size),
        "n_groups": bootstrap["n_groups"],
        "instrument_spearman": bootstrap["left_score"],
        "instrument_spearman_ci95": bootstrap["derived_ci95"],
        "baseline_spearman": bootstrap["right_score"],
        "difference": bootstrap["difference"],
        "difference_ci95": bootstrap["difference_ci95"],
        "n_finite_draws": bootstrap["n_finite_draws"],
        "unit_floor": floor,
    }


def evaluate_gate(contrast: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a candidate instrument may be applied to generated sequences.

    The only function licensed to answer that. It reads a
    :func:`transfer_contrast` and applies :data:`GATE_CONDITIONS` verbatim; it
    chooses no threshold, because there is none to choose -- both conditions are
    "an interval excludes zero".
    """

    if not contrast.get("resolved"):
        return {
            "passed": False,
            "G1_transfers": False,
            "G2_beats_the_cheap_baseline": False,
            "reason": (
                "the cross-dataset contrast was not resolvable: "
                f"{contrast.get('reason', 'no interval was computed')}"
            ),
            "conditions": dict(GATE_CONDITIONS),
        }
    low_1, high_1 = contrast["instrument_spearman_ci95"]
    low_2, high_2 = contrast["difference_ci95"]
    g1 = bool(low_1 > 0.0)
    g2 = bool(low_2 > 0.0)
    reasons = []
    if not g1:
        reasons.append(
            f"G1 failed: cross-dataset Spearman {contrast['instrument_spearman']:.4f} "
            f"[{low_1:.4f}, {high_1:.4f}] does not exclude zero from below"
        )
    if not g2:
        reasons.append(
            f"G2 failed: the increment over the composition-and-length baseline is "
            f"{contrast['difference']:.4f} [{low_2:.4f}, {high_2:.4f}], which does not "
            "exclude zero from below, so the instrument adds nothing to a free feature"
        )
    return {
        "passed": bool(g1 and g2),
        "G1_transfers": g1,
        "G2_beats_the_cheap_baseline": g2,
        "reason": "; ".join(reasons) if reasons else "both declared conditions hold",
        "conditions": dict(GATE_CONDITIONS),
    }


def transformers_compatibility_shim() -> dict[str, Any]:
    """Make third-party code written for transformers 4.41 import under 4.57.

    The PRIME checkpoint's vendored modelling file imports
    ``find_pruneable_heads_and_indices`` and ``prune_linear_layer`` from
    ``transformers.modeling_utils``; both moved to ``transformers.pytorch_utils``.
    The names are rebound on the module object **in this process** rather than by
    editing the downloaded file, so nothing under the model directory is modified
    and the change cannot outlive the run. The rebinding is recorded in the
    artefact, because patching somebody else's code to obtain a number is part of
    how that number was obtained.
    """

    import transformers
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils

    rebound: list[str] = []
    for name in ("find_pruneable_heads_and_indices", "prune_linear_layer", "apply_chunking_to_forward"):
        if not hasattr(modeling_utils, name):
            if not hasattr(pytorch_utils, name):
                raise ImportError(
                    f"transformers {transformers.__version__} carries {name!r} in "
                    "neither modeling_utils nor pytorch_utils; the shim is out of date "
                    "and the checkpoint must not be loaded on a guess"
                )
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
            rebound.append(name)
    return {
        "transformers_version": transformers.__version__,
        "target_version_of_vendored_code": "4.41",
        "rebound_into_transformers_modeling_utils": rebound,
        "source_module": "transformers.pytorch_utils",
        "files_modified_on_disk": [],
        "why": (
            "the checkpoint ships its own modelling code written against an older "
            "transformers; these names moved module. Rebinding them in-process keeps "
            "the downloaded directory untouched"
        ),
    }
