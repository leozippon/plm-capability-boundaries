"""In-process ESMFold2 mean CA-pLDDT, for a reward that is paid during a run.

The project's folding instrument is an evidence tree: one directory per
sequence digest, a prediction object, a PDB, a worker receipt and a replayable
index, written by ``scripts/capability/generation/run_structure_evidence.py``.
That is the right shape for a cohort measured once and read many times, and the
wrong shape for a reward paid forty-eight times a step inside a training loop,
where the structure is consumed immediately and only one scalar survives.

This module is the same measurement without the tree. It keeps every field of
the contract that decides the number -- the staged ESMFold2 checkpoint, its own
``num_loops`` and ``inference_num_steps``, the declared diffusion-sample count,
the highest-pTM selection rule, the content-addressed seed rule, bfloat16 -- and
reads the mean CA-pLDDT through
:func:`src.capability.generation.structure_evidence.summarize_esmfold2`, which
is the one implementation of that read. It writes nothing, so it makes no claim
to be an evidence record: a run that needs the tree runs the runner.

Two consequences are declared rather than worked around. The fold is not
bit-reproducible -- the project measured that at the same seed on the same card
-- so a reward paid here is conditional on its run, and the between-realisation
spread of a mean CA-pLDDT at the declared sample count is measured and recorded
beside the experiment rather than assumed away. And the seed rule is
content-addressed, so one sequence has one reward within a realisation however
many times it is sampled, which is what makes a cached score legitimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..generation import structure_evidence as se
from . import sequence_reward as sr

#: The declared diffusion-sample count. This is the staged checkpoint's own
#: published default and the count every tree in this project folded at, kept
#: here because the measured cost of 32 samples over 1 sample on an L20 is a
#: factor of 1.7 at these lengths -- the trunk dominates -- so there is nothing
#: to buy by departing from the project's contract and a departure would have to
#: be declared and bounded.
DIFFUSION_SAMPLES = 32

#: Seed rule. Content-addressed, so the noise enters the between-sequence spread
#: a resampling interval already resamples instead of being common-mode.
SEED_RULE = se.SEED_RULE_SEQUENCE_DIGEST

#: The reward's realisation. ``A`` is what training is paid against and what
#: every pool mean is read from; ``B`` is the independent realisation the
#: best-of-k control evaluates its selection on, so that selecting on a noisy
#: score and reporting the same noisy score cannot inflate the control.
BASE_SEED_REALISATION: dict[str, int] = {"A": 20261114, "B": 20261115}


@dataclass(frozen=True)
class FoldReading:
    """One sequence's reading, and the facts that identify the measurement."""

    sequence: str
    mean_ca_plddt: float
    ptm: float
    diffusion_sample_index: int
    seconds: float

    def as_record(self) -> dict[str, Any]:
        return {
            "length": len(self.sequence),
            "mean_ca_plddt": self.mean_ca_plddt,
            "ptm": self.ptm,
            "diffusion_sample_index": self.diffusion_sample_index,
            "seconds": self.seconds,
        }


class PlddtScorer:
    """A loaded ESMFold2 that answers mean CA-pLDDT for one sequence at a time.

    ``realisation`` selects the declared base seed. The cache is keyed on the
    sequence, which is exactly the scope the content-addressed seed rule makes
    correct: two identical sequences fold identically within a realisation, so
    re-folding one is cost without information. It is bounded, because a long
    run's distinct completions are not bounded.
    """

    def __init__(self, model_dir: Path, *, device: str, realisation: str = "A",
                 diffusion_samples: int = DIFFUSION_SAMPLES, cache_limit: int = 20_000) -> None:
        import torch
        from transformers import EsmFold2Model
        from transformers.models.esmfold2.protein_utils import prepare_protein_features

        if realisation not in BASE_SEED_REALISATION:
            raise KeyError(
                f"unknown realisation {realisation!r}; declared: "
                f"{sorted(BASE_SEED_REALISATION)}"
            )
        if diffusion_samples < 1:
            raise ValueError("a fold draws at least one diffusion sample")
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("the ESMFold2 instrument needs an assigned CUDA device")
        self.model_dir = Path(model_dir)
        config = se.require_esmfold2_checkpoint(self.model_dir)
        head = config.get("structure_head") or {}
        self.contract = {
            "model_type": config.get("model_type"),
            "num_loops": config.get("num_loops"),
            "inference_num_steps": head.get("inference_num_steps"),
            "published_default_num_diffusion_samples": head.get("num_diffusion_samples"),
        }
        self.device = torch.device(device)
        self.realisation = realisation
        self.base_seed = BASE_SEED_REALISATION[realisation]
        self.diffusion_samples = int(diffusion_samples)
        self.cache_limit = int(cache_limit)
        self._prepare = prepare_protein_features
        self._torch = torch
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        started = time.monotonic()
        self.model = EsmFold2Model.from_pretrained(
            self.model_dir, local_files_only=True, dtype=torch.bfloat16
        ).eval().to(self.device)
        self.load_seconds = time.monotonic() - started
        self._cache: dict[str, FoldReading] = {}
        self.folds = 0
        self.cache_hits = 0
        self.fold_seconds = 0.0

    def measurement_record(self) -> dict[str, Any]:
        """What identifies this instrument, for the run record."""

        return {
            "predictor": se.PREDICTOR_ID,
            "model_dir": str(self.model_dir),
            "checkpoint": self.contract,
            "num_diffusion_samples": self.diffusion_samples,
            "seed_rule": SEED_RULE,
            "realisation": self.realisation,
            "base_seed": self.base_seed,
            "dtype": "bfloat16",
            "selection_rule": "highest pTM among the diffusion samples",
            "device": str(self.device),
            "model_loaded_seconds": self.load_seconds,
        }

    def read(self, sequence: str) -> FoldReading:
        """Fold one sequence and read its mean CA-pLDDT, or return the cached reading."""

        if not sequence:
            raise ValueError("an empty sequence has no structure to fold")
        low, high = sr.STRUCTURE_SUPPORT
        if not low <= len(sequence) <= high:
            raise ValueError(
                f"a {len(sequence)}-residue sequence is outside the declared "
                f"structural support {sr.STRUCTURE_SUPPORT}, where a predicted "
                "confidence describes the input rather than a structure; the reward "
                "declaration pays the empty-completion value there and this "
                "instrument declines to fold it. At one residue the model raises an "
                "IndexError, so this is also the guard that keeps a tail outlier "
                "from ending a run"
            )
        cached = self._cache.get(sequence)
        if cached is not None:
            self.cache_hits += 1
            return cached
        torch = self._torch
        seed = se.fold_seed(sequence, base_seed=self.base_seed, rule=SEED_RULE)
        started = time.monotonic()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            features = self._prepare(sequence, device=self.device)
            output = self.model.fold(**features, num_diffusion_samples=self.diffusion_samples)
        summary = se.summarize_esmfold2(output, sequence=sequence)
        elapsed = time.monotonic() - started
        reading = FoldReading(
            sequence=sequence,
            mean_ca_plddt=float(summary["mean_ca_plddt"]),
            ptm=float(summary["ptm"]),
            diffusion_sample_index=int(summary["diffusion_sample_index"]),
            seconds=elapsed,
        )
        del output, features
        self.folds += 1
        self.fold_seconds += elapsed
        if len(self._cache) < self.cache_limit:
            self._cache[sequence] = reading
        return reading

    def read_many(self, sequences: Sequence[str]) -> list[FoldReading | None]:
        """One reading per sequence, ``None`` where the sequence is empty.

        ``None`` for a sequence outside the declared structural support, empty
        sequences included. Those are measured outcomes of the sampler rather than
        missing measurements, the reward declaration decides what they are paid,
        and this instrument declines to invent a structure for them -- which is
        also what stops a single one-residue completion from ending a run of
        thousands of folds.
        """

        low, high = sr.STRUCTURE_SUPPORT
        return [self.read(sequence) if low <= len(sequence) <= high else None
                for sequence in sequences]

    def usage(self) -> dict[str, Any]:
        return {
            "folds": self.folds,
            "cache_hits": self.cache_hits,
            "cached_sequences": len(self._cache),
            "fold_seconds_total": self.fold_seconds,
            "fold_seconds_mean": self.fold_seconds / self.folds if self.folds else None,
        }
