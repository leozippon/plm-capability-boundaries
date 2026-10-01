#!/usr/bin/env python3
"""The uncertainty audit of context information, run over statistics already on disk.

``archive/docs/R5/context-measurement-protocols.md`` is the frozen analysis
plan and this stage is its executable form. It loads no model, touches no GPU and
reaches no cluster: everything below is arithmetic over the per-record sufficient
statistics ``cohort_power.py --record-statistics`` persists beside its report,
which is the whole point of that sidecar. A second sweep of the panel buys nothing
this file cannot compute.

What it produces, in the order the pre-registration declares it.

**A group-level paired interval for I per arm per cohort draw.** The resampling
unit is the near-duplicate group, one cohort draw feeds both terms of
``I = H_baseline - H_model``, the reference is resampled independently and the
smoothed unigram is refitted inside every iteration. All of that lives in
``src.capability.context.information_bootstrap``; this stage's job is to assemble the arm
inputs honestly, and the one thing it must not do is invent a grouping. Group
assignments are not persisted on the base cohort path, so they are recomputed here
from the cohort records the companion ``cohort_*.json`` holds. Where those records
are unavailable the fallback to singleton groups is written into every affected
record as a declared limitation rather than applied quietly, because a
record-level interval is narrowest exactly where group dependence is strongest.

**The identification verdict, from the corrected interval rather than a floor.**
The interval a percentile bootstrap gives here is displaced upward by the reference
refit -- ``-log`` is convex, so resampling the reference lifts the drawn baselines
above the full-reference value -- and a lower-bound rule read off it passes 64 of
120 readings whose true information is zero. ``information_bootstrap`` now measures
that displacement separately and publishes the interval with it removed;
``budget.context_identification`` reads the corrected lower bound, and
``per_arm_identification_status`` is the verdict. The
``budget.SCREENING_CONTEXT_INFORMATION_NATS`` comparison is still reported, as
``screening_status``, and decides nothing here (EXP-R2-221,
``archive/docs/R5/context-measurement-protocols.md``).

**Contrasts that respect where pairing is defined.** Arms sharing one cohort draw
are bootstrapped in one ``bootstrap_arms`` call under common resample indices, so
their contrast is formed inside the iteration. Arms on different cohorts have no
common resampling unit and go through ``unpaired_contrast`` with that fact
attached. The panel does not sit on one cohort -- text arms on OpenWebText,
ProtGPT2 and the residue-level ProGen2 arms on Swiss-Prot, ZymCTRL on the
EC-labelled draw -- so this distinction is the difference between a contrast and a
fiction. The ProGen2-base / ProGen2-medium pair is named explicitly and its sign
is tracked across every draw, because four readings of that gap exist on disk and
one of the four has the opposite sign.

**And a block is a cohort draw, not a file.** Which arms share a resample index is
decided by the two digests every sidecar carries -- the records that were scored,
and the held-out reference they were scored against -- and never by which producer
invocation wrote them down. ``Cohort.digest`` hashes the records themselves, so two
cohort items agreeing on it hold the same records in the same order. Items exist to
make invocations: ``--dtype float32`` is process-global, so ProGen2-medium is
declared as its own item and lands in its own file while scoring the very same
Swiss-Prot draw as ProGen2-base against the very same reference. Keying pairing on
the file would let that split -- an artefact of how the numbers were produced --
decide whether the one contrast the pre-registration names is paired, and would
widen it for a reason the estimand knows nothing about. Every supplied sidecar
carrying one pair of digests is therefore assembled into one block, and where the
record text is supplied the records are checked to be the same records rather than
merely the same hash. A sidecar recording no reference digest is never merged: an
unverifiable reference is not a shared one.

**The three sensitivities the plan asks for.** Between-block spread is reported
beside the within-block interval and never folded into it; below eight blocks it
is reported as a spread with its degrees of freedom and explicitly not as an
interval, which is what the K = 2 pairs on disk have already been misread as. The
alpha sweep is pure CPU here because smoothing is applied at analysis time from
the persisted reference counts, so the vocabulary-dependent smoothing bias whose
published figures were withdrawn as unreproducible is measured rather than quoted,
and carried beside its analytic bound ``log(1 + aV/R)``. The leakage-removed arm
drops every reference record sharing a 30-mer with a cohort record; the sidecar
carries no sequence text, so it runs only when the reference records are supplied
and is emitted as explicitly unavailable otherwise.

**A control whose true information is zero.** ``--unigram-null-control`` adds, to
every block, one synthetic arm per measurable real arm whose predictive
distribution *is* a smoothed unigram fitted on the held-out reference of a
different supplied block. Both terms of ``I`` are then unigram estimates of one
population from independent samples, so the true value is zero by construction
while the measured one carries the estimator's own noise and its
vocabulary-dependent smoothing constant. It is the only point in this artefact
where the eligibility criteria are watched at a value that is known rather than
estimated, and the pre-registration's reason for reporting the sign rule without
adopting it is a prediction about exactly this point. The control joins the
panel of the arm it is built from, under the same resample indices and through
the same ``bootstrap_arms`` call, so there is no second estimator to argue
about; the block supplying its distribution and the block supplying the baseline
are named in every record and a control that cannot be shown to have two
different references is refused rather than reported.

Nothing is omitted silently. An arm whose cohort has too few effective groups is
refused with the reason attached, an arm absent from every sidecar is named as
absent rather than left out of the table, and a sensitivity that could not be
computed says why.

The sign criterion is reported and is **not** evidence. It is expected to pass on
every arm, and the pre-registration explains why: I carries a smoothing bias that
grows with vocabulary size and that no bootstrap can touch, so the sign test is
safe exactly where it is redundant and unreliable exactly where it would decide
something. Every record carrying ``sign_status`` carries that beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.arms import PANEL  # noqa: E402
from src.capability.core.budget import (  # noqa: E402
    LEGACY_FLOOR_NOTE,
    MIN_CONTEXT_INFORMATION_NATS,
    POWER_RECORDS_SCHEMA_VERSION,
    SCREENING_CONTEXT_INFORMATION_NATS,
    SCREENING_FLOOR_NOTE,
    context_identification,
    ratio_denominator_admissibility,
)
from src.capability.context.information_bootstrap import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    ESTIMAND,
    ArmPanel,
    ArmStatistics,
    CohortStatistics,
    ReferenceStatistics,
    SparseCounts,
    bootstrap_arms,
    unigram_null_control,
    unpaired_contrast,
)
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation.near_duplicates import (  # noqa: E402
    NEAR_DUPLICATE_CONTAINMENT,
    near_duplicate_groups,
)
from src.capability.core.measurement_support import SMOOTHING_SWEEP
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

SCHEMA_VERSION = "r2_transfer_context_information_bootstrap_v2"

DEFAULT_OUT = REPO_ROOT / "results/R5/context_information_bootstrap"

#: Modules whose contents decide the numbers in the artefact, hashed into the
#: metadata so that a report can be attributed to the arithmetic that produced it.
#: Repository-relative, because the six no longer share one package directory.
HASHED_MODULES: tuple[str, ...] = (
    "src/capability/context/information_bootstrap.py",
    "src/capability/generation/near_duplicates.py",
    "src/capability/core/budget.py",
    "src/capability/core/measurement_support.py",
    "src/capability/core/statistics.py",
    "src/capability/core/arms.py",
)

#: Added to a panel's seed for its leakage-removed re-run. That panel has a
#: different reference group universe, so its draw is a different draw whatever
#: seed it is given; the offset keeps it from colliding with another panel's
#: seed, which is what would make a contrast between the two neither independent
#: nor paired.
LEAKAGE_SEED_OFFSET = 1_000_003

#: Blocks required before a between-block spread is also reported as an interval.
#: E2's target, and the reason it is eight rather than two: with K = 2 the
#: variance component has one degree of freedom, and the two K = 2 pairs on disk
#: disagree with each other about its size by a factor of fifty.
MIN_BLOCKS_FOR_BETWEEN_BLOCK_INTERVAL = 8

#: k of the k-mer the leakage-removed sensitivity screens the reference on.
DEFAULT_LEAKAGE_KMER = 30

#: Appended to an arm's name to name the null control built on its cohort and
#: its tokenisation. One control per arm rather than one per cohort, because the
#: smoothing constant it carries is a property of the vocabulary and the panel's
#: text cohort alone spans four of them.
CONTROL_SUFFIX = "::unigram-null"

CONTROL_CONSTRUCTION_NOTE = (
    "A synthetic arm whose per-token predictive distribution IS the smoothed "
    "unigram q(v) = (r_src(v) + a) / (R_src + aV) fitted on the held-out "
    "reference of a DIFFERENT cohort block, scored against this block's own "
    "reference as baseline. Both terms of I are then smoothed unigram estimates "
    "of one population fitted on independent samples, so the true I is zero by "
    "construction while the measured I carries the estimator's own sampling "
    "noise and the same vocabulary-dependent smoothing constant every real arm "
    "carries. It is bootstrapped inside the same panel as this block's real "
    "arms, under the same resample indices and through the same estimator; "
    "there is no separate code path for it. One control is built per arm "
    "because the smoothing constant it carries is a property of the "
    "tokenisation, so two arms of one cohort that share a tokenizer produce the "
    "same control and the same reading, by construction and not by agreement."
)

CONTROL_READING_NOTE = (
    "At a known zero the screening floor behaves correctly by REFUSING "
    "the arm and the sign rule behaves correctly by FAILING; a sign PASS here is "
    "the pre-registered failure mode observed, and the measured departure from "
    "zero is the estimator's own bias plus whatever the two reference blocks "
    "differ by. Under a shift between corpus blocks a mismatched unigram costs "
    "more than the matched one, so that term is negative and a control reading "
    "ABOVE zero cannot be attributed to it."
)

SIGN_STATUS_NOTE = (
    "NON-EVIDENTIAL. The sign criterion is reported and not adopted. It is "
    "expected to pass on every arm and must not be read as a gate: I carries a "
    "smoothing bias that grows with vocabulary size and that no bootstrap can "
    "touch, so on an arm whose true context information sits near zero the sign "
    "of the measured I would be decided by the smoothing constant and the "
    "vocabulary rather than by the model. The operative verdict is "
    "per_arm_identification_status, from budget.context_identification, which "
    "reads the DISPLACEMENT-CORRECTED interval; sign_status reads the "
    "uncorrected one and is a different statistic, false-passing on 64 of 120 "
    "readings whose true value is zero. See "
    "archive/docs/R5/context-measurement-protocols.md."
)

PER_ARM_IDENTIFICATION_NOTE = (
    "THE OPERATIVE IDENTIFICATION VERDICT, adopted at EXP-R2-221 and taken from "
    "budget.context_identification: the displacement-corrected 95% interval for "
    "I lies strictly above zero, which is approximately I > 1.96*SE(I) and has "
    "no constant in it. It differs from sign_status only in that the interval it "
    "reads has had the Jensen displacement of L34/L42 removed; sign_status reads "
    "the uncorrected interval and is the rule EXP-R2-218 measured false-passing "
    "on 56 of 112 readings whose true value is zero, and 64 of 120 here"
)

LEGACY_SCREENING_STATUS_NOTE = (
    "screening_status is the PRE-INTERVAL SCREEN, retained as a reporting column "
    "so that every verdict recorded under it stays readable. It decides nothing "
    "here: the operative verdict is per_arm_identification_status. The two "
    "disagree only where a magnitude rule and the reading's own precision "
    "disagree, which is what EXP-R2-221 adopted the second over the first for"
)

SCREENING_NOTE = (
    "Screening, not confirmatory. A criterion here may keep an arm out of a "
    "downstream stage and may say that a denominator is too small to divide by. "
    "It may not be cited as evidence that an arm does or does not extract "
    "information from context, and no number in this artefact may be used to "
    "re-choose the floor it was measured against."
)

COMPARABILITY_NOTE = (
    "Nats per token is not cross-arm comparable and no interval repairs that: a "
    "unigram over merged BPE pieces already encodes the character-level "
    "dependencies inside each piece, so I under two tokenisations is two "
    "estimands. Cross-arm statements are restricted to arms sharing a "
    "tokenisation regime, or are made on relative_information with the caveat "
    "carried."
)

#: Cross-arm contrasts the pre-registration names, with what it says about them.
#: Every other within-cohort pair is still formed -- ``bootstrap_arms`` defaults
#: to all of them -- but these carry their standing note into the record.
DECLARED_CONTRASTS: tuple[dict[str, str], ...] = (
    {
        "left": "progen2-base",
        "right": "progen2-medium",
        "preregistration_note": (
            "The ordering of progen2-base and progen2-medium by context "
            "information is NOT SUPPORTED and must not be reported. The basis is "
            "cohort dependence, not sampling noise: across more than twenty paired "
            "readings in six stages, plain Swiss-Prot puts medium above base in 11 "
            "of 12 readings while the EC-labelled cohort puts base above medium in "
            "6 of 7, so the sign is a property of the arms and the cohort together. "
            "The earlier basis -- four readings, one of opposite sign, none carrying "
            "a valid interval -- is retired: it predated this stage's pairing repair "
            "of 2026-08-21, and the intervals it lacked now exist. Paired within a "
            "draw the gap does exclude zero on all eight EXP-R2-216 blocks, which "
            "qualifies a statement about one cohort and does not lift the retraction."
        ),
    },
)


# --------------------------------------------------------------------------- #
# Reading the sidecar and its companions
# --------------------------------------------------------------------------- #


def _scalar(array: Any) -> Any:
    return np.asarray(array).item()


def select_records(counts: SparseCounts, keep: np.ndarray) -> SparseCounts:
    """``counts`` restricted to the records ``keep`` selects, offsets rebuilt.

    Used twice: to drop reference records carrying no scored target, and to drop
    the ones the leakage screen removes. Both are removals of whole records from
    the resampling population, and both are counted into the artefact.
    """

    keep = np.asarray(keep, dtype=bool)
    if keep.size != counts.n_records:
        raise ValueError("the selection mask does not align with the records")
    starts = counts.record_offsets[:-1][keep]
    stops = counts.record_offsets[1:][keep]
    lengths = (stops - starts).astype(np.int64)
    entries = (
        np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in zip(starts, stops)])
        if lengths.size and int(lengths.sum())
        else np.zeros(0, dtype=np.int64)
    )
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
    return SparseCounts(
        unique_token_ids=counts.unique_token_ids[entries],
        counts=counts.counts[entries],
        record_offsets=offsets,
    )


def cohort_json_for(sidecar: Path) -> Path:
    """The frozen-cohort file ``cohort_power.py`` writes beside the sidecar.

    Derived rather than searched for: the producing stage names the two files
    from one cohort name and one digest prefix, so the mapping is a rename, and a
    reader that has to guess is a reader that can pick up another cohort's
    records.
    """

    name = sidecar.name
    if not name.startswith("power_") or not name.endswith(".records.npz"):
        raise ValueError(
            f"{sidecar} is not named power_<cohort>_<digest>.records.npz, so the "
            "companion cohort file cannot be derived; pass --cohort-json"
        )
    stem = name[len("power_") : -len(".records.npz")]
    return sidecar.parent / f"cohort_{stem}.json"


def read_record_strings(path: Path) -> list[str]:
    """A record list from a frozen-cohort file or from a bare JSON list."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records")
    if not isinstance(records, list) or not all(isinstance(r, str) for r in records):
        raise ValueError(f"{path} carries no list of record strings")
    return records


def grouping_of(
    records: list[str] | None,
    *,
    kind: str,
    containment: float,
    shingle: int | None,
    what: str,
    unavailable_reason: str | None,
    n_records: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Near-duplicate groups over ``records``, or singletons with the reason said.

    The fallback is not a quiet one. A singleton grouping resamples records, and
    a record-level interval is *narrower* than a group-level one exactly where
    the group structure matters, so an artefact that fell back without saying so
    would report its most confident-looking number in the case the group unit
    exists to catch.
    """

    if records is not None:
        unit = "residues" if kind == "protein" else "characters"
        groups, summary = near_duplicate_groups(
            records, unit=unit, containment=containment, shingle=shingle
        )
        return groups, {
            "available": True,
            "what": what,
            "computed_from": (
                "the persisted records, by near_duplicates.near_duplicate_groups"
            ),
            **summary,
        }
    return np.arange(max(int(n_records), 0), dtype=np.int64), {
        "available": False,
        "what": what,
        "fallback": "singleton groups, one record per resampling unit",
        "reason": unavailable_reason,
        "declared_limitation": (
            f"DECLARED LIMITATION: the {what} records were not available to this "
            "re-analysis, so each record was resampled as its own unit. "
            "Near-duplicate structure is therefore ignored, the interval is "
            "narrower than the evidence supports, and it is not an interval at "
            "the unit of dependence the pre-registration freezes. It is reported "
            "so that the fallback is visible, not so that it can be quoted."
        ),
        "n_records": int(n_records),
        "n_groups": int(n_records),
    }


@dataclass(frozen=True)
class ArmInput:
    """One arm of one cohort block, assembled or explicitly refused.

    The last three fields are provenance rather than statistics. A block is
    assembled from every sidecar addressing its draw, so which cohort item
    declared this arm, in which file, under which smoothing constant, is a fact
    about the arm and not about its block.
    """

    name: str
    statistics: ArmStatistics | None
    load_refusal: str | None
    vocab_size: int
    reference_tokens: int
    reference_records_dropped_empty: int
    reference_keep: np.ndarray | None
    cohort_name: str
    sidecar: Path
    smoothing: float


@dataclass(frozen=True)
class Sidecar:
    """One producer invocation's file, and what it declares about its own run."""

    path: Path
    sha256: str
    cohort_name: str
    smoothing: float
    max_len: int
    producer_seeds: dict[str, int]
    arms: tuple[str, ...]

    @property
    def record(self) -> dict[str, Any]:
        return {
            "sidecar": str(self.path),
            "sidecar_sha256": self.sha256,
            "cohort_name": self.cohort_name,
            "smoothing": self.smoothing if math.isfinite(self.smoothing) else None,
            "max_len": self.max_len,
            "producer_seeds": dict(self.producer_seeds),
            "arms": list(self.arms),
        }


@dataclass(frozen=True)
class Block:
    """One cohort draw: its records, its reference, and every arm scored on them.

    Identified by the two digests and assembled from every sidecar that carries
    them, which is usually one file and is not always one: a cohort item is the
    unit of *production*, and two items can score one draw. The unit of
    *resampling* is this block, and it is what pairing is defined on.
    """

    index: int
    sidecars: tuple[Sidecar, ...]
    cohort_digest: str
    cohort_kind: str
    reference_digest: str | None
    arms: list[ArmInput]
    cohort_records: list[str] | None
    reference_records: list[str] | None
    grouping: dict[str, Any]
    reference_grouping: dict[str, Any]

    @property
    def block_id(self) -> str:
        return f"b{self.index}"

    @property
    def cohort_names(self) -> list[str]:
        """Every cohort item declaring these records, in the order supplied."""

        return list(dict.fromkeys(sidecar.cohort_name for sidecar in self.sidecars))


def draw_key(sidecar: Path) -> tuple[str, str]:
    """What decides whether two sidecars address the same resampling units.

    The cohort digest and the reference digest, and nothing else -- not the
    cohort item's name, and not the file. Those two hashes are the identity of
    the records that would be resampled and of the reference the baseline would
    be refitted on, which is the whole of what a common resample index has to
    address.

    A sidecar recording no reference digest keys on its own path instead, so it
    is never merged with anything: an unverifiable reference is not a shared
    one, and pairing arms whose baselines cannot be shown to be the same sample
    is exactly the mistake this key exists to make impossible.
    """

    with np.load(sidecar) as npz:
        cohort = str(_scalar(npz["cohort_digest"]))
        reference = str(_scalar(npz["reference_digest"]))
    return cohort, reference or f"unverifiable-reference::{sidecar}"


def load_block(
    index: int,
    entries: Sequence[tuple[Path, Path | None, Path | None]],
    *,
    requested_arms: list[str] | None,
    containment: float,
    shingle: int | None,
) -> Block:
    """Assemble one cohort draw's sidecars into arm inputs, refusing rather than repairing.

    ``entries`` are the ``(sidecar, cohort_json, reference_json)`` triples of one
    draw -- every supplied sidecar carrying the same pair of digests, which is
    what :func:`draw_key` groups on. They are checked to agree on those digests
    and, wherever the record text was supplied, on the records themselves: the
    digest already implies the records, and the check is what makes the artefact
    rest on the implication rather than assume it. An arm scored by two of them
    is refused by name, because two readings of one arm on one draw are two
    measurements and not one panel arm.
    """

    if not entries:
        raise ValueError("a block needs at least one sidecar")

    sidecars: list[Sidecar] = []
    payloads: list[tuple[Sidecar, Any, list[str]]] = []
    digests: set[tuple[str, str]] = set()
    cohort_digest, reference_digest = "", None
    cohort_kind = "protein"
    cohort_records: list[str] | None = None
    reference_records: list[str] | None = None
    missing_cohort_json: list[Path] = []

    for sidecar, cohort_json, reference_json in entries:
        npz = np.load(sidecar)
        version = str(_scalar(npz["schema_version"]))
        if version != POWER_RECORDS_SCHEMA_VERSION:
            raise ValueError(
                f"{sidecar} declares schema {version!r} against this stage's "
                f"{POWER_RECORDS_SCHEMA_VERSION!r}; the field list is the whole of the "
                "sidecar's meaning, so rebuild rather than reinterpret"
            )
        cohort_digest = str(_scalar(npz["cohort_digest"]))
        reference_digest = str(_scalar(npz["reference_digest"])) or None
        digests.add((cohort_digest, reference_digest or ""))
        if len(digests) != 1:
            raise ValueError(
                f"{sidecar} was written against cohort {cohort_digest[:12]} and "
                f"reference {str(reference_digest)[:12]}, which is not the draw the "
                "sidecars beside it address; one block is one cohort draw, and a "
                "common resample index over two draws addresses neither"
            )
        present = [str(name) for name in np.asarray(npz["arms"]).tolist()]
        names = (
            present if requested_arms is None else [n for n in present if n in requested_arms]
        )

        item_name, item_kind = "unknown", cohort_kind
        path = cohort_json if cohort_json is not None else cohort_json_for(sidecar)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                stored = payload.get("cohort_digest")
                if stored is not None and stored != cohort_digest:
                    raise ValueError(
                        f"{path} holds cohort {str(stored)[:12]} and {sidecar} was "
                        f"written against {cohort_digest[:12]}; grouping one cohort's "
                        "records and resampling another's statistics is not a grouping"
                    )
                item_name = str(payload.get("cohort_name", item_name))
                item_kind = str(payload.get("cohort_kind", item_kind))
            records = read_record_strings(path)
            if cohort_records is not None and records != cohort_records:
                raise ValueError(
                    f"{path} and the frozen cohort supplied beside it agree on digest "
                    f"{cohort_digest[:12]} and hold different records; that digest is "
                    "the identity a common resample index rests on"
                )
            cohort_records, cohort_kind = records, item_kind
        else:
            missing_cohort_json.append(path)

        if reference_json is not None:
            if not reference_json.is_file():
                raise ValueError(f"--reference-json names a missing file: {reference_json}")
            records = read_record_strings(reference_json)
            if reference_records is not None and records != reference_records:
                raise ValueError(
                    f"{reference_json} and the reference supplied beside it agree on "
                    f"digest {str(reference_digest)[:12]} and hold different records; "
                    "that digest is the identity the shared baseline rests on"
                )
            reference_records = records

        provenance = Sidecar(
            path=sidecar,
            sha256=sha256_file(sidecar),
            cohort_name=item_name,
            smoothing=float(_scalar(npz["smoothing"])),
            max_len=int(_scalar(npz["max_len"])),
            producer_seeds=dict(
                zip(
                    [str(name) for name in np.asarray(npz["seed_names"]).tolist()],
                    [int(value) for value in np.asarray(npz["seed_values"]).tolist()],
                )
            ),
            arms=tuple(names),
        )
        sidecars.append(provenance)
        payloads.append((provenance, npz, names))

    cohort_unavailable: str | None = (
        None
        if cohort_records is not None
        else (
            "the companion frozen-cohort files ("
            + ", ".join(str(path) for path in missing_cohort_json)
            + ") do not exist, and a sidecar carries no sequence text of its own"
        )
    )
    reference_unavailable: str | None = (
        None
        if reference_records is not None
        else (
            "no --reference-json was supplied for this block's sidecars; a sidecar "
            "carries only order-free token counts, so the reference text has to come "
            "from the reference_<name>_<digest>.json cohort_power.py writes"
        )
    )

    n_cohort_records = (
        len(cohort_records)
        if cohort_records is not None
        else int(
            max(
                (
                    int(npz[f"{name}::record_index"].max()) + 1
                    for _, npz, names in payloads
                    for name in names
                ),
                default=0,
            )
        )
    )
    cohort_groups, grouping = grouping_of(
        cohort_records,
        kind=cohort_kind,
        containment=containment,
        shingle=shingle,
        what="cohort",
        unavailable_reason=cohort_unavailable,
        n_records=n_cohort_records,
    )

    n_reference_records = 0
    for _, npz, names in payloads:
        for name in names:
            key = f"{name}::reference_token_count"
            if key in npz.files:
                n_reference_records = max(n_reference_records, int(npz[key].size))
    if reference_records is not None and len(reference_records) != n_reference_records:
        raise ValueError(
            f"the supplied reference holds {len(reference_records)} records against "
            f"the sidecars' {n_reference_records} reference rows; the two do not "
            "describe the same reference set"
        )
    reference_groups, reference_grouping = grouping_of(
        reference_records,
        kind=cohort_kind,
        containment=containment,
        shingle=shingle,
        what="reference",
        unavailable_reason=reference_unavailable,
        n_records=n_reference_records,
    )

    arms: list[ArmInput] = []
    scored_by: dict[str, Path] = {}
    for provenance, npz, names in payloads:
        for name in names:
            if name in scored_by:
                raise ValueError(
                    f"{name} is scored by both {scored_by[name]} and {provenance.path}, "
                    "which address one cohort draw between them. Two readings of one "
                    "arm on one draw are two measurements, not one panel arm; supply "
                    "the one that is to be reported"
                )
            scored_by[name] = provenance.path
            arms.append(
                build_arm_input(npz, name, cohort_groups, reference_groups, provenance)
            )

    return Block(
        index=index,
        sidecars=tuple(sidecars),
        cohort_digest=cohort_digest,
        cohort_kind=cohort_kind,
        reference_digest=reference_digest,
        arms=arms,
        cohort_records=cohort_records,
        reference_records=reference_records,
        grouping=grouping,
        reference_grouping=reference_grouping,
    )


def build_arm_input(
    npz: Any,
    name: str,
    cohort_groups: np.ndarray,
    reference_groups: np.ndarray,
    provenance: Sidecar,
) -> ArmInput:
    """One arm's ``ArmStatistics``, or the reason it cannot have one."""

    smoothing = provenance.smoothing
    prefix = f"{name}::"
    vocab_size = int(_scalar(npz[f"{prefix}vocab_size"]))
    if f"{prefix}reference_counts_offsets" not in npz.files:
        return ArmInput(
            name=name,
            statistics=None,
            load_refusal=(
                "the sidecar carries no reference counts for this arm, so the "
                "held-out unigram this estimand is defined against does not exist "
                "in it. The plug-in baseline is a different estimand and is not "
                "substituted here"
            ),
            vocab_size=vocab_size,
            reference_tokens=0,
            reference_records_dropped_empty=0,
            reference_keep=None,
            cohort_name=provenance.cohort_name,
            sidecar=provenance.path,
            smoothing=smoothing,
        )
    if not math.isfinite(smoothing):
        return ArmInput(
            name=name,
            statistics=None,
            load_refusal=(
                "the sidecar records no smoothing constant, which its writer does "
                "only when no reference entered any figure; the baseline cannot be "
                "refitted without it"
            ),
            vocab_size=vocab_size,
            reference_tokens=0,
            reference_records_dropped_empty=0,
            reference_keep=None,
            cohort_name=provenance.cohort_name,
            sidecar=provenance.path,
            smoothing=smoothing,
        )

    record_index = np.asarray(npz[f"{prefix}record_index"], dtype=np.int64)
    if record_index.size and int(record_index.max()) >= cohort_groups.size:
        raise ValueError(
            f"{name}: the sidecar scores cohort record {int(record_index.max())} "
            f"but the grouping covers {cohort_groups.size} records"
        )
    cohort = CohortStatistics(
        clean_nll_sum=npz[f"{prefix}clean_nll_sum"],
        token_count=npz[f"{prefix}token_count"],
        n_symbols=npz[f"{prefix}n_symbols"],
        targets=SparseCounts(
            unique_token_ids=npz[f"{prefix}unique_token_ids"],
            counts=npz[f"{prefix}counts"],
            record_offsets=npz[f"{prefix}counts_offsets"],
        ),
        group_id=cohort_groups[record_index],
    )

    reference_targets = SparseCounts(
        unique_token_ids=npz[f"{prefix}reference_unique_token_ids"],
        counts=npz[f"{prefix}reference_counts"],
        record_offsets=npz[f"{prefix}reference_counts_offsets"],
    )
    stored_tokens = np.asarray(npz[f"{prefix}reference_token_count"], dtype=np.int64)
    if not np.array_equal(reference_targets.record_totals, stored_tokens):
        raise ValueError(
            f"{name}: the sidecar's reference token counts disagree with the counts "
            "they are supposed to sum to"
        )
    if stored_tokens.size != reference_groups.size:
        raise ValueError(
            f"{name}: {stored_tokens.size} reference rows against a grouping over "
            f"{reference_groups.size} records"
        )
    keep = stored_tokens > 0
    if not bool(keep.any()):
        return ArmInput(
            name=name,
            statistics=None,
            load_refusal="every reference record carries zero scored targets",
            vocab_size=vocab_size,
            reference_tokens=0,
            reference_records_dropped_empty=int(stored_tokens.size),
            reference_keep=None,
            cohort_name=provenance.cohort_name,
            sidecar=provenance.path,
            smoothing=smoothing,
        )
    return ArmInput(
        name=name,
        statistics=ArmStatistics(
            name=name,
            cohort=cohort,
            reference=ReferenceStatistics(
                token_count=stored_tokens[keep],
                targets=select_records(reference_targets, keep),
                group_id=reference_groups[keep],
            ),
            vocab_size=vocab_size,
            smoothing=smoothing,
        ),
        load_refusal=None,
        vocab_size=vocab_size,
        reference_tokens=int(stored_tokens.sum()),
        reference_records_dropped_empty=int((~keep).sum()),
        reference_keep=keep,
        cohort_name=provenance.cohort_name,
        sidecar=provenance.path,
        smoothing=smoothing,
    )


# --------------------------------------------------------------------------- #
# The null control: an arm whose true context information is zero
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NullControl:
    """One block's control for one arm, assembled or explicitly refused."""

    name: str
    block_id: str
    arm: str
    statistics: ArmStatistics | None
    record: dict[str, Any]


def control_source(
    block: Block, arm: str, blocks: list[Block]
) -> tuple[Block, ArmInput] | str:
    """The next supplied block that can supply an independent unigram, or why none can.

    Independence is the whole content of the control, so it is established from
    the artefacts rather than assumed: the source must be a different block,
    must carry this arm -- which is what makes its counts counts over the same
    inventory -- and must have been scored against a *different* held-out
    reference, named by digest. A missing digest is not treated as a different
    one; it is an unverifiable claim, and an unverifiable independence is what
    this control cannot survive.

    The search starts after this block and wraps, rather than starting at the
    first supplied block, so that K blocks of one arm draw their controls from K
    different references instead of all from the first. Each control is a valid
    null either way; what the rotation buys is that the K readings are near
    replicates rather than K comparisons against one and the same sample, whose
    spread would understate what a fresh source would give.
    """

    shared: list[str] = []
    unverifiable: list[str] = []
    for candidate in blocks[block.index + 1 :] + blocks[: block.index]:
        if candidate.block_id == block.block_id:
            continue
        entry = next(
            (
                item
                for item in candidate.arms
                if item.name == arm and item.statistics is not None
            ),
            None,
        )
        if entry is None:
            continue
        if block.reference_digest is None or candidate.reference_digest is None:
            unverifiable.append(candidate.block_id)
            continue
        if candidate.reference_digest == block.reference_digest:
            shared.append(candidate.block_id)
            continue
        return candidate, entry
    if shared:
        return (
            f"every other supplied block carrying {arm} was scored against the "
            f"same held-out reference as {block.block_id} "
            f"({str(block.reference_digest)[:12]}): {', '.join(shared)}. A control "
            "fitted on that reference is fitted on the baseline's own "
            "distribution, so its I would be zero identically rather than by "
            "measurement and its interval would describe nothing"
        )
    if unverifiable:
        return (
            f"the candidate source blocks for {arm} ({', '.join(unverifiable)}) or "
            f"{block.block_id} itself record no reference digest, so the two "
            "references cannot be shown to be different samples. The control's "
            "true value is zero only if they are"
        )
    return (
        f"no other supplied block carries {arm}, so no reference over this arm's "
        "inventory is available from an independent cohort block"
    )


def exact_overlap(left: list[str] | None, right: list[str] | None) -> int | None:
    """Records held by both sets, by exact content, or ``None`` if unknowable."""

    if left is None or right is None:
        return None
    return len(set(left) & set(right))


def cohort_copy_mask(
    cohort: list[str] | None, reference: list[str] | None
) -> np.ndarray | None:
    """Which reference records are verbatim copies of a scored cohort record.

    ``cohort_power.py`` already asserts this disjointness between a cohort
    and its *own* held-out reference, so applying it to the reference a control
    borrows from another block is not a new rule but the same one, reaching the
    one set it could not reach. It is what the control's zero rests on: a
    unigram fitted on the records it is scored against is not a null but a
    memoriser, and its ``I`` would be positive for a reason that has nothing to
    do with the criteria being watched. The copies are dropped rather than made
    grounds for refusal, because dropping them is what restores the property
    while refusing merely reports its absence.
    """

    if cohort is None or reference is None:
        return None
    scored = set(cohort)
    return np.asarray([record in scored for record in reference], dtype=bool)


def null_controls(block: Block, blocks: list[Block]) -> list[NullControl]:
    """One control per measurable arm of ``block``, or the reason it has none."""

    controls: list[NullControl] = []
    for entry in block.arms:
        if entry.statistics is None:
            continue
        arm = entry.statistics
        name = f"{entry.name}{CONTROL_SUFFIX}"
        common: dict[str, Any] = {
            "control_arm": name,
            "arm": entry.name,
            "block_id": block.block_id,
            "cohort_name": entry.cohort_name,
            "cohort_kind": block.cohort_kind,
            "cohort_digest": block.cohort_digest,
            "baseline_block_id": block.block_id,
            "baseline_reference_digest": block.reference_digest,
            "baseline_reference_tokens": int(arm.reference.token_count.sum()),
            "vocab_size": int(arm.vocab_size),
            "smoothing": float(arm.smoothing),
            "true_information_nats": 0.0,
            "construction": CONTROL_CONSTRUCTION_NOTE,
        }
        found = control_source(block, entry.name, blocks)
        if isinstance(found, str):
            controls.append(
                NullControl(
                    name=name,
                    block_id=block.block_id,
                    arm=entry.name,
                    statistics=None,
                    record={
                        **common,
                        "available": False,
                        "independent": False,
                        "refusal_reason": found,
                        "control_block_id": None,
                        "control_reference_digest": None,
                        "control_reference_tokens": None,
                    },
                )
            )
            continue
        source_block, source_entry = found
        source_reference = source_entry.statistics.reference
        copies = cohort_copy_mask(block.cohort_records, source_block.reference_records)
        n_copies: int | None = None
        if copies is not None:
            # ``copies`` indexes every sidecar row; the reference statistics hold
            # only the rows that carried a scored target, so the mask is taken
            # through the same selection before it is applied.
            keep = ~copies[source_entry.reference_keep]
            n_copies = int((~keep).sum())
            if not bool(keep.any()):
                controls.append(
                    NullControl(
                        name=name,
                        block_id=block.block_id,
                        arm=entry.name,
                        statistics=None,
                        record={
                            **common,
                            "available": False,
                            "independent": False,
                            "refusal_reason": (
                                "every record of "
                                f"{source_block.block_id}'s held-out reference "
                                "appears verbatim in the scored cohort, so no "
                                "disjoint sample survives the screen and the "
                                "control would be fitted on its own targets"
                            ),
                            "control_block_id": source_block.block_id,
                            "control_reference_digest": source_block.reference_digest,
                            "control_reference_tokens": None,
                            "cohort_copies_dropped_from_control_reference": n_copies,
                        },
                    )
                )
                continue
            if n_copies:
                source_reference = ReferenceStatistics(
                    token_count=source_reference.token_count[keep],
                    targets=select_records(source_reference.targets, keep),
                    group_id=source_reference.group_id[keep],
                )
        control_tokens = float(source_reference.token_count.sum())
        baseline_tokens = float(arm.reference.token_count.sum())
        pseudo = arm.smoothing * arm.vocab_size
        controls.append(
            NullControl(
                name=name,
                block_id=block.block_id,
                arm=entry.name,
                statistics=unigram_null_control(arm, source_reference, name=name),
                record={
                    **common,
                    "available": True,
                    "independent": True,
                    "refusal_reason": None,
                    "control_block_id": source_block.block_id,
                    "control_cohort_name": source_entry.cohort_name,
                    "control_reference_digest": source_block.reference_digest,
                    "control_reference_tokens": int(control_tokens),
                    "control_reference_records": int(source_reference.token_count.size),
                    # The two facts the zero rests on, counted rather than
                    # assumed. The first is how many verbatim copies of the
                    # scored cohort were removed from the borrowed reference
                    # before it was fitted; the second is reported because
                    # reference sets that share records give correlated
                    # unigrams, which shrinks the departure and flatters the
                    # null.
                    "cohort_copies_dropped_from_control_reference": n_copies,
                    "control_reference_records_before_screen": int(
                        source_entry.statistics.reference.token_count.size
                    ),
                    "baseline_control_reference_exact_overlap": exact_overlap(
                        block.reference_records, source_block.reference_records
                    ),
                    "exact_overlap_checked": n_copies is not None,
                    "exact_overlap_limitation": (
                        None
                        if n_copies is not None
                        else (
                            "DECLARED LIMITATION: the record text of one of the "
                            "two sets was not supplied, so the control's "
                            "reference could not be shown to exclude the scored "
                            "cohort by exact content. Differing reference "
                            "digests establish that the two references are not "
                            "the same set and nothing more. Pass --reference-json "
                            "for every sidecar to close this"
                        )
                    ),
                    # log(1 + aV/R) bounds how far additive smoothing can push a
                    # cross-entropy above its unsmoothed value. It applies to both
                    # terms of this control, so the departure the control measures
                    # is bounded by the larger of the two rather than by their sum,
                    # and the two are reported so that neither has to be guessed.
                    "analytic_smoothing_bound_nats": {
                        "baseline": math.log1p(pseudo / baseline_tokens),
                        "control": math.log1p(pseudo / control_tokens),
                    },
                },
            )
        )
    return controls


# --------------------------------------------------------------------------- #
# Panels: where a common resample index is defined, and where it is not
# --------------------------------------------------------------------------- #


def panel_signature(arm: ArmStatistics) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The group universes a common resample index has to address."""

    return (
        tuple(int(value) for value in np.unique(arm.cohort.group_id)),
        tuple(int(value) for value in np.unique(arm.reference.group_id)),
    )


@dataclass(frozen=True)
class Panel:
    """Arms of one cohort block sharing one resample index, and their result."""

    panel_id: str
    block: Block
    seed: int
    statistics: list[ArmStatistics]
    result: ArmPanel

    @property
    def names(self) -> list[str]:
        return [arm.name for arm in self.statistics]


def partition_block(
    block: Block, extra: list[ArmStatistics] | None = None
) -> list[list[ArmStatistics]]:
    """Arms of one block grouped by the universe a common index would address.

    Normally one partition: every arm of a cohort scores the same records, so the
    group universes coincide. They can fail to when an arm's tokenisation leaves a
    record with no scored target, and then a common index would not address the
    same units. ``bootstrap_arms`` refuses that outright, so the partition is made
    here and the arms that fall out of the main one are bootstrapped separately
    and reported as not contrastable against it -- rather than being handed to
    ``unpaired_contrast``, which is right for independent cohorts and wrong for
    arms that share records.
    """

    partitions: dict[tuple[tuple[int, ...], tuple[int, ...]], list[ArmStatistics]] = {}
    for statistics in [
        entry.statistics for entry in block.arms if entry.statistics is not None
    ] + list(extra or []):
        partitions.setdefault(panel_signature(statistics), []).append(statistics)
    return list(partitions.values())


def bootstrap_block(
    block: Block,
    first_seed: int,
    *,
    n_bootstrap: int,
    confidence: float,
    extra: list[ArmStatistics] | None = None,
) -> list[Panel]:
    """Every panel of one block, each under its own seed.

    ``extra`` arms join the panel of the real arms they share a group universe
    with. They do not perturb the real arms' numbers: ``bootstrap_arms`` draws
    the cohort and reference multiplicities before it visits any arm, so adding
    one leaves every other arm's draws identical.
    """

    panels: list[Panel] = []
    for ordinal, statistics in enumerate(partition_block(block, extra)):
        seed = first_seed + ordinal
        panels.append(
            Panel(
                panel_id=f"{block.block_id}p{ordinal}",
                block=block,
                seed=seed,
                statistics=statistics,
                result=bootstrap_arms(
                    statistics,
                    seed=seed,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                ),
            )
        )
    return panels


# --------------------------------------------------------------------------- #
# Per-arm rows
# --------------------------------------------------------------------------- #


def load_refusal_record(name: str, reason: str) -> dict[str, Any]:
    """The shape ``arm_row`` reads, for an arm that never reached the bootstrap."""

    return {
        "arm": name,
        "refused": True,
        "refusal_reason": reason,
        "statistics": None,
        "seed": None,
        "n_bootstrap": None,
        "confidence": None,
        "resampling_unit": "group",
        "unit_floor": None,
        "diagnostics": None,
        "cohort_draw_shared_between_terms": True,
        "reference_resampled": True,
    }


def arm_row(
    block: Block,
    source: ArmInput,
    panel_id: str | None,
    record: dict[str, Any],
    *,
    threshold: float,
    is_control: bool = False,
) -> dict[str, Any]:
    """One arm on one cohort draw: the estimate, its interval and how to read it.

    ``source`` is the arm's own input -- for a null control, the input of the arm
    it was built from -- because a block can be assembled from several sidecars
    and the file, the cohort item and the smoothing constant are then facts about
    the arm rather than about the block.
    """

    statistics = record["statistics"]
    information = statistics["information_nats_per_token"] if statistics else None
    relative = statistics["relative_information"] if statistics else None
    bits = statistics["information_bits_per_symbol"] if statistics else None
    diagnostics = record.get("diagnostics") or {}

    point = None if information is None else float(information["point"])
    low, high = (None, None) if information is None else information["interval"]
    if information is None:
        screening, screening_interval, sign = "REFUSED", "REFUSED", "REFUSED"
        disagreement = None
        clears_legacy = None
    else:
        screening = "PASS" if point >= threshold else "FAIL"
        screening_interval = (
            "above"
            if low >= threshold
            else "below"
            if high < threshold
            else "straddles"
        )
        sign = "PASS" if low > 0.0 else "FAIL"
        disagreement = screening != sign
        clears_legacy = bool(point >= MIN_CONTEXT_INFORMATION_NATS)
    corrected = (
        None if information is None else information["displacement_corrected_interval"]
    )
    identification = (
        None if corrected is None else context_identification(point, corrected[0])
    )
    per_arm = "REFUSED" if identification is None else identification["verdict"]
    return {
        "block_id": block.block_id,
        "panel_id": panel_id,
        "arm": record["arm"],
        "is_unigram_null_control": bool(is_control),
        "sidecar": str(source.sidecar),
        "cohort_digest": block.cohort_digest,
        "cohort_name": source.cohort_name,
        "cohort_kind": block.cohort_kind,
        "reference_digest": block.reference_digest,
        "smoothing": source.smoothing if math.isfinite(source.smoothing) else None,
        "refused": bool(record["refused"]),
        "refusal_reason": record["refusal_reason"],
        "seed": record["seed"],
        "n_bootstrap": record["n_bootstrap"],
        "confidence": record["confidence"],
        "resampling_unit": record["resampling_unit"],
        "cohort_draw_shared_between_terms": record["cohort_draw_shared_between_terms"],
        "reference_resampled": record["reference_resampled"],
        "context_information_nats": point,
        "bootstrap_ci_95": None if information is None else list(information["interval"]),
        "bootstrap_fraction_nonpositive": (
            None
            if information is None
            else 1.0 - float(information["fraction_of_draws_positive"])
        ),
        "bootstrap_se": None if information is None else information["bootstrap_se"],
        "bootstrap_bias": None if information is None else information["bootstrap_bias"],
        "median_bias_z0": None if information is None else information["median_bias_z0"],
        "interval_mc_se": None if information is None else information["interval_mc_se"],
        "reference_resampling_displacement": (
            None
            if information is None
            else information["reference_resampling_displacement"]
        ),
        "displacement_corrected_ci_95": None if corrected is None else list(corrected),
        "cohort_only_bootstrap_se": (
            None if information is None else information["cohort_only_bootstrap_se"]
        ),
        "reference_held_fixed_ci_95": (
            None
            if information is None
            else list(information["reference_held_fixed_interval"])
        ),
        "per_arm_identification_status": per_arm,
        "per_arm_identification_note": PER_ARM_IDENTIFICATION_NOTE,
        # The criterion's own record, so a reading carries what decided it.
        "context_identification": identification,
        "relative_information": None if relative is None else relative["point"],
        "relative_information_ci_95": None if relative is None else list(relative["interval"]),
        "information_bits_per_symbol": None if bits is None else bits["point"],
        "information_bits_per_symbol_ci_95": (
            None if bits is None else list(bits["interval"])
        ),
        "n_scored_records": diagnostics.get("n_records"),
        "n_scored_tokens": diagnostics.get("n_scored_tokens"),
        "n_reference_records": diagnostics.get("n_reference_records"),
        "n_reference_tokens": diagnostics.get("n_reference_tokens"),
        "n_groups": diagnostics.get("n_groups"),
        "n_effective_groups": diagnostics.get("n_effective_groups"),
        "largest_group_token_share": diagnostics.get("largest_group_token_share"),
        "n_singleton_groups": diagnostics.get("n_singleton_groups"),
        "top10_record_token_share": diagnostics.get("top10_record_token_share"),
        "cohort_token_share_unseen_in_reference": diagnostics.get(
            "cohort_token_share_unseen_in_reference"
        ),
        "cohort_token_share_reference_count_at_most_5": diagnostics.get(
            "cohort_token_share_reference_count_at_most_5"
        ),
        "reference_n_effective_groups": diagnostics.get("reference_n_effective_groups"),
        "screening_threshold_nats": float(threshold),
        "screening_status": screening,
        "screening_status_is_operative": False,
        "screening_status_note": LEGACY_SCREENING_STATUS_NOTE,
        "screening_floor_note": SCREENING_FLOOR_NOTE,
        "screening_interval_status": screening_interval,
        # The retired constant as a reporting column, so a reading here stays
        # comparable with the artefacts recorded under it. It decides nothing.
        "legacy_minimum_context_information_nats": MIN_CONTEXT_INFORMATION_NATS,
        "clears_legacy_floor": clears_legacy,
        "legacy_floor_note": LEGACY_FLOOR_NOTE,
        "sign_status": sign,
        "sign_status_is_evidential": False,
        "sign_status_note": SIGN_STATUS_NOTE,
        "status_disagreement": disagreement,
        "unit_floor": record["unit_floor"],
        "diagnostics": record["diagnostics"],
        "cohort_grouping": block.grouping,
        "reference_grouping": block.reference_grouping,
        "statistics": statistics,
        "comparability_note": COMPARABILITY_NOTE,
    }


# --------------------------------------------------------------------------- #
# Contrasts
# --------------------------------------------------------------------------- #


def declared_note(left: str, right: str) -> str | None:
    """The pre-registration's standing note on a named pair, if it has one."""

    for entry in DECLARED_CONTRASTS:
        if {entry["left"], entry["right"]} == {left, right}:
            return entry["preregistration_note"]
    return None


def within_cohort_contrasts(panels: list[Panel]) -> list[dict[str, Any]]:
    """Every pair of arms sharing a cohort draw, differenced inside the iteration."""

    rows: list[dict[str, Any]] = []
    for panel in panels:
        for key, record in panel.result.contrasts.items():
            rows.append(
                {
                    "key": key,
                    "block_id": panel.block.block_id,
                    "panel_id": panel.panel_id,
                    "cohort_digest": panel.block.cohort_digest,
                    "cohort_names": panel.block.cohort_names,
                    "seed": panel.seed,
                    "preregistration_note": declared_note(record["left"], record["right"]),
                    **record,
                }
            )
    return rows


def cross_cohort_contrasts(
    panels: list[Panel], *, confidence: float, requested: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Differences between arms on different cohort draws, labelled unpaired.

    Restricted to the pairs that mean something: the same arm on two draws, which
    is the block sensitivity E2 is about; the pairs the pre-registration names;
    and whatever ``--contrast`` asks for. An exhaustive cross-cohort product would
    be thousands of intervals that are each conservative by an unknown amount and
    none of which anybody asked for.
    """

    wanted = {frozenset(pair) for pair in requested}
    rows: list[dict[str, Any]] = []
    for left_panel, right_panel in combinations(panels, 2):
        if left_panel.block.block_id == right_panel.block.block_id:
            continue
        for left in left_panel.names:
            for right in right_panel.names:
                note = declared_note(left, right)
                if not (left == right or note or frozenset((left, right)) in wanted):
                    continue
                record = unpaired_contrast(
                    left_panel.result.arms[left],
                    right_panel.result.arms[right],
                    confidence=confidence,
                )
                rows.append(
                    {
                        "key": f"{left}@{left_panel.block.block_id}"
                        f"_minus_{right}@{right_panel.block.block_id}",
                        "left_block_id": left_panel.block.block_id,
                        "right_block_id": right_panel.block.block_id,
                        "left_cohort_digest": left_panel.block.cohort_digest,
                        "right_cohort_digest": right_panel.block.cohort_digest,
                        "same_arm_across_draws": left == right,
                        "preregistration_note": note,
                        **record,
                    }
                )
    return rows


def cross_panel_refusals(panels: list[Panel]) -> list[dict[str, Any]]:
    """Arms of one cohort whose group universes differ, so no index is common.

    Not handed to ``unpaired_contrast``: that interval is right for independent
    cohorts and too wide for arms that in fact share records, and these arms share
    records. The refusal names the arms and says why, which is the only honest
    thing available without re-deriving a common universe the estimand does not
    have.
    """

    rows: list[dict[str, Any]] = []
    for left_panel, right_panel in combinations(panels, 2):
        if left_panel.block.block_id != right_panel.block.block_id:
            continue
        for left in left_panel.names:
            for right in right_panel.names:
                rows.append(
                    {
                        "key": f"{left}_minus_{right}",
                        "block_id": left_panel.block.block_id,
                        "left": left,
                        "right": right,
                        "refused": True,
                        "refusal_reason": (
                            "these arms share a cohort draw but not a group "
                            "universe, so no common resample index addresses the "
                            "same units; an unpaired interval is not offered "
                            "instead, because it is correct for independent "
                            "cohorts and too wide for arms that share records"
                        ),
                        "statistics": None,
                    }
                )
    return rows


def declared_contrast_sign_tracking(
    within: list[dict[str, Any]], across: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Where a named pair's gap lands on every draw, and whether its sign holds."""

    tracked: list[dict[str, Any]] = []
    for entry in DECLARED_CONTRASTS:
        readings: list[dict[str, Any]] = []
        for record in [*within, *across]:
            if {record.get("left"), record.get("right")} != {entry["left"], entry["right"]}:
                continue
            if record.get("refused") or not record.get("statistics"):
                readings.append(
                    {
                        "key": record["key"],
                        "paired": record.get("paired"),
                        "gap_nats": None,
                        "refused": True,
                    }
                )
                continue
            block = record["statistics"]["information_nats_per_token"]
            oriented = 1.0 if record["left"] == entry["left"] else -1.0
            readings.append(
                {
                    "key": record["key"],
                    "paired": record.get("paired"),
                    "gap_nats": oriented * float(block["point"]),
                    "interval": sorted(oriented * value for value in block["interval"]),
                    "straddles_zero": block["interval"][0] < 0.0 < block["interval"][1],
                    "refused": False,
                }
            )
        signs = {
            (gap > 0.0)
            for gap in (r["gap_nats"] for r in readings)
            if gap is not None and gap != 0.0
        }
        tracked.append(
            {
                "left": entry["left"],
                "right": entry["right"],
                "orientation": f"gap = I({entry['left']}) - I({entry['right']})",
                "preregistration_note": entry["preregistration_note"],
                "readings": readings,
                "n_readings": len(readings),
                "sign_is_stable_across_readings": (None if not signs else len(signs) == 1),
                "ordering_may_be_reported": False,
            }
        )
    return tracked


# --------------------------------------------------------------------------- #
# Between-block variance
# --------------------------------------------------------------------------- #


def block_disjointness(blocks: list[Block]) -> dict[str, Any]:
    """Whether the supplied cohort draws actually hold different records."""

    missing = [b.block_id for b in blocks if b.cohort_records is None]
    if missing:
        return {
            "checked": False,
            "reason": (
                "the frozen-cohort records are unavailable for "
                f"{sorted(missing)}, so block disjointness cannot be verified and "
                "the spread below may mix a between-block component with a "
                "re-reading of overlapping records"
            ),
            "pairs": [],
            "all_disjoint": None,
        }
    pairs = []
    for left, right in combinations(blocks, 2):
        shared = len(set(left.cohort_records) & set(right.cohort_records))
        pairs.append(
            {
                "left": left.block_id,
                "right": right.block_id,
                "shared_records": int(shared),
                "disjoint": shared == 0,
            }
        )
    return {
        "checked": True,
        "reason": None,
        "pairs": pairs,
        "all_disjoint": all(pair["disjoint"] for pair in pairs),
    }


def between_block_variance(
    rows: list[dict[str, Any]], blocks: dict[str, Block]
) -> list[dict[str, Any]]:
    """The block-selection component, reported beside the interval and not inside it.

    The pre-registration is explicit that a variance component estimated from K
    blocks does not belong inside a percentile interval computed from one, so
    nothing here is combined with anything. Below eight blocks the spread is
    reported as a spread with its degrees of freedom and the interval is refused:
    at K = 2 the component has one degree of freedom, and the two K = 2 pairs
    already on disk disagree with each other about its size by a factor of fifty.
    """

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["context_information_nats"] is None:
            continue
        by_arm.setdefault(row["arm"], []).append(row)

    records: list[dict[str, Any]] = []
    for arm in sorted(by_arm):
        entries = sorted(by_arm[arm], key=lambda row: row["block_id"])
        points = [float(row["context_information_nats"]) for row in entries]
        k = len(points)
        common: dict[str, Any] = {
            "arm": arm,
            "n_blocks": k,
            "blocks": [
                {
                    "block_id": row["block_id"],
                    "cohort_digest": row["cohort_digest"],
                    "cohort_name": row["cohort_name"],
                    "context_information_nats": row["context_information_nats"],
                    "within_block_bootstrap_ci_95": row["bootstrap_ci_95"],
                }
                for row in entries
            ],
            "folded_into_the_bootstrap_interval": False,
            "note": (
                "the bootstrap covers within-cohort and within-reference sampling "
                "error only; this component is block-selection error and is "
                "reported beside it, never inside it"
            ),
        }
        if k < 2:
            records.append(
                {
                    **common,
                    "reported": False,
                    "reason": (
                        "one cohort block was supplied for this arm, so there is no "
                        "between-block component to estimate"
                    ),
                    "between_block_sd_nats": None,
                    "degrees_of_freedom": 0,
                    "between_block_interval": None,
                }
            )
            continue
        spread = float(np.std(points, ddof=1))
        disjoint = block_disjointness([blocks[row["block_id"]] for row in entries])
        interval: list[float] | None = None
        refusal: str | None = (
            f"K = {k} blocks gives {k - 1} degree(s) of freedom, below the "
            f"{MIN_BLOCKS_FOR_BETWEEN_BLOCK_INTERVAL}-block floor this stage "
            "requires before a between-block interval is published. The spread and "
            "its degrees of freedom are reported instead. A K = 2 pair is a "
            "difference, not an interval, and the experiment log forbids "
            "presenting it as one."
        )
        if k >= MIN_BLOCKS_FOR_BETWEEN_BLOCK_INTERVAL:
            from scipy import stats as _stats  # local: only this branch needs it

            half = float(_stats.t.ppf(0.975, k - 1)) * spread / math.sqrt(k)
            mean = float(np.mean(points))
            interval, refusal = [mean - half, mean + half], None
        records.append(
            {
                **common,
                "reported": True,
                "reason": None,
                "mean_nats": float(np.mean(points)),
                "between_block_sd_nats": spread,
                "between_block_range_nats": float(max(points) - min(points)),
                "degrees_of_freedom": k - 1,
                "blocks_disjoint": disjoint,
                "between_block_interval": interval,
                "between_block_interval_refusal": refusal,
            }
        )
    return records


# --------------------------------------------------------------------------- #
# E4: the smoothing sweep
# --------------------------------------------------------------------------- #


def unsmoothed_information(arm: ArmStatistics) -> tuple[float | None, str | None]:
    """``I`` against the unsmoothed held-out unigram, when one exists.

    It exists only when every cohort target token was seen in the reference. One
    unseen target makes the maximum-likelihood unigram assign it zero and the
    cross-entropy infinite, which is the reason the smoothing constant cannot be
    eliminated and can only be swept. Where it exists it is the fixed point the
    sweep is measured against: ``H_baseline(a) - H_baseline(0)`` is exactly
    ``sum_v c(v)/N [log(1 + aV/R) - log(1 + a/r(v))]``, which is at most
    ``log(1 + aV/R)`` because the second term is non-negative. That inequality is
    the analytic bound this stage reports and checks, rather than a claim about
    monotonicity, which does not hold in general: a token whose reference count is
    below ``R/V`` has its surprisal *reduced* by smoothing.
    """

    vocabulary = arm.vocab_size
    cohort = np.bincount(
        arm.cohort.targets.unique_token_ids,
        weights=arm.cohort.targets.counts.astype(np.float64),
        minlength=vocabulary,
    )
    reference = np.bincount(
        arm.reference.targets.unique_token_ids,
        weights=arm.reference.targets.counts.astype(np.float64),
        minlength=vocabulary,
    )
    support = cohort > 0
    if bool(np.any(reference[support] <= 0.0)):
        return None, (
            "at least one cohort target token is unseen in the reference, so the "
            "unsmoothed held-out unigram is infinite and the sweep has no "
            "zero-smoothing anchor on this arm"
        )
    tokens = float(cohort.sum())
    total = float(reference.sum())
    baseline = float(
        -(cohort[support] * (np.log(reference[support]) - math.log(total))).sum() / tokens
    )
    model = float(arm.cohort.clean_nll_sum.sum() / arm.cohort.token_count.sum())
    return baseline - model, None


def alpha_sensitivity(
    panels: list[Panel],
    *,
    sweep: list[float],
    n_bootstrap: int,
    confidence: float,
) -> list[dict[str, Any]]:
    """``I`` re-bootstrapped across the smoothing ladder, per arm and per draw.

    The whole sweep is a CPU job because smoothing is applied at analysis time
    from the persisted reference counts, which is what makes the withdrawn
    published figures replaceable by measurement here. Every alpha runs at the
    panel's own seed, so the draws are identical across the ladder and a
    difference between two alphas is a paired difference rather than two
    independent bootstraps.
    """

    records: list[dict[str, Any]] = []
    for panel in panels:
        by_alpha = {
            float(alpha): bootstrap_arms(
                [replace(arm, smoothing=float(alpha)) for arm in panel.statistics],
                seed=panel.seed,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                contrasts=(),
            )
            for alpha in sweep
        }
        for arm in panel.statistics:
            reference_tokens = float(arm.reference.token_count.sum())
            unsmoothed, unsmoothed_reason = unsmoothed_information(arm)
            entries: list[dict[str, Any]] = []
            for alpha in sweep:
                statistics = by_alpha[float(alpha)].arms[arm.name].record["statistics"]
                bound = math.log1p(float(alpha) * arm.vocab_size / reference_tokens)
                information = (
                    None
                    if statistics is None
                    else float(statistics["information_nats_per_token"]["point"])
                )
                excess = (
                    None
                    if information is None or unsmoothed is None
                    else information - unsmoothed
                )
                entries.append(
                    {
                        "alpha": float(alpha),
                        "information_nats": information,
                        "bootstrap_ci_95": (
                            None
                            if statistics is None
                            else list(statistics["information_nats_per_token"]["interval"])
                        ),
                        "baseline_entropy_nats_per_token": (
                            None
                            if statistics is None
                            else statistics["baseline_entropy_nats_per_token"]["point"]
                        ),
                        "analytic_bound_nats": bound,
                        "excess_over_unsmoothed_nats": excess,
                        "within_analytic_bound": (
                            None if excess is None else bool(excess <= bound + 1e-9)
                        ),
                    }
                )
            measured = [e["information_nats"] for e in entries if e["information_nats"] is not None]
            ordered = sorted(
                (e for e in entries if e["information_nats"] is not None),
                key=lambda e: e["alpha"],
            )
            records.append(
                {
                    "block_id": panel.block.block_id,
                    "panel_id": panel.panel_id,
                    "arm": arm.name,
                    "cohort_digest": panel.block.cohort_digest,
                    "seed": panel.seed,
                    "vocab_size": int(arm.vocab_size),
                    "reference_tokens": int(reference_tokens),
                    "headline_smoothing": float(arm.smoothing),
                    "unsmoothed_information_nats": unsmoothed,
                    "unsmoothed_unavailable_reason": unsmoothed_reason,
                    "by_alpha": entries,
                    "information_range_nats": (
                        None if not measured else float(max(measured) - min(measured))
                    ),
                    "nondecreasing_in_alpha": (
                        None
                        if len(ordered) < 2
                        else all(
                            later["information_nats"] >= earlier["information_nats"] - 1e-12
                            for earlier, later in zip(ordered, ordered[1:])
                        )
                    ),
                    "note": (
                        "the analytic bound log(1 + aV/R) bounds how far smoothing "
                        "can push I above its unsmoothed value; it is not a "
                        "monotonicity claim, because a token whose reference count "
                        "is below R/V has its surprisal reduced by smoothing. No "
                        "bias correction is applied: the constant cannot be "
                        "eliminated, only swept and reported"
                    ),
                }
            )
    return records


# --------------------------------------------------------------------------- #
# E5: the leakage-removed sensitivity
# --------------------------------------------------------------------------- #


def kmer_set(record: str, k: int) -> set[str]:
    return {record[i : i + k] for i in range(len(record) - k + 1)} if len(record) >= k else set()


def leaked_reference_mask(
    cohort_records: list[str], reference_records: list[str], k: int
) -> np.ndarray:
    """Reference records sharing at least one k-mer with any cohort record.

    ``pathways.assert_disjoint`` checks exact content equality, which does not
    reach a reference record that is a near-copy of a scored one. Such a record
    makes the fitted unigram fit the cohort better than a held-out one should,
    which lowers ``H_baseline`` and deflates ``I`` -- unequally across arms, since
    the protein cohorts carry far more near-duplicate structure than the text
    ones.
    """

    pool: set[str] = set()
    for record in cohort_records:
        pool |= kmer_set(record, k)
    return np.asarray(
        [bool(kmer_set(record, k) & pool) for record in reference_records], dtype=bool
    )


def leakage_sensitivity(
    panels: list[Panel],
    *,
    k: int,
    n_bootstrap: int,
    confidence: float,
) -> list[dict[str, Any]]:
    """``I`` recomputed with the leaked part of the reference dropped, or refused."""

    records: list[dict[str, Any]] = []
    for panel in panels:
        block = panel.block
        inputs = {entry.name: entry for entry in block.arms}
        if block.reference_records is None or block.cohort_records is None:
            reason = (
                "the 30-mer overlap between reference and cohort cannot be "
                "determined from what is on disk: the sidecar stores order-free "
                "token counts, not sequence text, and "
                + (
                    "no --reference-json was supplied; cohort_power.py "
                    "persists the held-out reference beside the cohort as "
                    "reference_<name>_<digest>.json, so pass that file"
                    if block.reference_records is None
                    else f"the frozen-cohort file for {block.block_id} is missing"
                )
                + ". The sensitivity is reported as unavailable rather than "
                "approximated from token counts, which cannot see a shared k-mer"
            )
            for arm in panel.statistics:
                records.append(
                    {
                        "block_id": block.block_id,
                        "panel_id": panel.panel_id,
                        "arm": arm.name,
                        "is_unigram_null_control": arm.name.endswith(CONTROL_SUFFIX),
                        "kmer": int(k),
                        "available": False,
                        "reason": reason,
                        "information_nats": None,
                        "bootstrap_ci_95": None,
                        "delta_vs_headline_nats": None,
                    }
                )
            continue

        leaked = leaked_reference_mask(block.cohort_records, block.reference_records, k)
        for arm in panel.statistics:
            # A null control carries its parent arm's baseline reference, so the
            # parent's row of the screen is the one that applies to it. Only that
            # reference is screened: the control's own distribution comes from
            # another block and is not part of this sensitivity, which is stated
            # on the record rather than left to be inferred.
            parent = arm.name.removesuffix(CONTROL_SUFFIX)
            entry = inputs[parent]
            keep = ~leaked[entry.reference_keep]
            headline = panel.result.arms[arm.name].record["statistics"]
            headline_point = (
                None
                if headline is None
                else float(headline["information_nats_per_token"]["point"])
            )
            common = {
                "block_id": block.block_id,
                "panel_id": panel.panel_id,
                "arm": arm.name,
                "is_unigram_null_control": parent != arm.name,
                "screened_reference": (
                    "the baseline reference only; a null control's own unigram is "
                    "fitted on another block and is not screened here"
                    if parent != arm.name
                    else "the baseline reference, which is the only one this arm has"
                ),
                "kmer": int(k),
                "available": True,
                "n_reference_records_before": int(keep.size),
                "n_reference_records_dropped": int((~keep).sum()),
                "headline_information_nats": headline_point,
                "note": (
                    "a sensitivity beside the headline, not a replacement for it; "
                    "the headline stays on the declared reference"
                ),
            }
            if not bool(keep.any()):
                records.append(
                    {
                        **common,
                        "reason": (
                            "every reference record shares a "
                            f"{k}-mer with the cohort, so no held-out reference "
                            "survives the screen"
                        ),
                        "information_nats": None,
                        "bootstrap_ci_95": None,
                        "delta_vs_headline_nats": None,
                    }
                )
                continue
            reduced = replace(
                arm,
                reference=ReferenceStatistics(
                    token_count=arm.reference.token_count[keep],
                    targets=select_records(arm.reference.targets, keep),
                    group_id=arm.reference.group_id[keep],
                ),
            )
            result = bootstrap_arms(
                [reduced],
                seed=panel.seed + LEAKAGE_SEED_OFFSET,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                contrasts=(),
            ).arms[arm.name]
            statistics = result.record["statistics"]
            point = (
                None
                if statistics is None
                else float(statistics["information_nats_per_token"]["point"])
            )
            records.append(
                {
                    **common,
                    "seed": panel.seed + LEAKAGE_SEED_OFFSET,
                    "reason": result.record["refusal_reason"],
                    "refused": bool(result.record["refused"]),
                    "n_reference_tokens_after": int(reduced.reference.token_count.sum()),
                    "information_nats": point,
                    "bootstrap_ci_95": (
                        None
                        if statistics is None
                        else list(statistics["information_nats_per_token"]["interval"])
                    ),
                    "delta_vs_headline_nats": (
                        None
                        if point is None or headline_point is None
                        else point - headline_point
                    ),
                    "unit_floor": result.record["unit_floor"],
                }
            )
    return records


# --------------------------------------------------------------------------- #
# What the criteria do at a known zero
# --------------------------------------------------------------------------- #


def null_control_summary(
    controls: list[NullControl],
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    confidence: float,
) -> dict[str, Any]:
    """Every control's reading, and what each criterion did where I is zero.

    This is the only place in the artefact where a criterion is watched at a
    point whose true value is known. The floor behaves correctly here by
    refusing and the sign rule behaves correctly by failing; a sign PASS is the
    pre-registered failure mode, and because a nominal interval leaves a tail
    outside itself, the rate matters as much as the event. Both are counted, and
    the departure of the measured value from zero is reported beside the
    analytic smoothing bound it is to be read against.
    """

    reading_of = {
        (row["block_id"], row["arm"]): row
        for row in rows
        if row.get("is_unigram_null_control")
    }
    readings: list[dict[str, Any]] = []
    for control in controls:
        row = reading_of.get((control.block_id, control.name))
        if row is None or row["context_information_nats"] is None:
            continue
        bounds = control.record["analytic_smoothing_bound_nats"]
        larger = max(bounds["baseline"], bounds["control"])
        departure = float(row["context_information_nats"])
        readings.append(
            {
                "control_arm": control.name,
                "arm": control.arm,
                "block_id": control.block_id,
                "panel_id": row["panel_id"],
                "cohort_name": control.record["cohort_name"],
                "cohort_kind": control.record["cohort_kind"],
                "baseline_block_id": control.record["baseline_block_id"],
                "control_block_id": control.record["control_block_id"],
                "vocab_size": control.record["vocab_size"],
                "cohort_copies_dropped_from_control_reference": control.record[
                    "cohort_copies_dropped_from_control_reference"
                ],
                "baseline_control_reference_exact_overlap": control.record[
                    "baseline_control_reference_exact_overlap"
                ],
                "true_information_nats": 0.0,
                "measured_information_nats": departure,
                "bootstrap_ci_95": row["bootstrap_ci_95"],
                "bootstrap_se": row["bootstrap_se"],
                # The same interval with the L34/L42 displacement removed, and
                # the two objects a reader must be able to tell it from: the
                # displacement itself, and the narrower interval that dropping
                # the reference resampling would have given instead.
                "displacement_corrected_ci_95": row["displacement_corrected_ci_95"],
                "reference_resampling_displacement": row[
                    "reference_resampling_displacement"
                ],
                "cohort_only_bootstrap_se": row["cohort_only_bootstrap_se"],
                "reference_held_fixed_ci_95": row["reference_held_fixed_ci_95"],
                "per_arm_identification_status": row["per_arm_identification_status"],
                # The displacement of the bootstrap distribution away from the
                # estimate on the data. It is reported and never applied, and at
                # a known zero it is what the sign rule ends up reading: a
                # resampled reference gives E[H_baseline] above H_baseline by
                # Jensen, so the interval can sit entirely above the point.
                "bootstrap_bias_nats": row["bootstrap_bias"],
                "median_bias_z0": row["median_bias_z0"],
                "interval_lies_entirely_above_the_point": bool(
                    row["bootstrap_ci_95"][0] > row["context_information_nats"]
                ),
                "interval_covers_zero": bool(
                    row["bootstrap_ci_95"][0] <= 0.0 <= row["bootstrap_ci_95"][1]
                ),
                "analytic_smoothing_bound_nats": bounds,
                "abs_departure_over_larger_bound": (
                    None if larger <= 0.0 else abs(departure) / larger
                ),
                "screening_status": row["screening_status"],
                "screening_interval_status": row["screening_interval_status"],
                "sign_status": row["sign_status"],
                "status_disagreement": row["status_disagreement"],
            }
        )

    by_cohort: dict[str, dict[str, Any]] = {}
    for reading in readings:
        entry = by_cohort.setdefault(
            reading["cohort_name"],
            {
                "cohort_name": reading["cohort_name"],
                "cohort_kind": reading["cohort_kind"],
                "n_readings": 0,
                "arms": [],
                "departures_nats": [],
                "biases_nats": [],
                "n_floor_pass": 0,
                "n_sign_pass": 0,
            },
        )
        entry["n_readings"] += 1
        if reading["arm"] not in entry["arms"]:
            entry["arms"].append(reading["arm"])
        # The true value is zero, so the measured value is the departure.
        entry["departures_nats"].append(reading["measured_information_nats"])
        entry["biases_nats"].append(reading["bootstrap_bias_nats"])
        entry["n_floor_pass"] += int(reading["screening_status"] == "PASS")
        entry["n_sign_pass"] += int(reading["sign_status"] == "PASS")
    for entry in by_cohort.values():
        departures = entry.pop("departures_nats")
        biases = entry.pop("biases_nats")
        entry["arms"] = sorted(entry["arms"])
        entry["mean_bootstrap_bias_nats"] = float(np.mean(biases))
        entry["max_bootstrap_bias_nats"] = float(np.max(biases))
        entry["mean_departure_nats"] = float(np.mean(departures))
        entry["max_abs_departure_nats"] = float(np.max(np.abs(departures)))
        entry["min_departure_nats"] = float(np.min(departures))
        entry["max_departure_nats"] = float(np.max(departures))
        entry["n_departures_above_zero"] = int(sum(d > 0.0 for d in departures))

    # Counted over readings, not over arm names: one arm carries one reading per
    # block, and collapsing them would report a rate eight times too small.
    floor_pass = [r for r in readings if r["screening_status"] == "PASS"]
    sign_pass = [r for r in readings if r["sign_status"] == "PASS"]
    floor_pass_arms = sorted({r["control_arm"] for r in floor_pass})
    sign_pass_arms = sorted({r["control_arm"] for r in sign_pass})
    tail = (1.0 - confidence) / 2.0
    return {
        "requested": True,
        "construction": CONTROL_CONSTRUCTION_NOTE,
        "how_to_read": CONTROL_READING_NOTE,
        "true_information_nats": 0.0,
        "n_controls_requested": len(controls),
        "n_controls_measured": len(readings),
        "provenance": [control.record for control in controls],
        "controls_without_an_independent_source": [
            {
                "control_arm": control.record["control_arm"],
                "block_id": control.record["block_id"],
                "refusal_reason": control.record["refusal_reason"],
            }
            for control in controls
            if not control.record["available"]
        ],
        "readings": readings,
        "independence_check": {
            "what": (
                "the zero rests on two facts: the control's unigram was not "
                "fitted on the records it scores, and the two references are "
                "different samples. The first is enforced by dropping every "
                "borrowed reference record that is a verbatim copy of a scored "
                "record, which is the disjointness the producing stage already "
                "asserts within a block; the second is established by reference "
                "digest. Where the record text was not supplied the first "
                "cannot be checked, and every affected record says so"
            ),
            "n_readings_with_exact_overlap_checked": sum(
                1
                for r in readings
                if r["cohort_copies_dropped_from_control_reference"] is not None
            ),
            "n_readings_that_dropped_cohort_copies": sum(
                1
                for r in readings
                if (r["cohort_copies_dropped_from_control_reference"] or 0) > 0
            ),
            "max_cohort_copies_dropped": max(
                (
                    r["cohort_copies_dropped_from_control_reference"]
                    for r in readings
                    if r["cohort_copies_dropped_from_control_reference"] is not None
                ),
                default=None,
            ),
            "max_baseline_control_reference_overlap": max(
                (
                    r["baseline_control_reference_exact_overlap"]
                    for r in readings
                    if r["baseline_control_reference_exact_overlap"] is not None
                ),
                default=None,
            ),
        },
        "by_cohort": [by_cohort[name] for name in sorted(by_cohort)],
        "criteria_at_a_known_zero": {
            "n_readings": len(readings),
            "floor_nats": float(threshold),
            "floor_correct_behaviour_at_zero": "FAIL (the arm is refused)",
            "n_floor_pass_readings": len(floor_pass),
            "floor_pass_arms": floor_pass_arms,
            "observed_floor_pass_rate": (
                None if not readings else len(floor_pass) / len(readings)
            ),
            "floor_behaves": not floor_pass,
            "floor_statement": (
                f"the {threshold:g} nats/token screening floor refuses every "
                "control, which is the correct behaviour at a true zero"
                if not floor_pass
                else f"the floor PASSES on {len(floor_pass)} of {len(readings)} "
                "readings whose true context information is zero by construction, "
                "on " + ", ".join(floor_pass_arms)
            ),
            "sign_correct_behaviour_at_zero": (
                "FAIL (the interval's lower bound is not above zero)"
            ),
            "n_sign_pass_readings": len(sign_pass),
            "sign_pass_arms": sign_pass_arms,
            "sign_behaves": not sign_pass,
            "expected_false_pass_rate_at_a_true_zero": float(tail),
            "observed_sign_pass_rate": (
                None if not readings else len(sign_pass) / len(readings)
            ),
            "sign_pass_mean_bootstrap_bias_nats": (
                None
                if not sign_pass
                else float(np.mean([r["bootstrap_bias_nats"] for r in sign_pass]))
            ),
            "n_sign_pass_with_the_interval_above_the_point": sum(
                r["interval_lies_entirely_above_the_point"] for r in sign_pass
            ),
            "sign_statement": (
                "the sign rule fails on every control, which is the correct "
                "behaviour at a true zero; at this confidence level a rate of "
                f"about {tail:.3f} would still be expected from the interval's own "
                "tail, so agreement on this many controls is weak evidence for the "
                "rule and none at all for adopting it"
                if not sign_pass
                else f"the sign rule PASSES on {len(sign_pass)} of {len(readings)} "
                "readings whose true context information is zero by construction, "
                "on " + ", ".join(sign_pass_arms) + f", against the {tail:.3f} that "
                "the interval's own tail would give. This is the pre-registered "
                "failure mode. Where the interval also lies entirely above the "
                "point estimate the rule is reading the bootstrap's displacement "
                "rather than the measurement, and bootstrap_bias_nats is the size "
                "of that displacement"
            ),
            "sign_rule_is_adopted": False,
            "note": SCREENING_NOTE,
        },
    }


# --------------------------------------------------------------------------- #
# Held-out validation of the screening floor and the identification criterion
# --------------------------------------------------------------------------- #


#: Candidate thresholds the recalibration searches. EXP-R2-218's own grid, so a
#: fold's threshold and the panel-wide one are read off the same ladder.
HELD_OUT_TAU_GRID: tuple[float, ...] = tuple(
    round(0.005 * step, 6) for step in range(0, 301)
)

#: EXP-R2-218's declared design target for Objective A's false-positive rate.
#: It is a convention on an error rate, not a measured quantity, and the
#: recalibration inherits it rather than re-choosing it.
HELD_OUT_TARGET_FALSE_POSITIVE_RATE = 0.05

HELD_OUT_NOTE = (
    "EXP-R2-218 read its threshold off the same panel it evaluated it on, which "
    "makes it a screening result. This section cross-fits that selection: on each "
    "fold the threshold is re-derived from the calibration units ALONE, by the "
    "same rule and on the same grid, and is then scored on units the derivation "
    "never saw. The shipped floor and the shipped Fieller denominator condition "
    "are scored on the same held-out units beside it, so the comparison is "
    "between two rules on one set of readings rather than between two sets."
)


def _smallest_admissible_tau(values: Sequence[float], target: float) -> float | None:
    """Smallest grid threshold whose pass rate over ``values`` is at most ``target``.

    This is Objective A's selection rule and the only place it is written down.
    The calibration fold and the full panel must be scored by one rule or the
    held-out comparison measures the difference between two rules instead of the
    difference between two sets of units.
    """

    if not values:
        return None
    array = np.asarray(list(values), dtype=float)
    for tau in HELD_OUT_TAU_GRID:
        if float(np.mean(array >= tau)) <= target:
            return float(tau)
    return None


def _exact_upper_bound(successes: int, trials: int, confidence: float) -> float | None:
    """One-sided Clopper-Pearson upper limit on a binomial rate.

    A held-out rate of zero is not a rate of zero: it is a rate consistent with
    everything below this bound, and on a panel resolved to a couple of dozen
    independent cohorts that bound is much larger than the target it is being
    read against. Reporting the point estimate alone would be the same mistake
    this whole exercise exists to correct, one level up.
    """

    if trials <= 0:
        return None
    if successes >= trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def _reading_unit(reading: dict[str, Any]) -> tuple[str, Any]:
    """The identity of the *measurement* a control reading carries.

    The block's records and the arm's vocabulary, and nothing else. Two arms of
    one block sharing a tokenisation produce one number by construction, so this
    key is what a false-positive rate is actually resolved to.
    """

    return (reading["block_id"], reading["vocab_size"])


def null_reading_equivalence(readings: list[dict[str, Any]]) -> dict[str, Any]:
    """Which control readings are one measurement rather than several.

    ``CONTROL_CONSTRUCTION_NOTE`` states the duplication as a property of the
    construction; this measures it, because the unit count decides how finely a
    held-out error rate can be resolved and an assumed unit count is exactly the
    kind of thing that silently inflates one. A class whose members disagree is
    refused rather than reported: the disagreement would mean a control reading
    is not the property the whole accounting rests on, and a partition built on
    a false premise would misassign folds as well as miscount units.
    """

    by_unit: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for reading in readings:
        by_unit.setdefault(_reading_unit(reading), []).append(reading)

    classes: list[set[str]] = []
    for unit, members in sorted(by_unit.items(), key=lambda item: str(item[0])):
        distinct = {
            (
                float(member["measured_information_nats"]),
                tuple(float(bound) for bound in member["bootstrap_ci_95"]),
                float(member["bootstrap_se"]),
            )
            for member in members
        }
        if len(distinct) > 1:
            named = ", ".join(sorted(member["arm"] for member in members))
            raise SystemExit(
                f"block {unit[0]} vocabulary {unit[1]}: the controls of {named} "
                "share a block and a tokenisation but do not carry the same "
                f"reading ({len(distinct)} distinct values). The null control is "
                "declared to be a property of the records and the vocabulary "
                "alone, and the held-out unit count rests on that; a disagreement "
                "here means the declaration is wrong and the accounting below "
                "would be wrong with it"
            )
        names = {member["arm"] for member in members}
        overlapping = [existing for existing in classes if existing & names]
        for existing in overlapping:
            classes.remove(existing)
            names = names | existing
        classes.append(names)

    labelled = sorted(("+".join(sorted(names)), sorted(names)) for names in classes)
    label_of = {arm: label for label, names in labelled for arm in names}
    return {
        "what": (
            "one control reading is a property of the block's records and the "
            "arm's vocabulary; arms sharing both carry the identical number"
        ),
        "n_readings": len(readings),
        "n_distinct_measurements": len(by_unit),
        "n_cohort_draws": len({reading["block_id"] for reading in readings}),
        "tokenisation_classes": [
            {"label": label, "arms": names, "n_arms": len(names)}
            for label, names in labelled
        ],
        "arm_to_tokenisation_class": label_of,
    }


def _held_out_fold(
    calibration: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    held_out_real: list[dict[str, Any]],
    *,
    threshold: float,
    target: float,
) -> dict[str, Any]:
    """One fold: derive on the calibration units, score on the held-out ones."""

    tau_point = _smallest_admissible_tau(
        [reading["measured_information_nats"] for reading in calibration], target
    )
    tau_interval = _smallest_admissible_tau(
        [reading["bootstrap_ci_95"][0] for reading in calibration], target
    )
    tau_corrected = _smallest_admissible_tau(
        [reading["displacement_corrected_ci_95"][0] for reading in calibration], target
    )
    if tau_point is None or tau_interval is None or tau_corrected is None:
        return {
            "refused": True,
            "refusal_reason": (
                "no threshold on the grid brings the calibration fold's "
                f"false-positive rate to {target:g} or below, so this fold "
                "derives nothing to validate"
            ),
        }

    rules: dict[str, list[bool]] = {
        "recalibrated_point_rule": [
            reading["measured_information_nats"] >= tau_point for reading in held_out
        ],
        "recalibrated_interval_rule": [
            reading["bootstrap_ci_95"][0] >= tau_interval for reading in held_out
        ],
        "shipped_point_rule": [
            reading["measured_information_nats"] >= threshold for reading in held_out
        ],
        "shipped_interval_rule": [
            reading["bootstrap_ci_95"][0] >= threshold for reading in held_out
        ],
        "shipped_fieller_denominator_admissibility": [
            ratio_denominator_admissibility(
                reading["measured_information_nats"], reading["bootstrap_se"]
            )["admissible"]
            for reading in held_out
        ],
        # The candidate of the displacement-corrected pre-registration: a
        # per-arm rule with no constant in it at all.
        "candidate_displacement_corrected_rule_at_zero": [
            reading["displacement_corrected_ci_95"][0] > 0.0 for reading in held_out
        ],
        "recalibrated_displacement_corrected_rule": [
            reading["displacement_corrected_ci_95"][0] >= tau_corrected
            for reading in held_out
        ],
        # The declared sensitivity: dropping the reference resampling removes
        # the displacement AND the reference's share of the width, so a lower
        # rate here is not evidence that the displacement was the problem.
        "sensitivity_reference_held_fixed_rule_at_zero": [
            reading["reference_held_fixed_ci_95"][0] > 0.0 for reading in held_out
        ],
    }
    units = [_reading_unit(reading) for reading in held_out]
    n_distinct = len(set(units))
    cohort_draws = [reading["block_id"] for reading in held_out]

    disagreeing = [
        {
            "arm": row["arm"],
            "block_id": row["block_id"],
            "context_information_nats": row["context_information_nats"],
            "verdict_under_the_recalibrated_threshold": (
                "PASS" if row["context_information_nats"] >= tau_point else "FAIL"
            ),
            "verdict_under_the_shipped_floor": (
                "PASS" if row["context_information_nats"] >= threshold else "FAIL"
            ),
        }
        for row in held_out_real
        if (row["context_information_nats"] >= tau_point)
        != (row["context_information_nats"] >= threshold)
    ]
    return {
        "refused": False,
        "n_calibration_readings": len(calibration),
        "n_calibration_distinct_measurements": len(
            {_reading_unit(reading) for reading in calibration}
        ),
        "n_held_out_readings": len(held_out),
        "n_held_out_distinct_measurements": n_distinct,
        "recalibrated_tau_nats": {
            "point_rule": tau_point,
            "interval_rule": tau_interval,
            "displacement_corrected_interval_rule": tau_corrected,
        },
        "n_held_out_cohort_draws": len(set(cohort_draws)),
        "n_held_out_false_positives": {
            rule: int(sum(fired)) for rule, fired in rules.items()
        },
        "n_held_out_distinct_false_positives": {
            rule: len({unit for unit, hit in zip(units, fired) if hit})
            for rule, fired in rules.items()
        },
        "n_held_out_cohort_draw_false_positives": {
            rule: len({draw for draw, hit in zip(cohort_draws, fired) if hit})
            for rule, fired in rules.items()
        },
        "held_out_false_positive_rate": {
            rule: (None if not held_out else float(sum(fired) / len(held_out)))
            for rule, fired in rules.items()
        },
        "max_held_out_null_departure_nats": (
            None
            if not held_out
            else max(reading["measured_information_nats"] for reading in held_out)
        ),
        "real_arm_verdicts": {
            "n_readings": len(held_out_real),
            "n_changing_verdict": len(disagreeing),
            "readings_that_change_verdict": disagreeing,
        },
    }


def _held_out_scheme(
    name: str,
    question: str,
    null_readings: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    unit_of_reading: Any,
    unit_of_row: Any,
    buffer_of: Any,
    *,
    threshold: float,
    confidence: float,
    target: float,
) -> dict[str, Any]:
    """Every fold of one cross-fitting scheme, and the pooled rate over them."""

    units = sorted({unit_of_reading(reading) for reading in null_readings})
    if len(units) < 2:
        return {
            "scheme": name,
            "question": question,
            "refused": True,
            "refusal_reason": (
                f"cross-fitting needs at least two units and this scheme has "
                f"{len(units)}; a single unit cannot be held out of its own "
                "derivation"
            ),
        }

    unassigned = [row for row in real_rows if unit_of_row(row) not in set(units)]
    folds: list[dict[str, Any]] = []
    for unit in units:
        excluded = buffer_of(unit)
        calibration = [
            reading
            for reading in null_readings
            if unit_of_reading(reading) not in excluded
        ]
        held_out = [
            reading for reading in null_readings if unit_of_reading(reading) == unit
        ]
        held_out_real = [row for row in real_rows if unit_of_row(row) == unit]
        fold = _held_out_fold(
            calibration,
            held_out,
            held_out_real,
            threshold=threshold,
            target=target,
        )
        folds.append(
            {
                "held_out_unit": unit,
                "calibration_units_withheld_as_a_buffer": sorted(
                    str(item) for item in excluded if item != unit
                ),
                **fold,
            }
        )

    scored = [fold for fold in folds if not fold["refused"]]
    if not scored:
        return {
            "scheme": name,
            "question": question,
            "refused": True,
            "refusal_reason": "every fold refused; nothing was validated",
            "folds": folds,
        }

    n_readings = sum(fold["n_held_out_readings"] for fold in scored)
    n_distinct = sum(fold["n_held_out_distinct_measurements"] for fold in scored)
    n_cohort_draws = sum(fold["n_held_out_cohort_draws"] for fold in scored)
    rules = sorted(scored[0]["n_held_out_false_positives"])
    pooled: dict[str, Any] = {}
    for rule in rules:
        hits = sum(fold["n_held_out_false_positives"][rule] for fold in scored)
        distinct_hits = sum(
            fold["n_held_out_distinct_false_positives"][rule] for fold in scored
        )
        cohort_draw_hits = sum(
            fold["n_held_out_cohort_draw_false_positives"][rule] for fold in scored
        )
        pooled[rule] = {
            "n_false_positives": hits,
            "n_held_out_readings": n_readings,
            "rate_over_readings": None if not n_readings else hits / n_readings,
            "exact_upper_95_over_readings": _exact_upper_bound(
                hits, n_readings, confidence
            ),
            "n_distinct_false_positives": distinct_hits,
            "n_held_out_distinct_measurements": n_distinct,
            "rate_over_distinct_measurements": (
                None if not n_distinct else distinct_hits / n_distinct
            ),
            "exact_upper_95_over_distinct_measurements": _exact_upper_bound(
                distinct_hits, n_distinct, confidence
            ),
            "n_cohort_draw_false_positives": cohort_draw_hits,
            "n_held_out_cohort_draws": n_cohort_draws,
            "rate_over_cohort_draws": (
                None if not n_cohort_draws else cohort_draw_hits / n_cohort_draws
            ),
            "exact_upper_95_over_cohort_draws": _exact_upper_bound(
                cohort_draw_hits, n_cohort_draws, confidence
            ),
            "meets_the_target_over_readings": (
                None if not n_readings else bool(hits / n_readings <= target)
            ),
            "meets_the_target_over_distinct_measurements": (
                None if not n_distinct else bool(distinct_hits / n_distinct <= target)
            ),
        }

    taus = [fold["recalibrated_tau_nats"]["point_rule"] for fold in scored]
    interval_taus = [
        fold["recalibrated_tau_nats"]["interval_rule"] for fold in scored
    ]
    corrected_taus = [
        fold["recalibrated_tau_nats"]["displacement_corrected_interval_rule"]
        for fold in scored
    ]
    changed = sum(fold["real_arm_verdicts"]["n_changing_verdict"] for fold in scored)
    return {
        "scheme": name,
        "question": question,
        "refused": False,
        "n_folds": len(scored),
        "n_folds_refused": len(folds) - len(scored),
        "recalibrated_tau_nats": {
            "point_rule": {
                "min": min(taus),
                "max": max(taus),
                "distinct_values": sorted(set(taus)),
                "n_folds_below_the_shipped_floor": sum(
                    1 for tau in taus if tau < threshold
                ),
            },
            "interval_rule": {
                "min": min(interval_taus),
                "max": max(interval_taus),
                "distinct_values": sorted(set(interval_taus)),
            },
            "displacement_corrected_interval_rule": {
                "min": min(corrected_taus),
                "max": max(corrected_taus),
                "distinct_values": sorted(set(corrected_taus)),
            },
        },
        "margin_from_the_largest_recalibrated_tau_to_the_shipped_floor_nats": (
            threshold - max(taus)
        ),
        "pooled_held_out_false_positive_rate": pooled,
        "real_arm_verdict_agreement": {
            "n_held_out_readings": sum(
                fold["real_arm_verdicts"]["n_readings"] for fold in scored
            ),
            # A real arm on a unit this scheme has no fold for is never scored
            # against a fold threshold, so it is named rather than dropped: an
            # agreement rate counted over an unstated denominator is the failure
            # this section exists to measure.
            "n_readings_on_no_held_out_unit": len(unassigned),
            "readings_on_no_held_out_unit": sorted(
                {(row["arm"], row["block_id"]) for row in unassigned}
            ),
            "n_changing_verdict": changed,
            "agrees_everywhere": changed == 0,
        },
        "folds": folds,
    }


ADOPTION_CRITERION_NOTE = (
    "archive/docs/R5/context-measurement-protocols.md freezes three "
    "conditions, all of which must hold before the displacement-corrected "
    "per-arm rule may replace budget.SCREENING_CONTEXT_INFORMATION_NATS. A1: the "
    "held-out false-positive rate of the corrected rule at zero is at most the "
    "target over readings AND over distinct measurements, under every fold "
    "scheme. A2: no reading is admissible as a ratio denominator while "
    "unidentified, so the two criteria are nested rather than crossing. A3: no "
    "reading admitted by the shipped floor is refused by the corrected rule. The "
    "rate over cohort draws is reported and is deliberately not a condition: 24 "
    "units cannot resolve the target, so a criterion on it would be decided by a "
    "single reading"
)


def _adoption_criterion(
    schemes: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    *,
    threshold: float,
    target: float,
) -> dict[str, Any]:
    """The pre-registered conditions, evaluated and reported as a whole.

    Written as one function so that the verdict cannot be assembled by a reader
    picking whichever of the three conditions the numbers happen to satisfy. A
    condition that cannot be evaluated -- no scored scheme, no real row carrying
    a standard error -- makes the verdict ``INDETERMINATE`` rather than ``ADOPT``.
    """

    rule = "candidate_displacement_corrected_rule_at_zero"
    per_scheme = [
        {
            "scheme": scheme["scheme"],
            "rate_over_readings": scheme["pooled_held_out_false_positive_rate"][rule][
                "rate_over_readings"
            ],
            "rate_over_distinct_measurements": scheme[
                "pooled_held_out_false_positive_rate"
            ][rule]["rate_over_distinct_measurements"],
            "rate_over_cohort_draws": scheme["pooled_held_out_false_positive_rate"][
                rule
            ]["rate_over_cohort_draws"],
            "exact_upper_95_over_readings": scheme[
                "pooled_held_out_false_positive_rate"
            ][rule]["exact_upper_95_over_readings"],
            "exact_upper_95_over_distinct_measurements": scheme[
                "pooled_held_out_false_positive_rate"
            ][rule]["exact_upper_95_over_distinct_measurements"],
            "exact_upper_95_over_cohort_draws": scheme[
                "pooled_held_out_false_positive_rate"
            ][rule]["exact_upper_95_over_cohort_draws"],
            "meets_the_target_over_readings": scheme[
                "pooled_held_out_false_positive_rate"
            ][rule]["meets_the_target_over_readings"],
            "meets_the_target_over_distinct_measurements": scheme[
                "pooled_held_out_false_positive_rate"
            ][rule]["meets_the_target_over_distinct_measurements"],
        }
        for scheme in schemes
    ]
    a1 = bool(per_scheme) and all(
        entry["meets_the_target_over_readings"]
        and entry["meets_the_target_over_distinct_measurements"]
        for entry in per_scheme
    )

    scorable = [
        row
        for row in real_rows
        if row["context_information_nats"] is not None
        and row["bootstrap_se"] is not None
        and row["per_arm_identification_status"] in ("PASS", "FAIL")
    ]
    admissible_but_unidentified = [
        {
            "arm": row["arm"],
            "block_id": row["block_id"],
            "context_information_nats": row["context_information_nats"],
            "bootstrap_se": row["bootstrap_se"],
        }
        for row in scorable
        if ratio_denominator_admissibility(
            row["context_information_nats"], row["bootstrap_se"]
        )["admissible"]
        and row["per_arm_identification_status"] != "PASS"
    ]
    lost = [
        {
            "arm": row["arm"],
            "block_id": row["block_id"],
            "context_information_nats": row["context_information_nats"],
            "displacement_corrected_ci_95": row["displacement_corrected_ci_95"],
        }
        for row in scorable
        if row["context_information_nats"] >= threshold
        and row["per_arm_identification_status"] != "PASS"
    ]
    a2 = bool(scorable) and not admissible_but_unidentified
    a3 = bool(scorable) and not lost
    evaluable = bool(per_scheme) and bool(scorable)
    verdict = "INDETERMINATE" if not evaluable else ("ADOPT" if a1 and a2 and a3 else "DECLINE")
    return {
        "preregistration": (
            "archive/docs/R5/context-measurement-protocols.md"
        ),
        "note": ADOPTION_CRITERION_NOTE,
        "candidate_rule": (
            "the displacement-corrected 95% interval for I lies strictly above zero"
        ),
        "target_false_positive_rate": float(target),
        "shipped_floor_nats": float(threshold),
        "n_real_readings_scored": len(scorable),
        "A1_held_out_false_positive_rate": {
            "holds": a1,
            "by_scheme": per_scheme,
        },
        "A2_nested_with_ratio_admissibility": {
            "holds": a2,
            "n_admissible_but_unidentified": len(admissible_but_unidentified),
            "readings_admissible_but_unidentified": admissible_but_unidentified,
        },
        "A3_no_admitted_reading_is_lost": {
            "holds": a3,
            "n_lost": len(lost),
            "readings_lost": lost,
        },
        "verdict": verdict,
        "adopted": verdict == "ADOPT",
    }


def held_out_threshold_validation(
    null_readings: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    *,
    threshold: float,
    confidence: float,
    target_false_positive_rate: float = HELD_OUT_TARGET_FALSE_POSITIVE_RATE,
) -> dict[str, Any]:
    """Cross-fitted validation of the identification floor on units held out of its derivation.

    Two schemes, because two different generalisations are at stake and the panel
    supplies a unit for each. Holding out a **cohort draw** asks whether a
    threshold derived on seven draws controls its error rate on an eighth, which
    is the sampling question. Holding out a **tokenisation class** asks whether a
    threshold derived without ever seeing a vocabulary controls its error rate on
    that vocabulary, which is the question a single constant in nats per token
    actually raises: the null's magnitude is set by the smoothing constant, and
    the smoothing constant is a property of the inventory.

    Holding out an *arm* is not among them, and the reason is measured rather
    than asserted: arms sharing a block and a vocabulary carry one and the same
    reading, so a fold that held out ``gpt2`` would leave four bit-identical
    copies of its held-out readings in the calibration fold and would report a
    generalisation it never tested. ``null_reading_equivalence`` reports the
    collapse it rests on.

    The cohort-draw scheme is run twice, once with the blocks coupled to the
    held-out one by the control rotation withheld from the calibration fold as
    well. A control borrows the *next* block's reference, so a held-out block's
    reading is a function of a calibration block's reference; the buffered
    variant is what says whether that coupling is carrying the result.
    """

    usable = [
        reading
        for reading in null_readings
        if reading.get("measured_information_nats") is not None
        and reading.get("bootstrap_ci_95") is not None
        and reading.get("displacement_corrected_ci_95") is not None
        and reading.get("bootstrap_se")
    ]
    real_rows = [
        row
        for row in arm_rows
        if not row["is_unigram_null_control"]
        and row["context_information_nats"] is not None
    ]
    if len(usable) < 2:
        return {
            "requested": True,
            "refused": True,
            "refusal_reason": (
                f"{len(usable)} measured null control readings; a held-out "
                "validation needs at least two units and this is not enough to "
                "form one fold, let alone report a rate"
            ),
            "note": HELD_OUT_NOTE,
        }

    equivalence = null_reading_equivalence(usable)
    class_of = equivalence["arm_to_tokenisation_class"]

    borrowed_from = {
        reading["block_id"]: reading["control_block_id"] for reading in usable
    }

    def rotation_buffer(block_id: str) -> set[str]:
        coupled = {block_id}
        source = borrowed_from.get(block_id)
        if source is not None:
            coupled.add(source)
        coupled |= {
            other for other, source in borrowed_from.items() if source == block_id
        }
        return coupled

    schemes = [
        _held_out_scheme(
            "leave_one_cohort_draw_out",
            "does a threshold derived on the other cohort draws control its "
            "false-positive rate on a draw it never saw?",
            usable,
            real_rows,
            lambda reading: reading["block_id"],
            lambda row: row["block_id"],
            lambda unit: {unit},
            threshold=threshold,
            confidence=confidence,
            target=target_false_positive_rate,
        ),
        _held_out_scheme(
            "leave_one_cohort_draw_out_buffered_against_the_control_rotation",
            "the same question with the blocks coupled to the held-out draw by "
            "the borrowed reference withheld from the derivation as well, so the "
            "rotation cannot be what carries the answer",
            usable,
            real_rows,
            lambda reading: reading["block_id"],
            lambda row: row["block_id"],
            rotation_buffer,
            threshold=threshold,
            confidence=confidence,
            target=target_false_positive_rate,
        ),
        _held_out_scheme(
            "leave_one_tokenisation_class_out",
            "does a threshold derived without ever seeing a vocabulary control "
            "its false-positive rate on that vocabulary?",
            usable,
            real_rows,
            lambda reading: class_of[reading["arm"]],
            lambda row: class_of.get(row["arm"], ""),
            lambda unit: {unit},
            threshold=threshold,
            confidence=confidence,
            target=target_false_positive_rate,
        ),
    ]

    scored = [scheme for scheme in schemes if not scheme["refused"]]
    adoption = _adoption_criterion(
        scored, real_rows, threshold=threshold, target=target_false_positive_rate
    )
    shipped_holds = all(
        scheme["pooled_held_out_false_positive_rate"]["shipped_point_rule"][
            "n_false_positives"
        ]
        == 0
        for scheme in scored
    )
    recalibrated_holds = all(
        scheme["pooled_held_out_false_positive_rate"]["recalibrated_point_rule"][
            "meets_the_target_over_readings"
        ]
        for scheme in scored
    )
    never_exceeds = all(
        scheme["recalibrated_tau_nats"]["point_rule"]["max"] <= threshold
        for scheme in scored
    )
    verdicts_agree = all(
        scheme["real_arm_verdict_agreement"]["agrees_everywhere"] for scheme in scored
    )
    return {
        "requested": True,
        "refused": False,
        "note": HELD_OUT_NOTE,
        "preregistration": "archive/docs/R5/context-measurement-protocols.md",
        "shipped_floor_nats": float(threshold),
        "target_false_positive_rate": float(target_false_positive_rate),
        "tau_grid_step_nats": HELD_OUT_TAU_GRID[1] - HELD_OUT_TAU_GRID[0],
        "unit_accounting": equivalence,
        "displacement_corrected_identification": adoption,
        "schemes": schemes,
        "conclusion": {
            "shipped_floor_admits_no_held_out_null": shipped_holds,
            "recalibrated_threshold_meets_its_own_target_out_of_sample":
                bool(recalibrated_holds),
            "no_fold_asks_for_more_than_the_shipped_floor": bool(never_exceeds),
            "real_arm_verdicts_are_unchanged": bool(verdicts_agree),
            "statement": _held_out_statement(
                scored,
                threshold=threshold,
                target=target_false_positive_rate,
                shipped_holds=shipped_holds,
                recalibrated_holds=bool(recalibrated_holds),
                never_exceeds=bool(never_exceeds),
                verdicts_agree=bool(verdicts_agree),
            ),
        },
    }


def _held_out_statement(
    schemes: list[dict[str, Any]],
    *,
    threshold: float,
    target: float,
    shipped_holds: bool,
    recalibrated_holds: bool,
    never_exceeds: bool,
    verdicts_agree: bool,
) -> str:
    """What the folds say, written from the folds and not from an expectation."""

    if not schemes:
        return "no scheme produced a scored fold, so nothing was validated"
    worst = max(
        scheme["pooled_held_out_false_positive_rate"]["recalibrated_point_rule"][
            "rate_over_readings"
        ]
        for scheme in schemes
    )
    bound = max(
        scheme["pooled_held_out_false_positive_rate"]["shipped_point_rule"][
            "exact_upper_95_over_distinct_measurements"
        ]
        for scheme in schemes
    )
    largest_tau = max(
        scheme["recalibrated_tau_nats"]["point_rule"]["max"] for scheme in schemes
    )
    parts = [
        (
            f"the shipped {threshold:g} nats/token floor admits no held-out null "
            f"reading under any scheme, with a one-sided 95% upper limit of "
            f"{bound:.4f} on its false-positive rate over distinct measurements"
            if shipped_holds
            else f"the shipped {threshold:g} nats/token floor admits held-out null "
            "readings, so it does not survive this validation"
        ),
        (
            "the threshold re-derived on each calibration fold reaches a pooled "
            f"held-out rate of {worst:.4f} against its own {target:g} target"
            + (", which it meets" if recalibrated_holds else ", which it fails")
        ),
        (
            f"no fold's re-derived threshold exceeds the shipped floor (largest "
            f"{largest_tau:g} nats)"
            if never_exceeds
            else "some fold asks for a threshold above the shipped floor, which is "
            "the direction that would make the shipped floor too lax"
        ),
        (
            "no real arm changes verdict between a fold's threshold and the "
            "shipped floor"
            if verdicts_agree
            else "at least one real arm changes verdict between a fold's threshold "
            "and the shipped floor"
        ),
    ]
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# The comparison table
# --------------------------------------------------------------------------- #


def summarise(
    rows: list[dict[str, Any]],
    *,
    expected_arms: list[str],
    alpha_records: list[dict[str, Any]],
    leakage_records: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """One row per arm: the primary result, the sensitivities, and what changes.

    An arm with no record at all is named here as ``ABSENT``. Four of the fifteen
    panel arms have never been through the held-out estimator, and an empty cell
    in a comparison table is read as a pass far more often than it is read as a
    gap; naming it is the only thing that stops that.
    """

    control_arms = {
        row["arm"] for row in rows if row.get("is_unigram_null_control")
    }
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    alpha_by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in alpha_records:
        alpha_by_arm.setdefault(record["arm"], []).append(record)
    leakage_by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in leakage_records:
        leakage_by_arm.setdefault(record["arm"], []).append(record)

    entries: list[dict[str, Any]] = []
    for arm in sorted(set(expected_arms) | set(by_arm)):
        measured = by_arm.get(arm, [])
        verdicts = [row["screening_status"] for row in measured]
        passes = [v for v in verdicts if v == "PASS"]
        fails = [v for v in verdicts if v == "FAIL"]
        if not measured:
            status = "ABSENT"
        elif not passes and not fails:
            status = "REFUSED"
        elif passes and fails:
            status = "INCONSISTENT"
        elif passes:
            status = "PASS"
        else:
            status = "FAIL"
        ranges = [
            record["information_range_nats"]
            for record in alpha_by_arm.get(arm, [])
            if record["information_range_nats"] is not None
        ]
        entries.append(
            {
                "arm": arm,
                "present": bool(measured),
                "status": status,
                "absent_reason": (
                    None
                    if measured
                    else (
                        "no supplied cohort-power sidecar carries this arm, so it "
                        "has not been measured on this estimand here. Absence is "
                        "not a pass"
                    )
                ),
                "blocks": [
                    {
                        "block_id": row["block_id"],
                        "cohort_name": row["cohort_name"],
                        "cohort_digest": row["cohort_digest"],
                        "context_information_nats": row["context_information_nats"],
                        "bootstrap_ci_95": row["bootstrap_ci_95"],
                        "screening_status": row["screening_status"],
                        "screening_interval_status": row["screening_interval_status"],
                        "sign_status": row["sign_status"],
                        "status_disagreement": row["status_disagreement"],
                        "refused": row["refused"],
                        "refusal_reason": row["refusal_reason"],
                    }
                    for row in sorted(measured, key=lambda row: row["block_id"])
                ],
                "is_unigram_null_control": arm in control_arms,
                "alpha_sensitivity_range_nats": (None if not ranges else float(max(ranges))),
                "leakage_removed": [
                    {
                        "block_id": record["block_id"],
                        "available": record["available"],
                        "information_nats": record["information_nats"],
                        "delta_vs_headline_nats": record["delta_vs_headline_nats"],
                        "reason": record["reason"],
                    }
                    for record in sorted(
                        leakage_by_arm.get(arm, []), key=lambda record: record["block_id"]
                    )
                ],
                "sign_status_is_evidential": False,
            }
        )

    # The synthetic null controls are excluded from the panel-level conclusion
    # and reported in their own section. They are not arms of the panel and an
    # eligibility verdict on one is a statement about the criterion rather than
    # about a model, so folding them in here would put a designed zero into a
    # table that reads as a measurement of the panel.
    panel_rows = [row for row in rows if row["arm"] not in control_arms]
    measured_rows = [
        row for row in panel_rows if row["context_information_nats"] is not None
    ]
    straddling = sorted(
        {
            row["arm"]
            for row in measured_rows
            if row["screening_interval_status"] == "straddles"
        }
    )
    disagreeing = sorted({row["arm"] for row in measured_rows if row["status_disagreement"]})
    refused = sorted({row["arm"] for row in panel_rows if row["refused"]})
    absent = sorted(entry["arm"] for entry in entries if not entry["present"])
    return {
        "threshold_nats": float(threshold),
        "unigram_null_control_arms": sorted(control_arms),
        "control_note": (
            "an arm named here is a synthetic unigram null control, not a model; "
            "its verdicts are excluded from the panel-level statements below and "
            "reported under unigram_null_control"
        ),
        "operative_gate": (
            "identification: budget.context_identification, which asks whether "
            "the arm's displacement-corrected 95% interval for I lies strictly "
            "above zero. It is per-arm and has no constant in it. It says the "
            "arm read above no-context and NOT that its reading may be divided "
            "by; that is budget.ratio_denominator_admissibility, which is "
            "strictly stronger and therefore nested inside it. The "
            f"{SCREENING_CONTEXT_INFORMATION_NATS:g}-nat screening floor is "
            "reported beside every reading as screening_status and decides "
            "nothing here"
        ),
        "screening_floor_note": SCREENING_FLOOR_NOTE,
        "legacy_minimum_context_information_nats": MIN_CONTEXT_INFORMATION_NATS,
        "legacy_floor_note": LEGACY_FLOOR_NOTE,
        "arms": entries,
        "arms_with_no_record": absent,
        "absence_note": (
            "an arm listed here has no record in any supplied sidecar. It has not "
            "been measured on this estimand and its absence is not a pass"
        ),
        "arms_with_a_refused_interval": refused,
        "arms_whose_interval_straddles_the_threshold": straddling,
        "arms_where_the_sign_rule_disagrees_with_the_floor": disagreeing,
        "conclusion": {
            "interval_alters_the_unmeasurable_set": bool(straddling),
            "interval_statement": (
                f"no arm's interval straddles the {threshold:g} nats/token "
                "screening floor, so reporting the floor with an interval does not "
                "move any arm across it on these draws"
                if not straddling
                else f"the interval crosses the {threshold:g} nats/token screening "
                "floor for "
                + ", ".join(straddling)
                + ", so for those arms the point comparison and the interval do not "
                "agree about measurability and the point verdict is not supported "
                "by this draw alone"
            ),
            "sign_rule_alters_the_unmeasurable_set": bool(disagreeing),
            "sign_rule_statement": (
                "the sign criterion agrees with the floor on every measured arm "
                "here, which is what the pre-registration expects and is not "
                "evidence for either"
                if not disagreeing
                else "the sign criterion and the floor disagree on "
                + ", ".join(disagreeing)
                + "; the floor remains the operative gate and the sign criterion is "
                "NOT adopted, so the existing conclusion about which arms are "
                "unmeasurable is unchanged"
            ),
            "sign_rule_is_adopted": False,
            "note": SCREENING_NOTE,
        },
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar",
        type=Path,
        nargs="+",
        required=True,
        help="power_<cohort>_<digest>.records.npz files, one per cohort item; "
        "the ones carrying the same cohort and reference digests are one draw "
        "and are bootstrapped under one resample index",
    )
    parser.add_argument(
        "--cohort-json",
        type=Path,
        nargs="*",
        default=[],
        help="frozen-cohort files, aligned with --sidecar; derived from each "
        "sidecar's name when omitted. They carry the records the near-duplicate "
        "grouping is computed from, and without them the resampling unit falls "
        "back to the record with that limitation declared in every affected row",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        nargs="*",
        default=[],
        help="held-out reference records, aligned with --sidecar: the "
        "reference_<name>_<digest>.json cohort_power.py writes beside the "
        "cohort. Both the reference grouping and the leakage-removed "
        "sensitivity are unavailable without it",
    )
    parser.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="restrict to these arms; every arm in every sidecar by default",
    )
    parser.add_argument(
        "--expected-arms",
        nargs="*",
        default=sorted(PANEL),
        help="arms the summary table must account for, so that an arm with no "
        "record is named as absent rather than left out",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--threshold-nats",
        type=float,
        default=SCREENING_CONTEXT_INFORMATION_NATS,
        help="the pre-interval screening floor, reported against every reading "
        "as screening_status and deciding nothing here: the operative verdict is "
        "per_arm_identification_status, from budget.context_identification. It "
        "is not a denominator criterion either: whether a reading may be divided "
        "by is budget.ratio_denominator_admissibility, a per-arm bound on that "
        "arm's own standard error, which this stage publishes as bootstrap_se",
    )
    parser.add_argument(
        "--containment",
        type=float,
        default=NEAR_DUPLICATE_CONTAINMENT,
        help="near-duplicate containment threshold for the grouping",
    )
    parser.add_argument(
        "--shingle",
        type=int,
        default=None,
        help="shingle length for the grouping; the unit's declared default when omitted",
    )
    parser.add_argument(
        "--alpha-sweep",
        type=float,
        nargs="*",
        default=list(SMOOTHING_SWEEP),
        help="smoothing ladder for E4; pathways.SMOOTHING_SWEEP by default",
    )
    parser.add_argument(
        "--leakage-kmer",
        type=int,
        default=DEFAULT_LEAKAGE_KMER,
        help="k of the k-mer the E5 reference screen uses",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        default=None,
        help="an extra cross-cohort pair to difference; repeatable. Pairs within "
        "one cohort are always formed and need not be named",
    )
    parser.add_argument(
        "--unigram-null-control",
        action="store_true",
        help="add one synthetic control arm per measurable arm per block, whose "
        "predictive distribution is a smoothed unigram fitted on the held-out "
        "reference of a DIFFERENT supplied block. Its true context information "
        "is zero by construction, which is the only point at which the "
        "eligibility criteria can be watched behaving at a known zero. Needs at "
        "least two blocks carrying the arm with different reference digests, and "
        "names the two blocks in every record",
    )
    parser.add_argument(
        "--held-out-threshold-validation",
        action="store_true",
        help="cross-fit the identification floor's calibration: on every fold "
        "re-derive the threshold from the calibration units alone, by "
        "EXP-R2-218's own rule, and score it -- and the shipped floor beside it "
        "-- on units the derivation never saw. Needs --unigram-null-control, "
        "because the units are the readings whose true information is zero",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--report-name", default="context_information_bootstrap.json"
    )
    args = parser.parse_args(argv)
    for name, values in (("--cohort-json", args.cohort_json), ("--reference-json", args.reference_json)):
        if values and len(values) != len(args.sidecar):
            raise SystemExit(
                f"{name} takes one path per --sidecar ({len(args.sidecar)}); "
                f"got {len(values)}"
            )
    if not args.alpha_sweep or any(value <= 0.0 for value in args.alpha_sweep):
        raise SystemExit("--alpha-sweep must name at least one positive constant")
    if args.leakage_kmer < 1:
        raise SystemExit("--leakage-kmer must be positive")
    if args.held_out_threshold_validation and not args.unigram_null_control:
        raise SystemExit(
            "--held-out-threshold-validation cross-fits the calibration of the "
            "identification floor over readings whose true context information "
            "is zero, and those readings are the null controls; pass "
            "--unigram-null-control as well"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()

    # Every block is loaded before any is bootstrapped, because the null control
    # is built from a *different* block's reference and cannot be assembled until
    # the candidates are known.
    # One cohort draw can arrive in several files. ``--dtype float32`` is
    # process-global, so an arm needing it is declared as its own cohort item and
    # scored by its own invocation -- against the same records and the same
    # reference as the items beside it, which is what the two digests say and
    # what pairing follows. The sidecars are sorted into draws before any of them
    # is read, so a draw's near-duplicate grouping and its resample indices are
    # computed once and are the same for every arm of it.
    supplied = [
        (
            Path(sidecar),
            Path(args.cohort_json[index]) if args.cohort_json else None,
            Path(args.reference_json[index]) if args.reference_json else None,
        )
        for index, sidecar in enumerate(args.sidecar)
    ]
    draws: dict[tuple[str, str], list[tuple[Path, Path | None, Path | None]]] = {}
    for entry in supplied:
        draws.setdefault(draw_key(entry[0]), []).append(entry)
    blocks = [
        load_block(
            index,
            entries,
            requested_arms=list(args.arms) if args.arms else None,
            containment=float(args.containment),
            shingle=args.shingle,
        )
        for index, entries in enumerate(draws.values())
    ]
    controls: list[NullControl] = (
        [control for block in blocks for control in null_controls(block, blocks)]
        if args.unigram_null_control
        else []
    )
    controls_of: dict[str, list[NullControl]] = {}
    for control in controls:
        controls_of.setdefault(control.block_id, []).append(control)

    panels: list[Panel] = []
    seed_cursor = int(args.seed)
    for block in blocks:
        block_panels = bootstrap_block(
            block,
            seed_cursor,
            n_bootstrap=int(args.n_bootstrap),
            confidence=float(args.confidence),
            extra=[
                control.statistics
                for control in controls_of.get(block.block_id, [])
                if control.statistics is not None
            ],
        )
        seed_cursor += max(len(block_panels), 1)
        panels.extend(block_panels)

    panel_of: dict[tuple[str, str], Panel] = {
        (panel.block.block_id, name): panel for panel in panels for name in panel.names
    }
    rows: list[dict[str, Any]] = []
    for block in blocks:
        inputs = {entry.name: entry for entry in block.arms}
        for entry in block.arms:
            if entry.statistics is None:
                rows.append(
                    arm_row(
                        block,
                        entry,
                        None,
                        load_refusal_record(entry.name, entry.load_refusal or "unstated"),
                        threshold=float(args.threshold_nats),
                    )
                )
                continue
            panel = panel_of[(block.block_id, entry.name)]
            rows.append(
                arm_row(
                    block,
                    entry,
                    panel.panel_id,
                    panel.result.arms[entry.name].record,
                    threshold=float(args.threshold_nats),
                )
            )
        for control in controls_of.get(block.block_id, []):
            if control.statistics is None:
                continue
            panel = panel_of[(block.block_id, control.name)]
            rows.append(
                arm_row(
                    block,
                    inputs[control.arm],
                    panel.panel_id,
                    panel.result.arms[control.name].record,
                    threshold=float(args.threshold_nats),
                    is_control=True,
                )
            )

    requested_pairs = [(left, right) for left, right in (args.contrast or [])]
    within = within_cohort_contrasts(panels)
    across = cross_cohort_contrasts(
        panels, confidence=float(args.confidence), requested=requested_pairs
    )
    alpha_records = alpha_sensitivity(
        panels,
        sweep=[float(value) for value in args.alpha_sweep],
        n_bootstrap=int(args.n_bootstrap),
        confidence=float(args.confidence),
    )
    leakage_records = leakage_sensitivity(
        panels,
        k=int(args.leakage_kmer),
        n_bootstrap=int(args.n_bootstrap),
        confidence=float(args.confidence),
    )
    payload_control = (
        null_control_summary(
            controls,
            rows,
            threshold=float(args.threshold_nats),
            confidence=float(args.confidence),
        )
        if args.unigram_null_control
        else {
            "requested": False,
            "note": (
                "no null control was requested; pass --unigram-null-control with "
                "at least two blocks carrying an arm against different held-out "
                "references to measure the criteria at a known zero"
            ),
        }
    )
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256_file(Path(__file__)),
            "module_sha256": {
                name: sha256_file(REPO_ROOT / name) for name in HASHED_MODULES
            },
            "preregistration": "archive/docs/R5/context-measurement-protocols.md",
            "estimand": ESTIMAND,
            "screening_note": SCREENING_NOTE,
            "comparability_note": COMPARABILITY_NOTE,
            "sign_status_note": SIGN_STATUS_NOTE,
            "minimum_bootstrap_units": int(MINIMUM_BOOTSTRAP_UNITS),
            "interval_field_note": (
                "bootstrap_ci_95 carries the nominal 95% default this analysis is "
                "specified at; when --confidence is moved the interval is at the "
                "level the confidence field beside it states, and that field is "
                "the authoritative one"
            ),
            "seeds": {
                "base": int(args.seed),
                "panels": {panel.panel_id: panel.seed for panel in panels},
                "leakage_offset": LEAKAGE_SEED_OFFSET,
                "producer_seeds": {
                    block.block_id: {
                        str(sidecar.path): sidecar.producer_seeds
                        for sidecar in block.sidecars
                    }
                    for block in blocks
                },
            },
            "configuration": {
                key: (
                    [str(item) for item in value]
                    if isinstance(value, list) and value and isinstance(value[0], Path)
                    else str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in sorted(vars(args).items())
            },
        },
        "blocks": [
            {
                "block_id": block.block_id,
                "cohort_digest": block.cohort_digest,
                "cohort_names": block.cohort_names,
                "cohort_kind": block.cohort_kind,
                "reference_digest": block.reference_digest,
                # One entry per producer invocation. Several of them means
                # several cohort items scored this one draw, which the digests
                # above establish and which is why their arms are paired.
                "sidecars": [sidecar.record for sidecar in block.sidecars],
                "arms_in_block": [entry.name for entry in block.arms],
                "arms_refused_at_load": {
                    entry.name: entry.load_refusal
                    for entry in block.arms
                    if entry.load_refusal is not None
                },
                "reference_records_dropped_empty": {
                    entry.name: entry.reference_records_dropped_empty
                    for entry in block.arms
                },
                "cohort_grouping": block.grouping,
                "reference_grouping": block.reference_grouping,
                "panels": [
                    {
                        "panel_id": panel.panel_id,
                        "arms": panel.names,
                        "seed": panel.seed,
                        "common_resample_indices": True,
                    }
                    for panel in panels
                    if panel.block.block_id == block.block_id
                ],
            }
            for block in blocks
        ],
        "arm_results": rows,
        "contrasts": {
            "within_cohort_paired": within,
            "cross_cohort_unpaired": across,
            "same_cohort_unpairable": cross_panel_refusals(panels),
            "declared": declared_contrast_sign_tracking(within, across),
            "note": (
                "pairing is defined within a cohort and nowhere else; a contrast "
                "that crosses cohorts is reported as unpaired with that fact "
                "attached, and non-overlap of two independently bootstrapped "
                "intervals is not a test of difference"
            ),
        },
        "between_block_variance": between_block_variance(
            rows, {block.block_id: block for block in blocks}
        ),
        "alpha_sensitivity": {
            "sweep": [float(value) for value in args.alpha_sweep],
            "records": alpha_records,
            "note": (
                "E4. Smoothing is applied at analysis time from the persisted "
                "reference counts, so this sweep is CPU arithmetic over the same "
                "statistics the headline is computed from. It replaces the "
                "withdrawn figures in pathways.LAPLACE_SMOOTHING with measurement "
                "on the corpora actually used"
            ),
        },
        "leakage_removed_sensitivity": {
            "kmer": int(args.leakage_kmer),
            "records": leakage_records,
            "note": (
                "E5. A sensitivity beside the headline, not a correction to it; "
                "the headline stays on the declared reference"
            ),
        },
        "unigram_null_control": payload_control,
        "held_out_threshold_validation": (
            held_out_threshold_validation(
                payload_control["readings"],
                rows,
                threshold=float(args.threshold_nats),
                confidence=float(args.confidence),
            )
            if args.held_out_threshold_validation
            else {
                "requested": False,
                "note": (
                    "no held-out validation was requested; pass "
                    "--held-out-threshold-validation with --unigram-null-control "
                    "to cross-fit the identification floor's calibration"
                ),
            }
        ),
        "summary": summarise(
            rows,
            expected_arms=list(args.expected_arms),
            alpha_records=alpha_records,
            leakage_records=leakage_records,
            threshold=float(args.threshold_nats),
        ),
    }

    destination = Path(args.out) / args.report_name
    write_json(destination, payload)

    print(
        f"{'arm':18s} {'block':6s} {'I nats/token':>13s} "
        f"{'95% interval':>22s}  {'screen':6s} {'interval':10s} sign*"
    )
    for row in rows:
        if row["context_information_nats"] is None:
            print(
                f"{row['arm']:18s} {row['block_id']:6s} "
                f"{'REFUSED':>13s} {'-':>22s}  {'-':6s} {'-':10s} -"
            )
            continue
        low, high = row["bootstrap_ci_95"]
        print(
            f"{row['arm']:18s} {row['block_id']:6s} "
            f"{row['context_information_nats']:+13.4f} "
            f"[{low:+9.4f}, {high:+9.4f}]  "
            f"{row['screening_status']:6s} "
            f"{row['screening_interval_status']:10s} "
            f"{row['sign_status']}"
        )
    print("* sign_status is NON-EVIDENTIAL and is expected to pass on every arm")
    control = payload["unigram_null_control"]
    if control.get("requested"):
        criteria = control["criteria_at_a_known_zero"]
        print(
            f"null control: {control['n_controls_measured']} of "
            f"{control['n_controls_requested']} measured; at a true zero the floor "
            f"passes {criteria['n_floor_pass_readings']} of {criteria['n_readings']} "
            f"readings and the sign rule passes {criteria['n_sign_pass_readings']}, "
            f"against the {criteria['expected_false_pass_rate_at_a_true_zero']:.3f} "
            "the interval's own tail would give"
        )
        for entry in control["by_cohort"]:
            print(
                f"  {entry['cohort_name']:32s} n={entry['n_readings']:3d} "
                f"mean departure {entry['mean_departure_nats']:+8.4f} nats, "
                f"range [{entry['min_departure_nats']:+.4f}, "
                f"{entry['max_departure_nats']:+.4f}], "
                f"sign PASS {entry['n_sign_pass']}, floor PASS {entry['n_floor_pass']}"
            )
    validation = payload["held_out_threshold_validation"]
    if validation.get("requested") and not validation.get("refused"):
        accounting = validation["unit_accounting"]
        print(
            f"held-out validation: {accounting['n_readings']} null readings are "
            f"{accounting['n_distinct_measurements']} distinct measurements over "
            f"{accounting['n_cohort_draws']} cohort draws and "
            f"{len(accounting['tokenisation_classes'])} tokenisation classes"
        )
        for scheme in validation["schemes"]:
            if scheme["refused"]:
                print(f"  {scheme['scheme']}: REFUSED -- {scheme['refusal_reason']}")
                continue
            pooled = scheme["pooled_held_out_false_positive_rate"]
            tau = scheme["recalibrated_tau_nats"]["point_rule"]
            print(
                f"  {scheme['scheme']}: {scheme['n_folds']} folds, re-derived tau "
                f"{tau['min']:.3f}-{tau['max']:.3f} nats; held-out FPR "
                f"recalibrated {pooled['recalibrated_point_rule']['rate_over_readings']:.4f}, "
                f"shipped point {pooled['shipped_point_rule']['rate_over_readings']:.4f}, "
                f"shipped interval {pooled['shipped_interval_rule']['rate_over_readings']:.4f}, "
                f"Fieller "
                f"{pooled['shipped_fieller_denominator_admissibility']['rate_over_readings']:.4f}; "
                f"real arms changing verdict "
                f"{scheme['real_arm_verdict_agreement']['n_changing_verdict']}"
            )
        print(f"  {validation['conclusion']['statement']}")
    absent = payload["summary"]["arms_with_no_record"]
    if absent:
        print(f"arms with no record at all (absence is not a pass): {', '.join(absent)}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
