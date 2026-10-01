"""EXP-R2-247: rank association among gate, scoring, generation, and size.

Each checkpoint carries one frozen value per axis: G1 (gate), S1-S3 (scoring),
P (parameter count), U1-U5 (unconditional generation) and C2-C5 (conditional
generation). A cell is available with a recorded value, pending, or unavailable
with a stated reason. For every declared pair of axes, Spearman rank correlation
is computed across the checkpoints whose two cells are both available; n=2
yields +1 or -1. A pair with any pending cell gets no coefficient, and a missing
cell is never filled in. With n >= 5 the record adds a 95% percentile interval
from 4000 checkpoint-level bootstrap resamples (seed 20260918) and the
leave-one-out coefficients; with n = 4 only the leave-one-out coefficients.
The same pairs are repeated within each model lineage. The result is descriptive: it supports no
parameter-count causal claim and no model ranking, conditional and unconditional
generation are kept apart, and homologous-context AUROC is not an axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

CAMPAIGN = "EXP-R2-247"
EXPECT_INVENTORY = "cross_measure_inventory.json"
EXPECT_ASSOCIATION = "cross_measure_association.json"
SEED = 20260918
BOOTSTRAP = 4000

AXES = (
    "G1",
    "S1",
    "S2",
    "S3",
    "P",
    "U1",
    "U2",
    "U3",
    "U4",
    "U5",
    "C2",
    "C3",
    "C4",
    "C5",
)


@dataclass(frozen=True)
class Cell:
    """One frozen seat: available, pending, or named unavailable."""

    status: str
    value: float | None = None
    note: str = ""

    def resolved(self) -> bool:
        return self.status == "available" and self.value is not None

    def pending(self) -> bool:
        return self.status == "pending"

    def unavailable(self) -> bool:
        return self.status == "unavailable"


def _c(status: str, value: float | None = None, note: str = "") -> Cell:
    return Cell(status=status, value=value, note=note)


# Already-recorded quantities only, copied from earlier measurements of the same
# checkpoints. G1 is unavailable for the original protein cells because no
# eight-block design envelope was located for them. U1-U5 are the values from the
# EXP-R2-246 ESMFold2 census of generated sequences. The MegaScale design rho
# of Galactica and InstructProtein is recorded elsewhere but is not S3.
CELLS: dict[str, dict[str, Cell]] = {
    "protgpt2": {
        "G1": _c("unavailable", note="no 01-design eight-block min lower located"),
        "S1": _c("available", -0.0143),
        "S2": _c("available", 0.1343),
        "S3": _c("available", 0.0837),
        "P": _c("available", 774_030_080),
        "U1": _c("available", 616 / 800),
        "U2": _c("available", 309 / 800),
        "U3": _c("available", 21.94),
        "U4": _c("available", 0.2961, note="all-attempt lower; upper 0.5236"),
        "U5": _c("available", 43.81),
    },
    "zymctrl": {
        "G1": _c("unavailable", note="no 01-design eight-block min lower located"),
        "S1": _c("unavailable", note="217 assays skipped, no EC"),
        "S2": _c("unavailable", note="217 assays skipped, no EC"),
        "S3": _c("unavailable", note="no EC on a design"),
        "P": _c("unavailable", note="integer parameter count not located"),
        "C2": _c("available", 2488 / 2800),
        "C3": _c("available", 58.12),
        "C4": _c("available", 0.8949),
        "C5": _c("available", 70.00),
    },
    "progen2-small": {
        "G1": _c("unavailable", note="no 01-design eight-block min lower located"),
        "S1": _c("available", -0.0554),
        "S2": _c("available", 0.10116),
        "S3": _c("unavailable", note="natural control failed"),
        "P": _c("available", 151_100_000),
        "U1": _c("available", 8 / 800),
        "U2": _c("available", 197 / 800),
        "U3": _c("available", 8.48),
        "U4": _c("available", 0.2101, note="all-attempt lower; upper 0.2301"),
        "U5": _c("available", 33.50),
    },
    "progen2-base": {
        "G1": _c("unavailable", note="no 01-design eight-block min lower located"),
        "S1": _c("available", 0.0037),
        "S2": _c("available", 0.15311),
        "S3": _c("available", 0.0456),
        "P": _c("available", 764_803_616),
        "U1": _c("available", 45 / 800),
        "U2": _c("available", 205 / 800),
        "U3": _c("available", 9.56),
        "U4": _c("available", 0.2703, note="all-attempt lower; upper 0.3465"),
        "U5": _c("available", 37.25),
    },
    "progen2-medium": {
        "G1": _c("available", 1.1712, note="EXP-R2-224 min lower until 01 envelope"),
        "S1": _c("available", -0.0043),
        "S2": _c("available", 0.1522),
        "S3": _c("available", 0.0280),
        "P": _c("available", 764_803_616),
        "U1": _c("available", 32 / 800),
        "U2": _c("available", 300 / 800),
        "U3": _c("available", 18.61),
        "U4": _c("available", 0.3569, note="all-attempt lower; upper 0.3856"),
        "U5": _c("available", 35.94),
    },
    "progen2-large": {
        "G1": _c("available", 1.1115),
        "S1": _c("available", 0.0122),
        "S2": _c("available", 0.1688),
        "S3": _c("available", 0.0519),
        "P": _c("available", 2_779_400_000),
        "U1": _c("available", 48 / 800),
        "U2": _c("available", 298 / 800),
        "U3": _c("available", 15.03),
        "U4": _c("available", 0.3068, note="all-attempt lower; upper 0.3518"),
        "U5": _c("available", 38.50),
    },
    "progen2-xlarge": {
        "G1": _c("available", 1.4891),
        "S1": _c("available", 0.0039),
        "S2": _c("available", 0.1604),
        "S3": _c("available", 0.0480),
        "P": _c("available", 6_443_600_000),
        "U1": _c("available", 36 / 800),
        "U2": _c("available", 343 / 800),
        "U3": _c("available", 24.36),
        "U4": _c("available", 0.3325, note="all-attempt lower; upper 0.4187"),
        "U5": _c("available", 42.70),
    },
    "progen3-112m": {
        "G1": _c("available", 0.8276),
        "S1": _c("available", -0.0801),
        "S2": _c("available", 0.0688),
        "S3": _c("unavailable", note="natural control failed"),
        "P": _c("available", 112_000_000, note="card name; MoE total vs active undeclared"),
        "U1": _c("available", 413 / 800),
        "U2": _c("available", 252 / 800),
        "U3": _c("available", 16.79),
        "U4": _c("available", 0.2503),
        "U5": _c("available", 32.34),
    },
    "progen3-3b": {
        "G1": _c("available", 1.5228),
        "S1": _c("available", 0.0555),
        "S2": _c("available", 0.2044),
        "S3": _c("available", 0.1189),
        "P": _c("available", 3_000_000_000, note="card name; MoE total vs active undeclared"),
        "U1": _c("available", 501 / 800),
        "U2": _c("available", 496 / 800),
        "U3": _c("available", 36.58),
        "U4": _c("available", 0.5477),
        "U5": _c("available", 43.71),
    },
    "proteinglm-7b-clm": {
        "G1": _c("available", 1.5648),
        "S1": _c("available", 0.03841398352246689),
        "S2": _c("available", 0.19492202682472437),
        "S3": _c("available", 0.1395),
        "P": _c("available", 7_000_000_000, note="card name"),
        "U1": _c("available", 380 / 800),
        "U2": _c("available", 281 / 800),
        "U3": _c("available", 18.24),
        "U4": _c("available", 0.3751),
        "U5": _c("available", 38.62),
    },
    "protgpt3-1.3b": {
        "G1": _c("available", 0.8563327672967896),
        "S1": _c("available", -0.051230104982166104),
        "S2": _c("available", 0.1052779383200914),
        "S3": _c("unavailable", note="natural control failed"),
        "P": _c("available", 1_330_000_000, note="about 1.33B total; active undeclared"),
        "U1": _c("available", 345 / 800),
        "U2": _c("available", 316 / 800),
        "U3": _c("available", 23.24),
        "U4": _c("available", 0.3355),
        "U5": _c("available", 37.35),
    },
    "rita-xl": {
        "G1": _c("available", 1.1942),
        "S1": _c("available", 0.007493170359036331),
        "S2": _c("available", 0.16400121366129383),
        "S3": _c("available", 0.0332),
        "P": _c("available", 1_208_655_872),
        "U1": _c("available", 402 / 800),
        "U2": _c("available", 330 / 800),
        "U3": _c("available", 24.78),
        "U4": _c("available", 0.3755),
        "U5": _c("available", 49.11),
    },
    "galactica-125m": {
        "G1": _c("available", -0.1873, note="not identified; stays in later pairs"),
        "S1": _c("available", -0.3632542659991225),
        "S2": _c("available", -0.2138058083832622),
        "S3": _c("unavailable", note="natural control failed; design ρ recorded, not S3"),
        "P": _c("available", 125_000_000),
        "U1": _c("available", 3 / 800),
        "U2": _c("available", 0 / 800),
        "U3": _c("available", 0.16),
        "U4": _c("unavailable", note="all-attempt interval 0.0000–0.9988; not a readable point"),
        "U5": _c("unavailable", note="0/800 UniRef50 hits; median identity undefined"),
    },
    "galactica-1.3b": {
        "G1": _c("available", 0.0304),
        "S1": _c("available", -0.3071441203133887),
        "S2": _c("available", -0.15769566269752833),
        "S3": _c("unavailable", note="natural control failed; design ρ recorded, not S3"),
        "P": _c("available", 1_300_000_000),
        "U1": _c("available", 105 / 800),
        "U2": _c("available", 88 / 800),
        "U3": _c("available", 8.45),
        "U4": _c("available", 0.1257, note="all-attempt lower; upper 0.3882"),
        "U5": _c("available", 98.82),
    },
    "galactica-6.7b": {
        "G1": _c("available", 0.1196),
        "S1": _c("available", -0.260155939135868),
        "S2": _c("available", -0.11070748152000764),
        "S3": _c("unavailable", note="natural control failed; design ρ recorded, not S3"),
        "P": _c("available", 6_700_000_000),
        "U1": _c("available", 166 / 800),
        "U2": _c("available", 91 / 800),
        "U3": _c("available", 6.75),
        "U4": _c("available", 0.1674, note="all-attempt lower; upper 0.4349"),
        "U5": _c("available", 49.50),
    },
    "galactica-30b": {
        "G1": _c("available", 0.2452),
        "S1": _c("available", -0.20761300804723032),
        "S2": _c("available", -0.05816455043136994),
        "S3": _c("unavailable", note="natural control failed; design ρ recorded, not S3"),
        "P": _c("available", 30_000_000_000),
        "U1": _c("available", 285 / 800),
        "U2": _c("available", 138 / 800),
        "U3": _c("available", 9.02),
        "U4": _c("available", 0.1636, note="all-attempt lower; upper 0.4136"),
        "U5": _c("available", 60.75),
    },
    "instructprotein": {
        "G1": _c("available", 1.8885),
        "S1": _c("available", -0.0754471237598173),
        "S2": _c("available", 0.07400133385604303),
        "S3": _c("unavailable", note="natural control failed; design ρ recorded, not S3"),
        "P": _c("unavailable", note="exact integer not separately declared"),
        "U1": _c("available", 540 / 800),
        "U2": _c("available", 470 / 800),
        "U3": _c("available", 29.15),
        "U4": _c("available", 0.4760),
        "U5": _c("available", 64.77),
    },
    "llama-2-7b": {
        "G1": _c("available", 0.0452),
        "S1": _c("available", -0.3621264061142726),
        "S2": _c("available", -0.2133),
        "S3": _c("unavailable", note="natural control failed"),
        "P": _c("available", 7_000_000_000, note="card name"),
    },
    "prollama-stage-1": {
        "G1": _c("available", 0.4868),
        "S1": _c("available", -0.2192565852112736),
        "S2": _c("available", -0.0704),
        "S3": _c("unavailable", note="natural control failed"),
        "P": _c("available", 7_000_000_000, note="card name"),
        "U1": _c("available", 579 / 800),
        "U2": _c("available", 136 / 800),
        "U3": _c("available", 6.52),
        "U4": _c("available", 0.1728),
        "U5": _c("available", 38.42),
    },
    "prollama": {
        "G1": _c("available", 0.4598),
        "S1": _c("available", -0.20434482627118625),
        "S2": _c("available", -0.0555),
        "S3": _c("unavailable", note="natural control failed"),
        "P": _c("available", 7_000_000_000, note="card name"),
        "C2": _c("available", 380 / 3000),
        "C3": _c("available", 6.99),
        "C4": _c("available", 0.0392, note="all-attempt lower; upper 0.0396"),
        "C5": _c("available", 32.05),
        "U1": _c("available", 743 / 800),
        "U2": _c("available", 36 / 800),
        "U3": _c("available", 6.12),
        "U4": _c("available", 0.2129),
        "U5": _c("available", 36.90),
    },
}

LINEAGES: dict[str, tuple[str, ...]] = {
    "progen2": (
        "progen2-small",
        "progen2-base",
        "progen2-medium",
        "progen2-large",
        "progen2-xlarge",
    ),
    "galactica": (
        "galactica-125m",
        "galactica-1.3b",
        "galactica-6.7b",
        "galactica-30b",
    ),
    "progen3": ("progen3-112m", "progen3-3b"),
    "llama-prollama": ("llama-2-7b", "prollama-stage-1", "prollama"),
}

POOLED_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("P-G1-S1", "G1", "S1"),
    ("P-G1-S2", "G1", "S2"),
    ("P-G1-S3", "G1", "S3"),
    ("P-G1-P", "G1", "P"),
    ("P-S1-S3", "S1", "S3"),
    ("P-S2-S3", "S2", "S3"),
    ("P-S1-P", "S1", "P"),
    ("P-S2-P", "S2", "P"),
    ("P-S3-P", "S3", "P"),
    ("P-G1-U1", "G1", "U1"),
    ("P-G1-U2", "G1", "U2"),
    ("P-G1-U3", "G1", "U3"),
    ("P-G1-U4", "G1", "U4"),
    ("P-G1-U5", "G1", "U5"),
    ("P-S1-U1", "S1", "U1"),
    ("P-S1-U2", "S1", "U2"),
    ("P-S1-U3", "S1", "U3"),
    ("P-S1-U4", "S1", "U4"),
    ("P-S1-U5", "S1", "U5"),
    ("P-S2-U1", "S2", "U1"),
    ("P-S2-U2", "S2", "U2"),
    ("P-S2-U3", "S2", "U3"),
    ("P-S2-U4", "S2", "U4"),
    ("P-S2-U5", "S2", "U5"),
    ("P-S3-U1", "S3", "U1"),
    ("P-S3-U2", "S3", "U2"),
    ("P-S3-U3", "S3", "U3"),
    ("P-S3-U4", "S3", "U4"),
    ("P-S3-U5", "S3", "U5"),
    ("P-P-U1", "P", "U1"),
    ("P-P-U2", "P", "U2"),
    ("P-P-U3", "P", "U3"),
    ("P-P-U4", "P", "U4"),
    ("P-P-U5", "P", "U5"),
    ("P-G1-C2", "G1", "C2"),
    ("P-G1-C3", "G1", "C3"),
    ("P-G1-C4", "G1", "C4"),
    ("P-G1-C5", "G1", "C5"),
    ("P-P-C2", "P", "C2"),
    ("P-P-C3", "P", "C3"),
    ("P-P-C4", "P", "C4"),
    ("P-P-C5", "P", "C5"),
    ("P-S1-C2", "S1", "C2"),
    ("P-S1-C3", "S1", "C3"),
    ("P-S1-C4", "S1", "C4"),
    ("P-S1-C5", "S1", "C5"),
    ("P-S2-C2", "S2", "C2"),
    ("P-S2-C3", "S2", "C3"),
    ("P-S2-C4", "S2", "C4"),
    ("P-S2-C5", "S2", "C5"),
    ("P-S3-C2", "S3", "C2"),
)


def cell(arm: str, axis: str) -> Cell:
    row = CELLS.get(arm, {})
    return row.get(axis, _c("unavailable", note="axis not seated for this checkpoint"))


def pair_points(
    arms: Sequence[str],
    x: str,
    y: str,
    *,
    table: Mapping[str, Mapping[str, Cell]] | None = None,
) -> tuple[list[str], list[float], list[float], list[str], list[str]]:
    """Resolved pairs, plus named pending and unavailable drops."""

    used: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    pending: list[str] = []
    dropped: list[str] = []
    lookup = table if table is not None else CELLS
    for arm in arms:
        left = lookup.get(arm, {}).get(x, _c("unavailable", note="axis not seated"))
        right = lookup.get(arm, {}).get(y, _c("unavailable", note="axis not seated"))
        if left.pending() or right.pending():
            pending.append(arm)
            continue
        if left.resolved() and right.resolved():
            used.append(arm)
            xs.append(float(left.value))
            ys.append(float(right.value))
            continue
        dropped.append(arm)
    return used, xs, ys, pending, dropped


def spearman_rank(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation, including the n=2 ±1 case the freeze allows."""

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size != right.size or left.size < 2:
        return None
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    if left.min() == left.max() or right.min() == right.max():
        return None
    if left.size == 2:
        return 1.0 if (left[0] - left[1]) * (right[0] - right[1]) > 0 else -1.0
    value = float(stats.spearmanr(left, right).statistic)
    return None if not np.isfinite(value) else value


def leave_one_out(x: Sequence[float], y: Sequence[float]) -> list[float | None]:
    values: list[float | None] = []
    for index in range(len(x)):
        xs = [value for i, value in enumerate(x) if i != index]
        ys = [value for i, value in enumerate(y) if i != index]
        values.append(spearman_rank(xs, ys))
    return values


def checkpoint_bootstrap(
    x: Sequence[float],
    y: Sequence[float],
    *,
    seed: int = SEED,
    resamples: int = BOOTSTRAP,
) -> tuple[float, float] | None:
    if len(x) < 5:
        return None
    rng = np.random.default_rng(seed)
    n = len(x)
    draws: list[float] = []
    for _ in range(resamples):
        index = rng.integers(0, n, size=n)
        rho = spearman_rank([x[i] for i in index], [y[i] for i in index])
        if rho is not None:
            draws.append(rho)
    if len(draws) < 100:
        return None
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def associate_pair(
    pair_id: str,
    x: str,
    y: str,
    arms: Sequence[str],
    *,
    table: Mapping[str, Mapping[str, Cell]] | None = None,
) -> dict[str, Any]:
    used, xs, ys, pending, dropped = pair_points(arms, x, y, table=table)
    n = len(used)
    record: dict[str, Any] = {
        "pair_id": pair_id,
        "x": x,
        "y": y,
        "arms": list(used),
        "n": n,
        "pending": list(pending),
        "named_drop": list(dropped),
        "spearman": None,
        "interval": None,
        "leave_one_out": None,
        "computed": False,
        "reason": "",
    }
    if pending:
        record["reason"] = "pending cells remain; coefficient withheld"
        return record
    if n <= 1:
        record["reason"] = "n<=1; no coefficient"
        return record
    rho = spearman_rank(xs, ys)
    record["spearman"] = rho
    record["computed"] = rho is not None
    if rho is None:
        record["reason"] = "constant side or undefined ranks"
        return record
    if n >= 5:
        interval = checkpoint_bootstrap(xs, ys)
        record["interval"] = None if interval is None else {"lo": interval[0], "hi": interval[1]}
        record["leave_one_out"] = leave_one_out(xs, ys)
    elif n == 4:
        record["leave_one_out"] = leave_one_out(xs, ys)
    else:
        record["reason"] = f"n={n}; coefficient without interval"
    return record


def inventory(cells: Mapping[str, Mapping[str, Cell]] | None = None) -> dict[str, Any]:
    table = CELLS if cells is None else cells
    counts = {"available": 0, "pending": 0, "unavailable": 0}
    arms: dict[str, dict[str, Any]] = {}
    for name, row in table.items():
        entry = {}
        for axis, item in row.items():
            entry[axis] = {"status": item.status, "has_value": item.value is not None, "note": item.note}
            counts[item.status] = counts.get(item.status, 0) + 1
        arms[name] = entry
    return {
        "campaign": CAMPAIGN,
        "is_scientific_measurement": False,
        "counts": counts,
        "arms": arms,
        "forbids": [
            "parameter-count causal claim",
            "model ranking",
            "biology or function",
            "merging conditional and unconditional",
            "homologue-context AUROC as generation",
        ],
    }


def associate(
    *,
    cells: Mapping[str, Mapping[str, Cell]] | None = None,
    pairs: Iterable[tuple[str, str, str]] | None = None,
    arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    table = CELLS if cells is None else cells
    table_arms = list(table if arms is None else arms)
    chosen = tuple(POOLED_PAIRS if pairs is None else pairs)
    records = [
        associate_pair(pair_id, x, y, table_arms, table=table) for pair_id, x, y in chosen
    ]
    lineages = {
        name: [
            associate_pair(f"L-{name}-{x}-{y}", x, y, list(members), table=table)
            for pair_id, x, y in chosen
            if pair_id.startswith("P-")
        ]
        for name, members in LINEAGES.items()
    }
    pending_pairs = [row["pair_id"] for row in records if row["pending"]]
    computed = [row["pair_id"] for row in records if row["computed"]]
    return {
        "campaign": CAMPAIGN,
        "is_scientific_measurement": True,
        "descriptive_not_causal": True,
        "no_parameter_count_causal_claim": True,
        "pooled": records,
        "lineages": lineages,
        "pending_pairs": pending_pairs,
        "computed_pairs": computed,
        "inventory": inventory(table),
    }


def refuse_missing(axis: str, arm: str) -> None:
    item = cell(arm, axis)
    if item.pending() or not item.resolved():
        raise ValueError(f"{arm}:{axis} is {item.status}; refusing to invent a value")
