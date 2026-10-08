#!/usr/bin/env python3
"""Fit the small-domain stability head and validate it. The validation is the output.

Three instruments are evaluated side by side on identical rows:

``baseline``
    ridge on twenty residue fractions, length and log length. The cheap feature
    any apparent skill has to beat, because residue composition is free.
``plm``
    ridge on the baseline features **plus** frozen ESM2 embeddings. Baseline plus
    block, not block alone, so the reported increment is what the language model
    adds to composition rather than a contest between two arbitrary designs.
``prime``
    a third-party thermostability checkpoint's scalar head, fitted by other
    people on other data, scored elsewhere and read here. Ordinal only.

The penalty is chosen on the release's own validation folds and the test fold is
read exactly once, at the end. The decision that matters is not the test fold at
all but the cross-dataset check: an absolute free energy measured over a
different sequence population, resampled by wild-type cluster. Both declared gate
conditions are evaluated there, and the gate verdict written here is what the
application stage obeys.

CPU only -- everything expensive already happened in the embedding stage. It
accepts ``--device`` because the campaign queue injects it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import domain_stability as ds  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "domain_stability_fit.json"
SCHEMA_VERSION = "d1_domain_stability_fit_v1"

#: The ridge penalties searched, on the standardised design. Chosen on the
#: release's own validation folds, never on the test fold.
#:
#: The grid has to be wide enough at *both* ends for the baseline as well as the
#: language-model design, because the gate asks whether the embeddings beat the
#: cheap features. A twenty-two-feature baseline wants far less penalty than a
#: thirteen-hundred-feature one, and a grid whose floor binds would under-fit the
#: baseline and flatter the instrument being tested. An endpoint selection is
#: still flagged in the artefact.
ALPHA_GRID: tuple[float, ...] = (
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
)

#: The validation folds the penalty is chosen on.
TUNING_SPLITS: tuple[str, ...] = ("validation", "validation_online")


def load_tables(rows_path: Path, embed_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = json.loads((embed_dir / "embedding_index.json").read_text(encoding="utf-8"))
    embeddings = np.load(embed_dir / "embeddings.npy", mmap_mode="r")
    if embeddings.shape[0] != len(index["ids"]):
        raise SystemExit("the embedding array and its index disagree on row count")
    position = {identifier: i for i, identifier in enumerate(index["ids"])}
    missing = [row["id"] for row in rows if row["id"] not in position]
    if missing:
        raise SystemExit(
            f"{len(missing)} table rows carry no embedding (first {missing[:3]}); the "
            "embedding stage did not cover this table"
        )
    return {"rows": rows, "position": position, "embeddings": embeddings}


def feature_block(
    rows: list[dict[str, Any]],
    position: dict[str, int],
    embeddings: Any,
    *,
    with_embeddings: bool,
) -> np.ndarray:
    cheap = np.stack([ds.composition_features(str(row["sequence"])) for row in rows])
    if not with_embeddings:
        return cheap
    indices = [position[row["id"]] for row in rows]
    block = np.asarray(embeddings[indices], dtype=np.float64)
    return np.hstack([cheap, block])


def accumulate(
    rows: list[dict[str, Any]],
    position: dict[str, int],
    embeddings: Any,
    *,
    with_embeddings: bool,
    chunk: int,
) -> ds.RidgeGram:
    width = len(ds.BASELINE_FEATURES) + (embeddings.shape[1] if with_embeddings else 0)
    gram = ds.RidgeGram(width)
    # Accumulate in embedding-file order. The ridge normal equations are sums and
    # so are order-invariant, while the training fold is interleaved with the
    # other folds in the table, which would otherwise turn a million sequential
    # reads of a memory-mapped array on shared storage into a million random ones.
    ordered = sorted(rows, key=lambda row: position[row["id"]])
    for start in range(0, len(ordered), chunk):
        block = ordered[start : start + chunk]
        features = feature_block(block, position, embeddings, with_embeddings=with_embeddings)
        targets = np.asarray([row["measured_delta_g"] for row in block], dtype=np.float64)
        gram.add(features, targets)
    gram.finalise()
    return gram


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    loaded = load_tables(args.rows, args.embeddings)
    rows, position, embeddings = loaded["rows"], loaded["position"], loaded["embeddings"]
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row["split"]), []).append(row)

    train = by_split.get("train", [])
    if not train:
        raise SystemExit("the table carries no training fold")
    band = ds.licensed_band(int(row["length"]) for row in train)
    tune = [row for name in TUNING_SPLITS for row in by_split.get(name, [])]
    test = by_split.get("test", [])
    transfer = [row for row in by_split.get("transfer", []) if band[0] <= int(row["length"]) <= band[1]]
    transfer_all = by_split.get("transfer", [])
    if not tune or not test or not transfer:
        raise SystemExit(
            "the fit needs a tuning fold, a test fold and a cross-dataset fold inside "
            f"the licensed band {band}; got {len(tune)}, {len(test)}, {len(transfer)}"
        )

    fits: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, with_embeddings in (("baseline", False), ("plm", True)):
        gram = accumulate(
            train, position, embeddings, with_embeddings=with_embeddings, chunk=args.chunk
        )
        tune_features = feature_block(tune, position, embeddings, with_embeddings=with_embeddings)
        tune_truth = np.asarray([row["measured_delta_g"] for row in tune], dtype=np.float64)
        search = []
        for alpha in ALPHA_GRID:
            model = gram.solve(alpha)
            predicted = ds.predict_with(model, tune_features)
            search.append(
                {
                    "alpha": float(alpha),
                    "tuning_rmse": float(np.sqrt(np.mean((predicted - tune_truth) ** 2))),
                    "tuning_spearman": ds.regression_report(
                        tune_truth, predicted, unit=ds.QUANTITIES["measured_delta_g"]["unit"]
                    )["spearman"],
                }
            )
        best = min(search, key=lambda item: item["tuning_rmse"])
        if best["alpha"] in (ALPHA_GRID[0], ALPHA_GRID[-1]):
            # Not fatal, but a penalty chosen at an endpoint means the grid did
            # not bracket the optimum, which a reader must be told.
            best = dict(best, alpha_at_grid_edge=True)
        model = gram.solve(best["alpha"])
        fits[name] = {
            "design": (
                "composition and length"
                if not with_embeddings
                else "composition and length plus frozen ESM2 embeddings"
            ),
            "n_features": int(gram.n_features),
            "n_constant_features": int(gram.n_constant_features),
            "n_train_rows": int(gram.n_rows),
            "alpha_grid": list(ALPHA_GRID),
            "alpha_search": search,
            "alpha_selected": float(best["alpha"]),
            "alpha_at_grid_edge": bool(best.get("alpha_at_grid_edge", False)),
            "alpha_chosen_on": list(TUNING_SPLITS),
        }
        predictions[name] = {
            "test": ds.predict_with(
                model, feature_block(test, position, embeddings, with_embeddings=with_embeddings)
            ),
            "transfer": ds.predict_with(
                model,
                feature_block(transfer, position, embeddings, with_embeddings=with_embeddings),
            ),
            "transfer_all": ds.predict_with(
                model,
                feature_block(transfer_all, position, embeddings, with_embeddings=with_embeddings),
            ),
        }
        np.savez(
            args.out / f"model_{name}.npz",
            coefficients=model["coefficients"],
            intercept=np.asarray([model["intercept"]]),
            mean=model["mean"],
            scale=model["scale"],
        )

    prime: dict[str, np.ndarray] | None = None
    prime_record: dict[str, Any] | None = None
    if args.prime is not None:
        prime_record = json.loads((args.prime / "prime_stability_scores.json").read_text())
        scores = {
            str(json.loads(line)["id"]): float(json.loads(line)["prime_value_head"])
            for line in (args.prime / "prime_stability_scores.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        absent = [row["id"] for row in test + transfer if row["id"] not in scores]
        if absent:
            raise SystemExit(
                f"{len(absent)} evaluation rows carry no third-party score (first "
                f"{absent[:3]}); a partial second opinion is not read"
            )
        prime = {
            "test": np.asarray([scores[row["id"]] for row in test], dtype=np.float64),
            "transfer": np.asarray([scores[row["id"]] for row in transfer], dtype=np.float64),
        }

    def truth(block: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([row["measured_delta_g"] for row in block], dtype=np.float64)

    dg_unit = ds.QUANTITIES["measured_delta_g"]["unit"]
    prime_unit = ds.QUANTITIES["prime_value_head"]["unit"]
    held_out: dict[str, Any] = {}
    for name in ("baseline", "plm"):
        held_out[name] = {
            "test_fold": ds.regression_report(truth(test), predictions[name]["test"], unit=dg_unit),
            "cross_dataset_in_band": ds.regression_report(
                truth(transfer), predictions[name]["transfer"], unit=dg_unit
            ),
            "cross_dataset_full_band_extrapolation": ds.regression_report(
                truth(transfer_all), predictions[name]["transfer_all"], unit=dg_unit
            ),
        }
    if prime is not None:
        held_out["prime"] = {
            "test_fold": ds.regression_report(truth(test), prime["test"], unit=prime_unit),
            "cross_dataset_in_band": ds.regression_report(
                truth(transfer), prime["transfer"], unit=prime_unit
            ),
            "cross_dataset_full_band_extrapolation": None,
        }

    gates: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for name in ("plm",) + (("prime",) if prime is not None else ()):
        instrument = predictions["plm"]["transfer"] if name == "plm" else prime["transfer"]
        contrast = ds.transfer_contrast(
            truth(transfer),
            instrument,
            predictions["baseline"]["transfer"],
            [row["group"] for row in transfer],
            seed=args.seed,
            n_bootstrap=args.draws,
        )
        contrasts[name] = contrast
        gates[name] = ds.evaluate_gate(contrast)

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "licensed_band": list(band),
        "licensed_band_source": (
            "the minimum and maximum sequence length of the release's own training "
            "fold. The predictor is refused outside it"
        ),
        "support": {
            "n_train": len(train),
            "n_tuning": len(tune),
            "n_test": len(test),
            "n_cross_dataset_in_band": len(transfer),
            "n_cross_dataset_all_lengths": len(transfer_all),
            "n_cross_dataset_clusters": len({row["group"] for row in transfer}),
        },
        "fits": fits,
        "held_out": held_out,
        "cross_dataset_contrasts": contrasts,
        "gates": gates,
        "gate_conditions": dict(ds.GATE_CONDITIONS),
        "quantities": dict(ds.QUANTITIES),
        "prime": prime_record
        and {
            key: prime_record[key]
            for key in ("model", "n_scored", "provenance_warning", "third_party_code", "score_summary")
        },
        "sources": {
            "rows": str(args.rows),
            "rows_sha256": sha256_file(args.rows),
            "embeddings": str(args.embeddings),
            "prime": None if args.prime is None else str(args.prime),
        },
        "n_bootstrap": int(args.draws),
        "seed": int(args.seed),
        "reading_guide": [
            "the test fold measures fit on the training assay's own sequence "
            "population. It is not the decision",
            "the cross-dataset check measures transfer to a different sequence "
            "population measured by the same assay technology. It is the decision, and "
            "it is not evidence of transfer across measurement technology",
            "an instrument whose gate did not pass must not be applied to a generated "
            "sequence, whatever its test-fold numbers look like",
        ],
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True, help="the embedding stage's out dir")
    parser.add_argument("--prime", type=Path, default=None, help="the third-party scoring out dir")
    parser.add_argument("--chunk", type=int, default=20000)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    if args.draws < 1000:
        parser.error("--draws below 1000 gives percentile intervals this package will not publish")
    record = run(args)
    print(
        json.dumps(
            {
                "status": record["status"],
                "licensed_band": record["licensed_band"],
                "gates": {name: gate["passed"] for name, gate in record["gates"].items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
