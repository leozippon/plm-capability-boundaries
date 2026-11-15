#!/usr/bin/env python3
"""How far does a 0.25 low-complexity limit depend on the statistic that implements it?

Two implementations in this repository answer to the name "low-complexity
fraction", and they are not versions of one another.
``sequence_reward.low_complexity_fraction`` (S1) is the share of residues covered
by 12-residue windows at or below 2.2 bits of Shannon entropy; the
reward-provenance constraint puts a limit of 0.25 on it.
``objective_tuned_analysis._low_complexity`` (D5) is the distinct-5-mer fraction,
where higher means more diverse; the explicit-statistics control keeps the top
quarter of a base pool by it, which is a retained quantile and not a limit.

This producer measures both on one set of real sequences and reports how much the
0.25 decision moves. D5 is put on S1's scale as S2 = 1 - D5, the share of 5-mer
positions that repeat an earlier 5-mer, because that is the sense D5's name
suggests; two other readings are reported beside it so that the conclusion does
not hang on that choice: the literal one (D5 < 0.25 "passes") and the project's
own one (top quarter by D5).

The main set is 7,916 canonical sequences, none synthetic: a seeded 400 per arm
of the eight objective-tuned arms' folded structures, and every record of the
unconditional-generation evaluation cohort (nine generating arms, ProtGPT2 and
ZymCTRL pools, and whole Swiss-Prot records matched to the generations on
length). Reported on their own: the reward-provenance base pools, the whole
objective-tuned folded table, and a seeded natural short-sequence supplement
(12-49 residues), because the main set has no natural sequence below 50.

Both statistics are the repository's own functions, imported and not
reimplemented. Nothing is generated, folded or scored, and no GPU is touched.
Flags at the limit use integer arithmetic, so a sequence at exactly 0.25 is not
moved by a float comparison.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions import sequence_reward as sr  # noqa: E402
from src.capability.forcing import backbone_cohort as bc  # noqa: E402
from src.capability.forcing import forcing_design as fd  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation.near_duplicates import RESIDUE_SHINGLE  # noqa: E402

SEED = 20261114
PER_ARM = 400
DRAWS = 2000
SHORT_BINS = ((12, 24), (25, 49))
SHORT_PER_BIN = 300
LENGTH_BINS = ((0, 49, "<50"), (50, 99, "50-99"), (100, 199, "100-199"),
               (200, 299, "200-299"), (300, 399, "300-399"), (400, 400, "400"),
               (401, 10**9, ">400"))
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")

FOLD_DIR = "results/external_baseline_analysis/length_adjusted_contrasts_20261110"
# Digests of the retained inputs the main set is built from; the first two are the
# values the registry records for the objective-tuned fold table and its cohort.
PINNED = {
    "fold_table": ("{}/eighteen_shard/checkpoint_structures_18of68.tsv".format(FOLD_DIR),
                   "2fd2ee4b91b67ef7bcef91241ccfddc61353b6f51e5b788270be8a2df8aabf06"),
    "fold_cohort": ("{}/input/fold_cohort.jsonl".format(FOLD_DIR),
                    "348c4352c14b88a70fe1178fc2552bb49e66bd583665eaabe0740f2526a72069"),
    "evaluation_cohort": ("results/R6/generation_evaluation_20261008/evaluation_cohort.jsonl",
                          "dd282e3c7fc6ddcf479574608c7755f31998eb69e440bbcecf2d7d7fb3d111f2"),
    "swissprot": ("data/swissprot/uniprot_sprot.fasta.gz",
                  "5ba5cb332fc7794ab1c02075a79c8b3d95b573f9b244a38bb53558172e1f9b7b"),
}
RL_POOLS = "results/extensions/rl_environment_provenance_20261114/pools"
BASE_ARMS = ("protgpt3-112m", "protgpt3-1.3b", "protgpt3-10b", "progen2-small-hugohrban")


# ------------------------------------------------------------------ measurement


def measure(sequence: str) -> dict:
    """Both statistics and their integer counts for one sequence."""

    length = len(sequence)
    if length == 0:
        raise SystemExit("an empty sequence has neither statistic")
    s1 = sr.low_complexity_fraction(sequence)
    d5 = ota._low_complexity(sequence)  # the statistic the rule `low_complexity_residues` ranks by
    positions = length - RESIDUE_SHINGLE + 1
    labelled = round(s1 * length)
    if abs(labelled / length - s1) > 1e-12:
        raise SystemExit("low_complexity_fraction is not a count over the sequence length")
    distinct = 0
    if positions >= 1:
        distinct = round(d5 * positions)
        if abs(distinct / positions - d5) > 1e-12:
            raise SystemExit("distinct_shingle_fraction is not a count over the 5-mer positions")
    counts: dict[str, int] = {}
    for residue in sequence:
        counts[residue] = counts.get(residue, 0) + 1
    top_residue, top_count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return {
        "length": length, "labelled_residues": labelled, "s1": s1,
        "kmer_positions": max(positions, 0), "distinct_5mers": distinct, "d5": d5,
        "composition_entropy_bits": sr.composition_entropy_bits(sequence),
        "longest_run": sr.longest_single_residue_run(sequence),
        "top_residue": top_residue, "top_residue_share": top_count / length,
    }


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """The 0.25 flags, by integer arithmetic, and the other readings of the second 0.25."""

    if sr.LOW_COMPLEXITY_LIMIT != 0.25:
        raise SystemExit("the integer flags assume the declared limit is exactly 1/4")
    frame = frame.copy()
    positions = frame["kmer_positions"]
    repeats = positions - frame["distinct_5mers"]
    frame["s2"] = np.where(positions >= 1, repeats / positions.clip(lower=1), 1.0)
    frame["flag_s1"] = 4 * frame["labelled_residues"] >= frame["length"]
    # Below five residues D5 is 0.0 by the function's own convention, so S2 is 1.0.
    frame["flag_s2"] = np.where(positions >= 1, 4 * repeats >= positions, True)
    frame["s1_exactly_limit"] = 4 * frame["labelled_residues"] == frame["length"]
    frame["s2_exactly_limit"] = np.where(positions >= 1, 4 * repeats == positions, False)
    frame["literal_pass"] = frame["d5"] < sr.LOW_COMPLEXITY_LIMIT
    frame["forcing_flag"] = frame["d5"] < fd.MIN_DISTINCT_SHINGLE_FRACTION
    frame["no_repeated_5mer"] = repeats == 0
    return frame


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    agree = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float((agree - expected) / (1 - expected)) if expected < 1 else float("nan")


def confusion(frame: pd.DataFrame, a: str = "flag_s1", b: str = "flag_s2") -> dict:
    x, y = frame[a].to_numpy(bool), frame[b].to_numpy(bool)
    n = int(x.size)
    both, only_a = int((x & y).sum()), int((x & ~y).sum())
    only_b, neither = int((~x & y).sum()), int((~x & ~y).sum())
    return {"n": n, "both": both, "s1_only": only_a, "s2_only": only_b, "neither": neither,
            "agreement": (both + neither) / n if n else None,
            "kappa": cohen_kappa(x, y) if n else None,
            "prevalence_s1": float(x.mean()) if n else None,
            "prevalence_s2": float(y.mean()) if n else None}


def rank_correlation(frame: pd.DataFrame) -> dict:
    if len(frame) < 5 or frame["s1"].nunique() < 2 or frame["d5"].nunique() < 2:
        return {"spearman": None, "kendall_tau_b": None}
    return {"spearman": float(stats.spearmanr(frame["s1"], frame["d5"]).statistic),
            "kendall_tau_b": float(stats.kendalltau(frame["s1"], frame["d5"]).statistic)}


def describe_set(frame: pd.DataFrame) -> dict:
    out = {**confusion(frame), **rank_correlation(frame)}
    n = len(frame)
    # The ranking the two statistics share where neither flags: what is left once the scale is set aside.
    unflagged = frame[~frame["flag_s1"] & ~frame["flag_s2"]]
    out["n_unflagged"] = int(len(unflagged))
    out["spearman_among_unflagged"] = rank_correlation(unflagged)["spearman"]
    out["literal_pass"] = int(frame["literal_pass"].sum())
    out["sequence_reward_pass"] = int((~frame["flag_s1"]).sum())
    out["forcing_screen_flagged"] = int(frame["forcing_flag"].sum())
    out["no_repeated_5mer"] = int(frame["no_repeated_5mer"].sum())
    out["s1_exactly_limit"] = int(frame["s1_exactly_limit"].sum())
    out["s2_exactly_limit"] = int(frame["s2_exactly_limit"].sum())
    out["length"] = {"min": int(frame["length"].min()), "median": float(frame["length"].median()),
                     "max": int(frame["length"].max())} if n else None
    return out


def matched_prevalence(frame: pd.DataFrame) -> dict:
    """Raise S1's cut until its prevalence equals the S2 >= 0.25 prevalence, then compare."""

    target = float(frame["flag_s2"].mean())
    best = None
    for cut in np.unique(frame["s1"].to_numpy()):
        gap = abs(float((frame["s1"] >= cut).mean()) - target)
        if best is None or gap < best[0]:
            best = (gap, float(cut))
    cut = best[1]
    matched = frame.assign(flag_s1_matched=frame["s1"] >= cut)
    return {"s1_cut": cut, "prevalence_target": target,
            "prevalence_s1_at_cut": float(matched["flag_s1_matched"].mean()),
            **confusion(matched, "flag_s1_matched", "flag_s2")}


def bootstrap(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    s1, d5 = frame["s1"].to_numpy(), frame["d5"].to_numpy()
    x, y = frame["flag_s1"].to_numpy(bool), frame["flag_s2"].to_numpy(bool)
    draws: dict[str, list[float]] = {k: [] for k in
                                     ("spearman", "agreement", "s1_only", "s2_only", "kappa")}
    for _ in range(DRAWS):
        index = rng.integers(0, len(frame), len(frame))
        draws["spearman"].append(float(stats.spearmanr(s1[index], d5[index]).statistic))
        a, b = x[index], y[index]
        draws["agreement"].append(float((a == b).mean()))
        draws["s1_only"].append(float((a & ~b).mean()))
        draws["s2_only"].append(float((~a & b).mean()))
        draws["kappa"].append(cohen_kappa(a, b))
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for key, v in draws.items()}


def length_bin(length: int) -> str:
    for low, high, name in LENGTH_BINS:
        if low <= length <= high:
            return name
    raise ValueError(length)


# --------------------------------------------------------------------- the sets


def read_fold_table(path: Path) -> list[tuple[str, str, float]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        header = None
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            record = dict(zip(header, fields))
            rows.append((record["arm"], record["sequence_sha256"],
                         float(record["mean_ca_plddt_0_100"])))
    return rows


def objective_tuned_records(fold_table: Path, fold_cohort: Path) -> list[dict]:
    sequences = {}
    for line in fold_cohort.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            sequences[record["id"]] = record["sequence"]
    out = []
    for arm, digest, _ in read_fold_table(fold_table):
        sequence = sequences[digest]
        if hashlib.sha256(sequence.encode()).hexdigest() != digest:
            raise SystemExit(f"{digest}: the cohort sequence does not hash to its own key")
        out.append({"source": "OT_folded", "arm": arm, "sequence_sha256": digest,
                    "accession": "", "sequence": sequence})
    sample_ids = set()
    rng = random.Random(SEED)
    by_arm: dict[str, list[dict]] = {}
    for record in out:
        by_arm.setdefault(record["arm"], []).append(record)
    for arm in sorted(by_arm):
        pool = sorted(by_arm[arm], key=lambda r: r["sequence_sha256"])
        sample_ids.update(r["sequence_sha256"] for r in rng.sample(pool, PER_ARM))
    for record in out:
        record["in_ot_sample"] = record["sequence_sha256"] in sample_ids
    return out


def evaluation_records(path: Path) -> list[dict]:
    names = {"generated": "UG_generated", "natural": "NAT_swissprot_matched", "pool": "UG_pool"}
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if len(record["sequence"]) != record["length"]:
            raise SystemExit(f"{record['id']}: length does not match the sequence")
        out.append({"source": names[record["role"]], "arm": record["arm"] or "natural",
                    "sequence_sha256": record["sequence_sha256"], "accession": "",
                    "sequence": record["sequence"], "in_ot_sample": False})
    return out


def reward_pool_records(pools: Path) -> list[dict]:
    out = []
    for attempts in sorted(pools.glob("rlp_base_stream*/attempts.jsonl")):
        stream = attempts.parent.name.removeprefix("rlp_base_stream")
        for line in attempts.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["sequence"]:
                out.append({"source": "RL_base_pool", "arm": f"protgpt3-112m_s{stream}",
                            "sequence_sha256": record["sequence_sha256"], "accession": "",
                            "sequence": record["sequence"], "in_ot_sample": False})
    return out


def swissprot_short_records(path: Path) -> list[dict]:
    entries: list[tuple[str, str]] = []
    accession, chunks = "", []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if chunks:
                    entries.append((accession, "".join(chunks)))
                accession, chunks = line.split("|")[1], []
            else:
                chunks.append(line.strip())
    if chunks:
        entries.append((accession, "".join(chunks)))
    canonical = [(a, s) for a, s in entries if set(s) <= CANONICAL]
    rng = random.Random(SEED)
    out = []
    for low, high in SHORT_BINS:
        pool = sorted((s, a) for a, s in canonical if low <= len(s) <= high)
        for sequence, accession in rng.sample(pool, min(SHORT_PER_BIN, len(pool))):
            out.append({"source": "NAT_short_supplement", "arm": f"{low}-{high}",
                        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                        "accession": accession, "sequence": sequence, "in_ot_sample": False})
    return out


# ------------------------------------------------------------------ the analysis


def top_quarter_reading(frame: pd.DataFrame) -> dict:
    """The project's own use of D5: keep the top `TOP_FRACTION` of a base pool by it.

    Descriptive only. The real analysis ranks within (arm, stream) and breaks ties
    on the attempt key; here ties break on the sequence digest, which is content-free
    in the same way. The retention rule itself is audited elsewhere.
    """

    out = {"tie_break": "sequence digest, a content-free stand-in for the attempt key",
           "top_fraction": ota.TOP_FRACTION, "arms": {}}
    for arm in BASE_ARMS:
        pool = frame[(frame["source"] == "OT_folded") & (frame["arm"] == arm)]
        pool = pool.sort_values(["d5", "sequence_sha256"], ascending=[False, True])
        kept = pool.iloc[: int(len(pool) * ota.TOP_FRACTION)]
        out["arms"][arm] = {
            "n_pool": int(len(pool)), "n_retained": int(len(kept)),
            "no_repeated_5mer_share": float(pool["no_repeated_5mer"].mean()),
            "tie_group_exceeds_retained": bool(pool["no_repeated_5mer"].sum() >= len(kept)),
            "minimum_retained_d5": float(kept["d5"].min()),
            "retained_share_s1_at_or_above_limit": float(kept["flag_s1"].mean()),
            "pool_share_s1_at_or_above_limit": float(pool["flag_s1"].mean())}
    return out


def group_profiles(main: pd.DataFrame) -> tuple[dict, list[dict]]:
    main = main.assign(group=np.select(
        [main.flag_s1 & main.flag_s2, main.flag_s1 & ~main.flag_s2, ~main.flag_s1 & main.flag_s2],
        ["both", "s1_only", "s2_only"], "neither"),
        natural=main["source"] == "NAT_swissprot_matched")
    profiles = {}
    for (group, natural), part in main.groupby(["group", "natural"]):
        profiles[f"{group}|{'natural' if natural else 'generated'}"] = {
            "n": int(len(part)), "median_s1": float(part["s1"].median()),
            "s1_quartiles": [float(part["s1"].quantile(0.25)), float(part["s1"].quantile(0.75))],
            "median_s2": float(part["s2"].median()),
            "median_composition_entropy_bits": float(part["composition_entropy_bits"].median()),
            "median_longest_run": float(part["longest_run"].median()),
            "median_top_residue_share": float(part["top_residue_share"].median()),
            "median_length": float(part["length"].median()),
            "share_at_400": float((part["length"] == 400).mean()),
            "top_residues": part["top_residue"].value_counts().head(6).to_dict()}
    rng = random.Random(SEED)
    examples = []
    for key in sorted(profiles):
        group, kind = key.split("|")
        pool = main[(main.group == group) & (main.natural == (kind == "natural"))]
        pool = pool.sort_values("sequence_sha256")
        for _, row in pool.iloc[sorted(rng.sample(range(len(pool)), min(3, len(pool))))].iterrows():
            examples.append({"group": key, "source": row["source"], "arm": row["arm"],
                             "sequence_sha256": row["sequence_sha256"], "length": int(row["length"]),
                             "s1": float(row["s1"]), "s2": float(row["s2"])})
    return profiles, examples


def by_length(main: pd.DataFrame) -> list[dict]:
    rows = []
    main = main.assign(group=np.where(main["source"] == "NAT_swissprot_matched",
                                      "natural", "generated"),
                       bin=main["length"].map(length_bin))
    for (group, name), part in main.groupby(["group", "bin"]):
        rows.append({"group": group, "length_bin": name, **confusion(part),
                     **rank_correlation(part)})
    order = {name: i for i, (_, _, name) in enumerate(LENGTH_BINS)}
    return sorted(rows, key=lambda r: (r["group"], order[r["length_bin"]]))


def write_table(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, float_format="%.6g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path,
                        default=Path("results/extensions/low_complexity_mask_comparison_20261114"))
    args = parser.parse_args()
    repo = args.repo

    inputs = {}
    for name, (relative, expected) in PINNED.items():
        digest = sha256_file(repo / relative)
        if digest != expected:
            raise SystemExit(f"{relative}: sha256 {digest} is not the retained {expected}")
        inputs[name] = {"path": relative, "sha256": digest}

    records = objective_tuned_records(repo / PINNED["fold_table"][0], repo / PINNED["fold_cohort"][0])
    records += evaluation_records(repo / PINNED["evaluation_cohort"][0])
    pools = sorted((repo / RL_POOLS).glob("rlp_base_stream*/attempts.jsonl"))
    records += reward_pool_records(repo / RL_POOLS)
    inputs["reward_provenance_pools"] = [
        {"path": str(p.relative_to(repo)), "sha256": sha256_file(p)} for p in pools]
    records += swissprot_short_records(repo / PINNED["swissprot"][0])

    rows = []
    for record in records:
        if not set(record["sequence"]) <= CANONICAL:
            raise SystemExit(f"{record['sequence_sha256']}: a non-canonical residue")
        rows.append({k: record[k] for k in ("source", "arm", "sequence_sha256", "accession",
                                            "in_ot_sample")} | measure(record["sequence"]))
    frame = derive(pd.DataFrame(rows))
    frame["in_main_set"] = (frame["source"].isin(["UG_generated", "UG_pool", "NAT_swissprot_matched"])
                            | ((frame["source"] == "OT_folded") & frame["in_ot_sample"]))
    main_set = frame[frame["in_main_set"]]

    sets = {
        "main_set": main_set,
        "OT_sample": frame[(frame["source"] == "OT_folded") & frame["in_ot_sample"]],
        "UG_generated": frame[frame["source"] == "UG_generated"],
        "UG_pool": frame[frame["source"] == "UG_pool"],
        "NAT_swissprot_matched": frame[frame["source"] == "NAT_swissprot_matched"],
        "RL_base_pool": frame[frame["source"] == "RL_base_pool"],
        "NAT_short_supplement": frame[frame["source"] == "NAT_short_supplement"],
        "OT_folded_all": frame[frame["source"] == "OT_folded"],
    }
    per_arm = []
    for (source, arm), part in main_set.groupby(["source", "arm"]):
        if source != "NAT_swissprot_matched":
            per_arm.append({"source": source, "arm": arm, "median_length": float(part["length"].median()),
                            **confusion(part), **rank_correlation(part)})
    short = [{"length_bin": arm, **confusion(part), **rank_correlation(part)}
             for arm, part in sets["NAT_short_supplement"].groupby("arm")]
    profiles, examples = group_profiles(main_set)

    result = {
        "seed": SEED, "draws_per_arm": PER_ARM, "bootstrap_draws": DRAWS,
        "limit": sr.LOW_COMPLEXITY_LIMIT, "window": sr.LOW_COMPLEXITY_WINDOW,
        "entropy_bits": sr.LOW_COMPLEXITY_ENTROPY_BITS, "top_fraction": ota.TOP_FRACTION,
        "forcing_screen_d5_floor": fd.MIN_DISTINCT_SHINGLE_FRACTION,
        "kmer": RESIDUE_SHINGLE,
        "definitions": {
            "s1": "sequence_reward.low_complexity_fraction",
            "d5": "objective_tuned_analysis._low_complexity (distinct_shingle_fraction)",
            "s2": "1 - d5, the share of 5-mer positions that repeat an earlier 5-mer",
            "flag_s1": "s1 >= 0.25, integer arithmetic",
            "flag_s2": "s2 >= 0.25, integer arithmetic (1.0 below five residues, as D5 is 0.0 there)"},
        "implementations": {
            name: {"file": str(Path(inspect.getsourcefile(module)).relative_to(repo)),
                   "file_sha256": sha256_file(Path(inspect.getsourcefile(module))),
                   "function_source_sha256": hashlib.sha256(
                       inspect.getsource(function).encode()).hexdigest()}
            for name, module, function in (
                ("s1", sr, sr.low_complexity_fraction),
                ("d5", ota, ota._low_complexity),
                ("d5_underlying", bc, bc.distinct_shingle_fraction))},
        "inputs": inputs,
        "set_sizes": {name: int(len(part)) for name, part in sets.items()},
        "sets": {name: describe_set(part) for name, part in sets.items()},
        "main_set_bootstrap_95": bootstrap(main_set, np.random.default_rng(SEED)),
        "matched_prevalence": {name: matched_prevalence(sets[name])
                               for name in ("main_set", "OT_sample", "UG_generated")},
        "by_length": by_length(main_set),
        "by_arm": per_arm,
        "natural_short_by_length": short,
        "discordant_group_profiles": profiles,
        "top_quarter_by_d5": top_quarter_reading(frame),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)
    write_json(args.out / "mask_comparison.json", result)
    write_table(args.out / "tables/confusion_by_set.tsv",
                [{"set": name, **{k: v for k, v in entry.items() if k != "length"}}
                 for name, entry in result["sets"].items()])
    write_table(args.out / "tables/confusion_by_arm.tsv", per_arm)
    write_table(args.out / "tables/confusion_by_length.tsv", result["by_length"])
    write_table(args.out / "tables/natural_short_by_length.tsv", short)
    write_table(args.out / "tables/discordant_examples.tsv", examples)
    frame.drop(columns=["flag_s1", "flag_s2"]).assign(
        flag_s1=frame["flag_s1"].astype(int), flag_s2=frame["flag_s2"].astype(int)
    ).to_csv(args.out / "per_sequence.tsv", sep="\t", index=False, float_format="%.8g")
    main_row = result["sets"]["main_set"]
    print(json.dumps({"n": main_row["n"], "spearman": main_row["spearman"],
                      "s1_only": main_row["s1_only"], "s2_only": main_row["s2_only"],
                      "both": main_row["both"], "neither": main_row["neither"],
                      "agreement": main_row["agreement"], "kappa": main_row["kappa"]}, indent=1))


if __name__ == "__main__":
    main()
