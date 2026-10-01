#!/usr/bin/env python3
"""Apply a frozen grouping contract to the MegaScale WT backgrounds.

Reads the catalogue, the pinned parquet WT rows, the frozen contract and an
optional Pfam domain table, and writes an admitted group map plus a report. No
phenotype column is read: every merge source is sequence, name or annotation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from src.capability.core.family_grouping import Union, batch_align, encode, parent_background, reference_alignment

IDENTITY_FLOOR = 30.0
COVERAGE_FLOOR = 80.0
GAP_OPEN = 11.0
GAP_EXTEND = 1.0
VERIFY_PAIRS = 250
VERIFY_SEED = 20260923


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
    return sha.hexdigest()


def pfam_labels(paths, names) -> dict[str, list[str]]:
    """Accessions per background from HMMER domain tables filtered by --cut_ga."""

    labels: dict[str, set[str]] = {name: set() for name in names}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            accession, query = fields[1], fields[3]
            if query not in labels:
                raise ValueError(f"{path.name}: domain table names {query}, not a background")
            labels[query].add(accession)
    return {name: sorted(values) for name, values in labels.items()}


def verify_batch(sequences, codes, lengths, names) -> dict:
    """Contract check: the batched carry must equal an independent traceback."""

    rng = np.random.default_rng(VERIFY_SEED)
    drawn = set()
    while len(drawn) < VERIFY_PAIRS:
        left, right = (int(v) for v in rng.integers(0, len(names), 2))
        if left != right:
            drawn.add((min(left, right), max(left, right)))
    stems: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        stems.setdefault(parent_background(name), []).append(index)
    for members in stems.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                drawn.add((members[a], members[b]))
    pairs = sorted(drawn)
    statistics = batch_align(codes, lengths, np.array(pairs), gap_open=GAP_OPEN,
                             gap_extend=GAP_EXTEND)
    for index, (left, right) in enumerate(pairs):
        expected = reference_alignment(sequences[left], sequences[right],
                                       gap_open=GAP_OPEN, gap_extend=GAP_EXTEND)
        observed = (float(statistics.score[index]), int(statistics.columns[index]),
                    int(statistics.identical[index]), float(statistics.coverage_a[index]),
                    float(statistics.coverage_b[index]))
        if (abs(expected[0] - observed[0]) > 1e-6 or expected[1] != observed[1]
                or expected[2] != observed[2] or abs(expected[3] - observed[3]) > 1e-9
                or abs(expected[4] - observed[4]) > 1e-9):
            raise RuntimeError(
                f"batched alignment disagrees with the reference traceback on "
                f"{names[left]} vs {names[right]}: {observed} against {expected}"
            )
    return {"pairs_checked": len(pairs), "status": "passed"}


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--pfam-domtbl", type=Path, nargs="*", default=[])
    parser.add_argument("--pfam-asset", type=Path, default=None,
                        help="the pressed Pfam-A.hmm whose digest is recorded")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(path)

    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "megascale_family_grouping_contract_v1":
        raise ValueError("unexpected contract schema")

    catalogue = json.loads(args.catalogue.read_text())
    entries = {row["WT_name"]: row for row in catalogue}
    if len(entries) != len(catalogue):
        raise ValueError("duplicate catalogue backgrounds")

    frame = pd.concat([pd.read_parquet(path, columns=["mut_type", "WT_name", "WT_cluster", "aa_seq"])
                       for path in sorted(args.parquet)], ignore_index=True)
    wild = frame[frame.mut_type == "wt"]
    sequences_by_name = wild.groupby("WT_name").aa_seq.agg(lambda values: sorted(set(values)))
    clusters_by_name = frame.groupby("WT_name").WT_cluster.agg(
        lambda values: sorted({str(value) for value in values}))
    for name, row in entries.items():
        if name not in sequences_by_name.index:
            raise ValueError(f"{name} has no mut_type=='wt' row in the pinned files")
        if sequences_by_name[name] != [row["sequence"]]:
            raise ValueError(f"{name}: catalogue sequence differs from the pinned WT row")
        if len(clusters_by_name[name]) != 1:
            raise ValueError(f"{name} carries more than one WT_cluster value")
    outside = sorted(set(frame.WT_name) - set(entries))
    no_wild = sorted(set(frame.WT_name) - set(sequences_by_name.index))

    names = sorted(entries)
    sequences = [entries[name]["sequence"] for name in names]
    codes, lengths = encode(sequences)
    verification = verify_batch(sequences, codes, lengths, names)

    pairs = np.array([(a, b) for a in range(len(names)) for b in range(a + 1, len(names))])
    statistics = batch_align(codes, lengths, pairs, gap_open=GAP_OPEN, gap_extend=GAP_EXTEND)
    identity = statistics.percent_identity
    keep = statistics.edges(identity_floor=IDENTITY_FLOOR, coverage_floor=COVERAGE_FLOOR)

    labels = (pfam_labels(args.pfam_domtbl, names) if args.pfam_domtbl
              else {name: [] for name in names})

    union = Union(names)
    counts = {key: 0 for key in ("R1_exact_sequence", "R2_alignment", "R3_source_cluster",
                                 "R4_parent_background", "R5_pfam", "R6_design_stratum")}
    merges = dict(counts)

    def join(left: str, right: str, source: str) -> None:
        counts[source] += 1
        merges[source] += int(union.join(left, right, source))

    by_sequence: dict[str, list[str]] = {}
    by_cluster: dict[str, list[str]] = {}
    by_parent: dict[str, list[str]] = {}
    by_accession: dict[str, list[str]] = {}
    by_design_group: dict[str, list[str]] = {}
    for name in names:
        by_sequence.setdefault(entries[name]["sequence"], []).append(name)
        by_cluster.setdefault(clusters_by_name[name][0], []).append(name)
        by_parent.setdefault(parent_background(name), []).append(name)
        for accession in labels[name]:
            by_accession.setdefault(accession, []).append(name)
        if entries[name]["kind"] == "design":
            by_design_group.setdefault(entries[name]["group"], []).append(name)

    for source, table in (("R1_exact_sequence", by_sequence), ("R3_source_cluster", by_cluster),
                          ("R4_parent_background", by_parent), ("R5_pfam", by_accession),
                          ("R6_design_stratum", by_design_group)):
        for members in table.values():
            for other in members[1:]:
                join(members[0], other, source)
    scope = contract["union_sources"].get("R2_alignment_scope", "all_pairs")
    if scope not in ("all_pairs", "within_stratum"):
        raise ValueError(f"undeclared alignment scope {scope!r}")
    alignment_edges = []
    cross_stratum: dict[str, list[dict]] = {}
    for index in np.flatnonzero(keep):
        left, right = names[pairs[index, 0]], names[pairs[index, 1]]
        crossing = entries[left]["kind"] != entries[right]["kind"]
        record = {
            "left": left, "right": right,
            "percent_identity": round(float(identity[index]), 3),
            "coverage_left": round(float(statistics.coverage_a[index]), 3),
            "coverage_right": round(float(statistics.coverage_b[index]), 3),
            "score": float(statistics.score[index]),
            "cross_stratum": crossing,
            "merged": not (crossing and scope == "within_stratum"),
        }
        alignment_edges.append(record)
        if crossing:
            for near, far in ((left, right), (right, left)):
                if entries[near]["kind"] == "natural":
                    cross_stratum.setdefault(near, []).append(
                        {"design": far, "percent_identity": record["percent_identity"],
                         "series": entries[far]["group"]})
        if record["merged"]:
            join(left, right, "R2_alignment")

    components = union.components()
    kinds = {root: sorted({entries[name]["kind"] for name in members})
             for root, members in components.items()}
    stratum = {root: ("natural" if value == ["natural"] else
                      "design" if value == ["design"] else "mixed")
               for root, value in kinds.items()}
    ordered = sorted(components, key=lambda root: (stratum[root], root))
    prefix = {"natural": "nat", "design": "des", "mixed": "mix"}
    counter: dict[str, int] = {}
    group_of: dict[str, str] = {}
    membership = {}
    for root in ordered:
        kind = stratum[root]
        counter[kind] = counter.get(kind, 0) + 1
        label = f"{prefix[kind]}-{counter[kind]:03d}"
        membership[label] = components[root]
        for name in components[root]:
            group_of[name] = label

    sizes = {label: len(members) for label, members in membership.items()}
    natural_names = [name for name in names if entries[name]["kind"] == "natural"]
    groups_by_stratum = {
        kind: sorted(label for label in membership if label.startswith(prefix[kind]))
        for kind in prefix
    }
    provenance = {
        "contract": {"path": str(args.contract), "sha256": digest(args.contract),
                     "frozen_utc": contract["frozen_utc"]},
        "catalogue": {"path": str(args.catalogue), "sha256": digest(args.catalogue)},
        "parquet": [{"path": str(path), "sha256": digest(path)} for path in sorted(args.parquet)],
        "pfam": {
            "domain_tables": [{"path": str(path), "sha256": digest(path)}
                              for path in sorted(args.pfam_domtbl)],
            "asset": ({"path": str(args.pfam_asset), "sha256": digest(args.pfam_asset)}
                      if args.pfam_asset else None),
            "labelled_backgrounds": sum(1 for name in names if labels[name]),
            "unlabelled_backgrounds": sum(1 for name in names if not labels[name]),
            "distinct_accessions": len({a for name in names for a in labels[name]}),
        },
        "code_sha256": {
            str(path): digest(path) for path in
            (Path(__file__).resolve(),
             Path(__file__).resolve().parents[3] / "src/capability/core/family_grouping.py")
        },
        "alignment": {"identity_floor_percent": IDENTITY_FLOOR,
                      "coverage_floor_percent": COVERAGE_FLOOR,
                      "gap_open": GAP_OPEN, "gap_extend": GAP_EXTEND,
                      "pairs_scored": int(len(pairs)),
                      "edges_accepted": int(keep.sum()),
                      "scope": scope,
                      "cross_stratum_edges": sum(1 for e in alignment_edges if e["cross_stratum"]),
                      "cross_stratum_edges_merged": sum(
                          1 for e in alignment_edges if e["cross_stratum"] and e["merged"])},
        "natural_backgrounds_with_cross_stratum_similarity": {
            name: {"edges": len(rows),
                   "max_percent_identity": max(r["percent_identity"] for r in rows),
                   "design_series": sorted({r["series"] for r in rows})}
            for name, rows in sorted(cross_stratum.items())},
        "verification": verification,
        "edge_counts": counts,
        "effective_merges": merges,
        "final_group_count": len(membership),
        "groups_by_stratum": {kind: len(value) for kind, value in groups_by_stratum.items()},
        "backgrounds": {"total": len(names),
                        "natural": len(natural_names),
                        "design": len(names) - len(natural_names)},
        "parquet_names_outside_catalogue": outside,
        "parquet_names_without_wt_row": no_wild,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"schema": "megascale_family_groups_v1", "status": "admitted",
         "provenance": provenance, "assignments": group_of}, indent=1) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {"provenance": provenance, "membership": membership, "group_sizes": sizes,
         "pfam_labels": labels, "alignment_edges": alignment_edges}, indent=1) + "\n")
    print(json.dumps({k: provenance[k] for k in
                      ("final_group_count", "groups_by_stratum", "edge_counts",
                       "effective_merges", "alignment", "verification")}, indent=1))
    print("pfam", json.dumps(provenance["pfam"]["labelled_backgrounds"]),
          "labelled of", len(names))


if __name__ == "__main__":
    main()
