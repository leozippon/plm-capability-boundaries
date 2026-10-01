"""Shared sequence groups utilities required by capability measurements."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np

def kmer_set(sequence: str, k: int) -> set[str]:
    if k < 1:
        raise ValueError("k must be positive")
    return {sequence[i : i + k] for i in range(max(0, len(sequence) - k + 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def homology_clusters(
    accessions: Sequence[str],
    sequences: Sequence[str],
    *,
    pfam_by_accession: Mapping[str, set[str]],
    kmer: int = 3,
    kmer_jaccard_threshold: float = 0.10,
    pfam_jaccard_threshold: float = 0.50,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Single-linkage homology clusters over domain architecture and k-mer overlap.

    The Pfam relation is Jaccard overlap of the two proteins' family sets, not
    the presence of any shared family. Sharing one hub domain - an ABC
    transporter cassette, a kinase fold - does not make two multi-domain
    proteins homologous, but under single linkage it chains the entire cohort
    into one cluster and leaves no split at all. Requiring that most of the
    architecture agrees keeps the relation close to actual homology, and
    single-domain proteins that share their only family still merge at Jaccard
    1.0. The k-mer criterion covers proteins with no Pfam annotation, which
    would otherwise be treated as unrelated to everything.
    """

    if len(accessions) != len(sequences) or not accessions:
        raise ValueError("accessions and sequences must be non-empty and aligned")
    if not 0.0 < kmer_jaccard_threshold <= 1.0:
        raise ValueError("kmer_jaccard_threshold must lie in (0, 1]")
    if not 0.0 < pfam_jaccard_threshold <= 1.0:
        raise ValueError("pfam_jaccard_threshold must lie in (0, 1]")
    n = len(accessions)
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    families = [set(pfam_by_accession.get(accession, set())) for accession in accessions]
    kmers = [kmer_set(sequence, kmer) for sequence in sequences]
    pfam_links = 0
    kmer_links = 0
    for left in range(n):
        for right in range(left + 1, n):
            if find(left) == find(right):
                continue
            if (
                families[left]
                and families[right]
                and _jaccard(families[left], families[right]) >= pfam_jaccard_threshold
            ):
                union(left, right)
                pfam_links += 1
            elif _jaccard(kmers[left], kmers[right]) >= kmer_jaccard_threshold:
                union(left, right)
                kmer_links += 1

    labels = np.asarray([find(index) for index in range(n)], dtype=np.int64)
    _, cluster_ids, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    unannotated = sum(1 for entry in families if not entry)
    return cluster_ids.astype(np.int64), {
        "n_proteins": n,
        "n_clusters": int(sizes.size),
        "largest_cluster_size": int(sizes.max()),
        "n_singleton_clusters": int((sizes == 1).sum()),
        "n_proteins_without_pfam": unannotated,
        "pfam_merges": pfam_links,
        "kmer_merges": kmer_links,
        "kmer_k": int(kmer),
        "kmer_jaccard_threshold": float(kmer_jaccard_threshold),
        "pfam_jaccard_threshold": float(pfam_jaccard_threshold),
    }


def homology_disjoint_split(
    cluster_ids: np.ndarray, *, train_fraction: float, seed: int, min_side: int = 2
) -> np.ndarray:
    """Boolean training mask that never splits a homology cluster."""

    if cluster_ids.ndim != 1 or cluster_ids.size < 2:
        raise ValueError("cluster_ids must be a one-dimensional array of at least two items")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    if min_side < 1:
        raise ValueError("min_side must be positive")
    generator = np.random.default_rng(seed)
    unique = np.unique(cluster_ids)
    if unique.size < 2:
        raise RuntimeError(
            "every protein falls in one homology cluster; a homology-disjoint "
            "split does not exist for this selection"
        )
    target = train_fraction * cluster_ids.size
    train_clusters: set[int] = set()
    assigned = 0
    for cluster in generator.permutation(unique):
        if assigned >= target:
            break
        train_clusters.add(int(cluster))
        assigned += int((cluster_ids == cluster).sum())
    mask = np.isin(cluster_ids, list(train_clusters))
    if int(mask.sum()) < min_side or int((~mask).sum()) < min_side:
        raise RuntimeError(
            f"homology-disjoint split gave {int(mask.sum())} train and "
            f"{int((~mask).sum())} test proteins with {unique.size} clusters over "
            f"{cluster_ids.size} proteins; the selection is too homology-collapsed "
            f"for a {min_side}-protein minimum on both sides"
        )
    return mask
