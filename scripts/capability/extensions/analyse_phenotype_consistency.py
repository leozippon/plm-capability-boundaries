#!/usr/bin/env python3
"""Execute CPU-only descriptive analysis of the complete frozen 33-model panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

# Pin before importing numerical libraries; this program never imports GPU tools.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.extensions.phenotype_consistency import (
    analyse, digest, frozen_category_labels, release_families, within_release_categories,
)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--within-category-source", type=Path,
                        help="Existing completed consistency directory; emit only supplementary frozen-category sensitivity")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    before = {"utc": datetime.now(timezone.utc).isoformat(), "cpu_only": True,
              "python": sys.executable, "python_version": sys.version,
              "threads": {v: os.environ[v] for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "cpu_count": os.cpu_count(), "disk_free_bytes": shutil.disk_usage(args.out).free,
              "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")}
    write_json(args.out / "resource_receipt.json", before)
    try:
        if args.within_category_source is not None:
            import pandas as pd
            source = args.within_category_source.resolve()
            completion = json.loads((source / "completion_receipt.json").read_text())
            if completion["status"] != "completed":
                raise ValueError("initial analysis not completed")
            for name, expected in completion["output_sha256"].items():
                if digest(source / name) != expected:
                    raise ValueError(f"initial output digest mismatch: {name}")
            releases = pd.read_csv(source / "release_means.csv")
            declaration = args.root / "scripts/capability/reporting/audit_followup_support.py"
            families = release_families(declaration)
            labels = frozen_category_labels(declaration)
            records = within_release_categories(releases, families, labels)
            lineage_paths = [source / "release_means.csv", source / "analysis.json", source / "completion_receipt.json",
                             declaration, Path(__file__), args.root / "src/capability/extensions/phenotype_consistency.py",
                             args.root / "tests/extensions/test_phenotype_consistency.py"]
            result = {"schema": "phenotype_consistency_within_frozen_categories_v1", "pairwise": records,
                      "source_sha256": {str(p.relative_to(args.root)): digest(p) for p in lineage_paths},
                      "initial_source_sha256": json.loads((source / "analysis.json").read_text())["source_sha256"],
                      "limitations": ["Frozen release categories, not protein-exposure or independent-training-history labels.",
                          "Each existing release mean equally weights its checkpoints. Eight releases per category; no pooling strata as independent replicates.",
                          "Descriptive sensitivity of overall associations to broad-category separation; no p-values, confidence intervals, bootstrap, new fits or inference.",
                          "Small strata, restricted ranges, endpoint/baseline differences and possible biological overlap limit interpretation."]}
            pd.DataFrame(records).to_csv(args.out / "within_category_pairwise.csv", index=False)
            scatter = releases.copy()
            scatter["category_label"] = scatter.release_category.map(labels)
            scatter.to_csv(args.out / "labelled_release_scatter.tsv", sep="\t", index=False)
            write_json(args.out / "analysis.json", result)
            write_json(args.out / "completion_receipt.json", {"status": "completed", "releases": len(releases),
                       "category_n": releases.groupby("release_category").size().to_dict(),
                       "elapsed_seconds": time.monotonic() - start,
                       "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "output_sha256": {p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()},
                       "preserved_initial_outputs": True, "no_fits": True, "no_model_inference": True})
            print(json.dumps(result["pairwise"], indent=2))
            return
        joined, releases, result = analyse(args.root.resolve())
        joined.to_csv(args.out / "joined_checkpoints.csv", index=False)
        releases.to_csv(args.out / "release_means.csv", index=False)
        import pandas as pd
        pd.DataFrame(result.pop("source_rows")).to_csv(args.out / "source_cells.csv", index=False)
        pd.DataFrame(result["pairwise"]).to_csv(args.out / "pairwise_stats.csv", index=False)
        pd.DataFrame(result["leave_one_release_out"]).to_csv(args.out / "leave_one_release_out.csv", index=False)
        for path in (Path(__file__), args.root / "src/capability/extensions/phenotype_consistency.py"):
            result["source_sha256"][str(path.relative_to(args.root))] = digest(path)
        write_json(args.out / "analysis.json", result)
        outputs = {p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()}
        write_json(args.out / "completion_receipt.json", {"status": "completed", "utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_seconds": time.monotonic() - start, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "checkpoints": len(joined), "releases": len(releases), "output_sha256": outputs,
                   "no_model_inference": True, "no_fits": True, "no_bootstrap": True})
        print(json.dumps({"status": "completed", "out": str(args.out), "pairwise": result["pairwise"]}, indent=2))
    except Exception as error:
        write_json(args.out / "failure_receipt.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
