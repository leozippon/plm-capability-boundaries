#!/usr/bin/env python3
"""Re-derive the manuscript's numbers, say exactly what changed, and re-check.

This is the path a recomputation lands on. When the readout-class sweep and the
per-block re-extraction report, their artifacts replace the ones the readout and
crossed-control families read; running this re-reads every artifact, rebuilds the
derived-number table and the figure-data contract, diffs the new table against
the previous one quantity by quantity, and re-runs the consistency checker.

The change report names every quantity whose value, interval, support, unit,
resampling unit, seed set, level or source digest moved, every quantity that
appeared or disappeared, every figure input table whose artifact binding
changed, and the checker's verdict before and after. Nothing is written into the
manuscript: the report is what a manuscript edit is made from.

    python scripts/capability/reporting/regenerate_manuscript_evidence.py

``--dry-run`` rebuilds into a scratch directory and reports the diff without
replacing the committed table or its regeneration report.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.capability.core.evidence_ledger import REPO_ROOT  # noqa: E402
from scripts.capability.reporting.build_derived_numbers import OUT_DIR, build  # noqa: E402
from scripts.capability.reporting.check_manuscript_evidence import check  # noqa: E402

COMPARED_FIELDS = ("value", "unit", "interval_low", "interval_high", "interval_kind",
                   "resampling_unit", "resampling_draws", "support_id", "seed_set", "level",
                   "provisional", "conditioning", "conditioning_settled", "verdict",
                   "weighting_convention", "source_sha256")


def rows(payload: dict) -> dict[str, dict]:
    return {row["id"]: row for row in payload["quantities"]}


def diff_tables(before: dict | None, after: dict) -> dict[str, object]:
    if before is None:
        return {"first_build": True, "appeared": sorted(rows(after)), "disappeared": [],
                "changed": [], "unchanged": 0}
    old, new = rows(before), rows(after)
    changed = []
    for identifier in sorted(set(old) & set(new)):
        moved = {field: [old[identifier][field], new[identifier][field]]
                 for field in COMPARED_FIELDS
                 if old[identifier].get(field) != new[identifier].get(field)}
        if moved:
            changed.append({"id": identifier, "claim": new[identifier]["claim"], "moved": moved})
    return {
        "first_build": False,
        "appeared": sorted(set(new) - set(old)),
        "disappeared": sorted(set(old) - set(new)),
        "changed": changed,
        "unchanged": len(set(old) & set(new)) - len(changed),
    }


def diff_gaps(before: dict | None, after: dict) -> dict[str, list[str]]:
    old = {gap["id"] for gap in (before or {}).get("gaps", [])}
    new = {gap["id"] for gap in after.get("gaps", [])}
    return {"opened": sorted(new - old), "closed": sorted(old - new)}


def diff_contract(before: dict | None, after: dict) -> dict[str, object]:
    if before is None:
        return {"first_build": True, "tables_with_changed_bindings": []}
    old = (before or {}).get("tables", {})
    new = after.get("tables", {})
    changed = []
    for name in sorted(set(old) & set(new)):
        if old[name].get("sha256") != new[name].get("sha256") or \
                old[name].get("artifact_bindings") != new[name].get("artifact_bindings"):
            changed.append(name)
    return {
        "first_build": False,
        "tables_with_changed_bindings": changed,
        "tables_added": sorted(set(new) - set(old)),
        "tables_removed": sorted(set(old) - set(new)),
        "figures_without_a_complete_binding": after.get("figures_without_a_complete_binding", []),
    }


def regenerate(out_dir: Path = OUT_DIR, *, dry_run: bool = False) -> dict[str, object]:
    table_path = out_dir / "derived-numbers.json"
    contract_path = out_dir / "figure-data-contract.json"
    report_path = out_dir / "consistency-report.json"
    before_table = json.loads(table_path.read_text()) if table_path.is_file() else None
    before_contract = json.loads(contract_path.read_text()) if contract_path.is_file() else None
    before_report = json.loads(report_path.read_text()) if report_path.is_file() else None

    target = out_dir
    scratch = None
    if dry_run:
        scratch = Path(tempfile.mkdtemp(prefix="manuscript-evidence-"))
        for name in ("checker-policy.json",):
            if (out_dir / name).is_file():
                shutil.copy(out_dir / name, scratch / name)
        target = scratch

    result = build(target)
    after_report = check(target)
    change = {
        "schema": "manuscript_evidence_regeneration_report_v1",
        "dry_run": dry_run,
        "table": diff_tables(before_table, result["table"]),
        "gaps": diff_gaps(before_table, result["table"]),
        "figure_data_contract": diff_contract(before_contract, result["contract"]),
        "checker": {
            "before": None if before_report is None else
                      {key: before_report[key] for key in ("status", "traced_numbers",
                                                           "untraced_numbers", "findings_by_check")},
            "after": {key: after_report[key] for key in ("status", "traced_numbers",
                                                         "untraced_numbers", "findings_by_check")},
        },
    }
    if scratch is None:
        (target / "regeneration-report.json").write_text(json.dumps(change, indent=2) + "\n")
    else:
        shutil.rmtree(scratch)
    return change


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="rebuild into a scratch directory and report the diff only")
    options = parser.parse_args()
    change = regenerate(options.out_dir, dry_run=options.dry_run)
    print(json.dumps({
        "dry_run": change["dry_run"],
        "quantities_changed": len(change["table"]["changed"]),
        "quantities_appeared": len(change["table"]["appeared"]),
        "quantities_disappeared": len(change["table"]["disappeared"]),
        "gaps_opened": change["gaps"]["opened"],
        "gaps_closed": change["gaps"]["closed"],
        "figure_tables_with_changed_bindings":
            change["figure_data_contract"].get("tables_with_changed_bindings", []),
        "checker": change["checker"]["after"],
        "report": None if options.dry_run else
                  str((options.out_dir / "regeneration-report.json").relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
