#!/usr/bin/env python3
"""Tabulate the requalified references against the archived -m 200 fits and the partial -m 2000 refit.

The gains differ in criterion: archived gains use a row-hash 20% split, new gains use whole identity
clusters with a purge, so a change in gain is not evidence about convergence.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.context.family_potts import sha256_file

OUT = Path(sys.argv[1])
ARCHIVE = Path("archive/logs/R3/pairwise_homologs_20260924")
m200 = {b["name"]: b for b in json.loads((ARCHIVE / "baseline/baseline_q.json").read_text())["backgrounds"]}
m2000 = {b["name"]: b for b in json.loads((ARCHIVE / "converge/baseline_q.json").read_text())["backgrounds"]}
new = json.loads((OUT / "qualification.json").read_text())["domains"]
keys = ("selected_lambda_e", "held_out_gain_over_independent_site", "final_converged", "converged_in_sweep", "selected_at_grid_edge")
rows = {}
for name, t in new.items():
    q = t["qualification"]
    rows[name] = {
        "archived_m200_rowhash": {k: m200[name].get(k) for k in keys},
        "archived_m2000_rowhash": {k: m2000[name].get(k) for k in keys} if name in m2000 else None,
        "requalified_cluster_split": {"selected_lambda_e": q["selected_lambda_e"], "held_out_gain_over_independent_site": q["held_out_gain_over_independent_site"],
                                      "final_converged": t["final_fit"]["converged"], "converged_in_sweep": q["sweep_fits_converged"],
                                      "final_iterations": t["final_fit"]["iterations"]}}
doc = {"schema": "family_reference_crosscheck_v1",
       "archived_m200_sha256": sha256_file(ARCHIVE / "baseline/baseline_q.json"),
       "archived_m2000_sha256": sha256_file(ARCHIVE / "converge/baseline_q.json"),
       "note": "gain criteria differ (row-hash split vs cluster split with purge); grids differ (5 vs 9 values)", "domains": rows}
(OUT / "archived_crosscheck.json").write_text(json.dumps(doc, indent=1) + "\n")
manifest = {p.name: sha256_file(p) for p in sorted(OUT.glob("*")) if p.is_file() and p.name != "manifest.json"}
(OUT / "manifest.json").write_text(json.dumps({"sha256": manifest}, indent=1) + "\n")
for name, r in rows.items():
    print(name, r["archived_m200_rowhash"]["final_converged"], r["archived_m200_rowhash"]["selected_lambda_e"], r["archived_m200_rowhash"]["selected_at_grid_edge"],
          "->", r["requalified_cluster_split"]["final_converged"], r["requalified_cluster_split"]["selected_lambda_e"])
