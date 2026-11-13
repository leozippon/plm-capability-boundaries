"""Run in the pod (stdlib only): one row per folded structure of the companion tree. No analysis.
pLDDT column is the stored mean CA-pLDDT on 0-100; pTM is raw 0-1."""
import json
import sys
from pathlib import Path
T = Path(sys.argv[1]); out = []
sigs = set()
for d in sorted((T / "objects").iterdir()):
    f = d / "result.json"
    if not f.is_file(): continue
    r = json.loads(f.read_text()); sigs.add(r.get("evaluation_signature"))
    out.append((r["sequence_sha256"], r["length"], r["status"], r.get("reason", ""), r.get("mean_ca_plddt", ""), r.get("ptm", ""),
                r.get("diffusion_sample_index", ""), r.get("evaluation_signature"), json.dumps(r.get("diffusion_sample_splits")), r.get("peak_gpu_memory_bytes", ""), r.get("elapsed_seconds", "")))
w = sys.stdout.write
w("# companion cohort, one row per structure. mean_ca_plddt_0_100 is native [0,1] x 100; ptm_0_1 is raw.\n")
w("\t".join(["sequence_sha256", "length_residues", "status", "reason", "mean_ca_plddt_0_100", "ptm_0_1", "diffusion_sample_index", "evaluation_signature", "diffusion_sample_splits", "peak_gpu_memory_bytes", "elapsed_seconds"]) + "\n")
for r in out: w("\t".join(map(str, r)) + "\n")
sys.stderr.write("rows %d signatures %s\n" % (len(out), sorted(map(str, sigs))))
