"""Emit one flat row per ok structure in the pinned eighteen-shard sample (stdout variant: writes nothing on the cluster).

No analysis, no aggregation, no strata. Units are carried in the column names as
well as the header block, because the stored-times-100 pLDDT convention against the
predictor native 0-1 scale has already produced one wrong claim in this campaign and
a table that travels to another reader is exactly where that slip would recur.

Fails loudly rather than silently collapsing if any digest maps to more than one arm,
since the one-row-per-structure shape depends on that being 1:1.
"""
import json
import sys
from pathlib import Path

V = Path("/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer/results/external_baseline/20261110074842_20277c6a196f")
D = V / "ot_fold_design_probe"
NUM_SHARDS = 68
DONE = set(range(18))
SEED = 20260905
RULE = "sequence_digest"
OUT = V / "ot_deep_fold" / "checkpoint_structures_18of68.tsv"

arms = {k: sorted(v) for k, v in json.loads((D / "digest_arm_map.json").read_text()).items()}
rows, multi, sigs = [], [], set()
for d in sorted((V / "ot_structure_evidence" / "objects").iterdir()):
    f = d / "result.json"
    if not f.is_file():
        continue
    r = json.loads(f.read_text())
    sha = r["sequence_sha256"]
    shard = int(sha, 16) % NUM_SHARDS
    if shard not in DONE or r.get("status") != "ok":
        continue
    a = arms.get(sha, [])
    if len(a) != 1:
        multi.append((sha, a))
        continue
    sigs.add(r["evaluation_signature"])
    rows.append((a[0], sha, r["length"], r["mean_ca_plddt"], r["ptm"],
                 shard, SEED, RULE, r["diffusion_sample_index"], r["evaluation_signature"]))
if multi:
    sys.exit("REFUSING: %d digests map to other than exactly one arm, e.g. %s" % (len(multi), multi[:3]))
if len(sigs) != 1:
    sys.exit("REFUSING: %d distinct evaluation signatures in one sample: %s" % (len(sigs), sorted(sigs)))

COLS = ["arm", "sequence_sha256", "length_residues", "mean_ca_plddt_0_100", "ptm_0_1",
        "shard_index_of_68", "fold_seed", "fold_seed_rule", "diffusion_sample_index",
        "evaluation_signature"]
hdr = [
    "# per-structure fold measurements, campaign objective_tuned_20261109",
    "# fold snapshot 20261110074842_20277c6a196f ; predictor ESMFold2 at 32 diffusion samples",
    "# pinned sample: cohort-tree structures with (sha256 of sequence as integer) %% 68 in {0..17},",
    "#   i.e. the union of the eighteen shards whose rc sentinels read 0; status == ok only",
    "# one row per structure; a digest maps to exactly one arm in this sample and that is asserted, not assumed",
    "#",
    "# UNITS, stated here and repeated in the column names:",
    "#   length_residues        residues, integer",
    "#   mean_ca_plddt_0_100    CA-pLDDT on the 0-100 scale. This is the predictor native [0,1]",
    "#                          value MULTIPLIED BY 100. Do not compare it to a 0-1 figure.",
    "#   ptm_0_1                pTM on the predictor native 0-1 scale. NOT rescaled. Do not",
    "#                          multiply by 100 to match pLDDT.",
    "#   diffusion_sample_index index of the selected sample among the 32 drawn, integer 0-31;",
    "#                          selection rule is highest pTM. Two runs may select different",
    "#                          samples for the same sequence.",
    "#   fold_seed/fold_seed_rule  the seed and rule the evaluation signature pins",
    "# NOTE: these values are not bitwise reproducible on a re-fold even at this seed and rule.",
    "#",
    "\t".join(COLS),
]
import sys
if True:
    fh = sys.stdout
    fh.write("\n".join(hdr) + "\n")
    for r in sorted(rows, key=lambda x: (x[0], x[1])):
        fh.write("\t".join(str(v) for v in r) + "\n")
print("rows written: %d" % len(rows), file=sys.stderr)
print("distinct arms: %d" % len({r[0] for r in rows}), file=sys.stderr)
print("single evaluation signature: %s" % sorted(sigs)[0], file=sys.stderr)
print("path: %s" % OUT, file=sys.stderr)
print("columns: %s" % ", ".join(COLS), file=sys.stderr)
