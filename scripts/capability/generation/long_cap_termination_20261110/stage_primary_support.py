"""Stage the companion cohort's declared primary support in the exact shape the R6 cohort's declared-band
producer reads, so that band is obtained by invoking that producer rather than by re-implementing it.

Support: the seventeenth amendment's primary reading -- naturally terminated sequences of 16 to 512 residues
(`results/long_cap_termination_20261110/analysis/per_structure_joined.tsv`, rows with `capped` false and 16 <= L <= 512),
which is the support `analysis/primary_identity_stratified.tsv` is already computed on. Columns are renamed to the
producer's input contract and no value is transformed; `identity` is the maximum percent identity over the query
that the producer calls `identity_over_query`. The producer's `10b` pair names this cohort's two arms, so its
other three pairs contribute no estimable cell and no band column.

This directory is produced by three commands from the repository root, in this order:

  P=/home/lzp/miniconda3/envs/ct/bin/python
  D=results/long_cap_termination_20261110/analysis/declared_simultaneous_band/
  E=scripts/capability/generation/length_adjusted_contrasts_20261110
  $P scripts/capability/generation/long_cap_termination_20261110/stage_primary_support.py
  PYTHONPATH=. RESDIR=$D $P $E/identity_stratified_contrast.py            > $D/declared_band.log 2>&1
  PYTHONPATH=. RESDIR=$D $P $E/identity_family_a_without_thin_cells.py    > $D/thin_cell_free_sensitivity.log 2>&1

The second command writes the declared family-A band and the per-cell simultaneous intervals; the third writes the
thin-cell-free band that both producers label a post-hoc sensitivity. Both are invoked unmodified. Their secondary
outputs here (`identity_shift_secondary.tsv`, `identity_stratum_levels_all_arms.tsv`) are by-products of that reuse
and are not this directory's subject; this cohort's own identity-shift and level tables sit one level up.
"""
import hashlib, pathlib
import pandas as pd

ROOT = pathlib.Path("/Data/lzp/plm-capability-boundaries")
SRC = ROOT / "results/long_cap_termination_20261110/analysis/per_structure_joined.tsv"
OUT = ROOT / "results/long_cap_termination_20261110/analysis/declared_simultaneous_band/natural_neighbour/per_structure_nearest_neighbour.tsv"
PRODUCERS = ["scripts/capability/generation/length_adjusted_contrasts_20261110/identity_stratified_contrast.py",
             "scripts/capability/generation/length_adjusted_contrasts_20261110/identity_family_a_without_thin_cells.py"]

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
t = pd.read_csv(SRC, sep="\t")
prim = t[(~t.capped) & (t.L >= 16) & (t.L <= 512)]
assert len(prim) == 1518, len(prim)
assert set(prim.arm) == {"protgpt3-10b", "protgpt3-10b-dpo"}, "the producer's '10b' pair must name these two arms"
assert not prim.capped.any() and prim[["plddt", "ptm", "identity"]].notna().all().all()
out = prim[["arm", "plddt", "ptm", "capped"]].assign(identity_over_query=prim.identity.values)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, sep="\t", index=False)
print("source      ", SRC.relative_to(ROOT), sha(SRC))
for p in PRODUCERS:
    print("producer    ", p, sha(ROOT / p))
print("staged      ", OUT.relative_to(ROOT), sha(OUT), len(out), "rows")
print(out.groupby("arm").size().to_dict())
