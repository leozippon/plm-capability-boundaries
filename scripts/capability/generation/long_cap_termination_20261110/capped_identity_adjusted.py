"""Identity adjustment of the capped contrast: tuned capped (31) against the declared base-capped subsample (69).
Same rows, strata and composition measures as analyse_companion.py; rows resampled within arm, B=5000."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/Data/lzp/plm-capability-boundaries")
from src.capability.context import homology as hm
H = "/Data/lzp/plm-capability-boundaries/results/long_cap_termination_20261110/"
t = pd.read_csv(H + "analysis/per_structure_joined.tsv", sep="\t"); c = t[t.capped].copy()
ORDER = list(hm.STRATUM_NAMES); rng = np.random.default_rng(20261112); B = 5000
def coef(d, ep, strata, comp, lin):
    X = [np.ones(len(d)), d.tuned.values]
    if strata:
        for s in ORDER[1:]: X.append((d.stratum_id == s).values.astype(float))
    if lin: X.append(((d.identity - d.identity.mean()) / d.identity.std()).values)
    if comp:
        for m in ("zlib9", "d3"): X.append(((d[m] - d[m].mean()) / d[m].std()).values)
    return np.linalg.lstsq(np.column_stack(X), d[ep].values, rcond=None)[0][1]
rows = []
for ep in ("plddt", "ptm"):
    for lab, kw in (("unadjusted", dict(strata=False, comp=False, lin=False)), ("identity strata dummies", dict(strata=True, comp=False, lin=False)),
                    ("identity linear", dict(strata=False, comp=False, lin=True)), ("identity strata + composition", dict(strata=True, comp=True, lin=False))):
        pt = coef(c, ep, **kw); b, tu = c[c.tuned == 0], c[c.tuned == 1]; bs = []
        for _ in range(B):
            d = pd.concat([b.iloc[rng.integers(0, len(b), len(b))], tu.iloc[rng.integers(0, len(tu), len(tu))]])
            bs.append(coef(d, ep, **kw))
        rows.append(dict(endpoint=ep, model=lab, n_base=len(b), n_tuned=len(tu), tuned_minus_base=pt, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
r = pd.DataFrame(rows); r.to_csv(H + "analysis/capped_identity_adjusted.tsv", sep="\t", index=False, float_format="%.4f"); print(r.round(3).to_string())
print(c.groupby(["arm_tag", "stratum_id"]).agg(n=("plddt", "size"), plddt=("plddt", "mean")).round(2))
