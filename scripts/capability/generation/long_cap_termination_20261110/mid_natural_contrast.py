"""Natural 513-767 contrast (the longest sequences the models finish), tuned minus base. Rows with length >= 671
were folded as two chunks of 16 samples; both arms are affected alike. B=5000 within-arm bootstrap."""
import sys
sys.path.insert(0, "/Data/lzp/plm-capability-boundaries")
import numpy as np
import pandas as pd
H = "/Data/lzp/plm-capability-boundaries/results/long_cap_termination_20261110/"
t = pd.read_csv(H + "analysis/per_structure_joined.tsv", sep="\t"); rng = np.random.default_rng(20261113)
m = t[(~t.capped) & (t.L >= 513) & (t.L <= 767)]; rows = []
for ep in ("plddt", "ptm"):
    b, tu = m[m.tuned == 0][ep].values, m[m.tuned == 1][ep].values
    bs = [tu[rng.integers(0, len(tu), len(tu))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(5000)]
    rows.append(dict(endpoint=ep, n_base=len(b), n_tuned=len(tu), base_mean=b.mean(), tuned_mean=tu.mean(), diff=tu.mean() - b.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                     L_base=m[m.tuned == 0].L.mean(), L_tuned=m[m.tuned == 1].L.mean(), identity_base=m[m.tuned == 0].identity.mean(), identity_tuned=m[m.tuned == 1].identity.mean()))
r = pd.DataFrame(rows); r.to_csv(H + "analysis/mid_natural_513_767.tsv", sep="\t", index=False, float_format="%.4f"); print(r.round(3).to_string())

# identity adjustment within the same rows
from src.capability.context import homology as hm  # noqa: E402
ORDER = list(hm.STRATUM_NAMES)
def coef(d, ep, strata, lin, comp):
    X = [np.ones(len(d)), d.tuned.values, ((d.L - d.L.mean()) / d.L.std()).values]
    if strata:
        for s in ORDER[1:]: X.append((d.stratum_id == s).values.astype(float))
    if lin: X.append(((d.identity - d.identity.mean()) / d.identity.std()).values)
    if comp:
        for c_ in ("zlib9", "d3"): X.append(((d[c_] - d[c_].mean()) / d[c_].std()).values)
    return np.linalg.lstsq(np.column_stack(X), d[ep].values, rcond=None)[0][1]
rows = []
for ep in ("plddt", "ptm"):
    for lab, kw in (("length only", dict(strata=False, lin=False, comp=False)), ("+ identity strata", dict(strata=True, lin=False, comp=False)), ("+ identity linear", dict(strata=False, lin=True, comp=False)), ("+ identity strata + composition", dict(strata=True, lin=False, comp=True))):
        b, tu = m[m.tuned == 0], m[m.tuned == 1]; bs = []
        for _ in range(5000):
            bs.append(coef(pd.concat([b.iloc[rng.integers(0, len(b), len(b))], tu.iloc[rng.integers(0, len(tu), len(tu))]]), ep, **kw))
        rows.append(dict(endpoint=ep, model=lab, tuned_minus_base=coef(m, ep, **kw), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
r = pd.DataFrame(rows); r.to_csv(H + "analysis/mid_natural_513_767_identity_adjusted.tsv", sep="\t", index=False, float_format="%.4f"); print(r.round(3).to_string())
print(m.groupby(["arm_tag", "stratum_id"]).agg(n=("plddt", "size"), plddt=("plddt", "mean")).round(2))
