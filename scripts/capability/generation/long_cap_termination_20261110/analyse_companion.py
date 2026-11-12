"""Companion-cohort analysis. Primary reading: naturally terminated sequences in the shared 16-512 strata,
tuned minus base, with the length shift carried (regression and bin matching), then the identity adjustment, then
the capped populations. Rows are resampled within arm; B=2000; seed fixed. pLDDT is 0-100, pTM raw 0-1."""
import json, sys, zlib
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/Data/lzp/plm-capability-boundaries")
from src.capability.context import homology as hm

H = "/Data/lzp/plm-capability-boundaries/results/long_cap_termination_20261110/"
rng = np.random.default_rng(20261111); B = 2000
lab = pd.read_csv(H + "fold/attempt_labels.tsv", sep="\t")
import glob
fold = pd.concat([pd.read_csv(f, sep="\t", comment="#") for f in sorted(glob.glob(H + "fold_table_*.tsv"))], ignore_index=True)
assert fold.sequence_sha256.is_unique and fold.evaluation_signature.nunique() == 1
seq = {json.loads(l)["id"]: json.loads(l)["sequence"] for l in open(H + "fold/fold_cohort.jsonl")}
nov = pd.DataFrame([json.loads(l) for l in open(H + "novelty/search/generated_novelty.jsonl")])
nov = nov.rename(columns={"nearest_corpus_identity": "identity"})[["id", "identity", "had_any_hit"]]
nov["identity"] = nov.identity.fillna(0.0)
t = lab.merge(fold.rename(columns={"length_residues": "Lf", "mean_ca_plddt_0_100": "plddt", "ptm_0_1": "ptm"}), on="sequence_sha256", how="left", validate="1:1")
unfolded = t[t.status.isna()]
print("attempts not folded (declared subsample or part not yet folded):", unfolded.groupby("population").size().to_dict())
assert (unfolded.length > 512).all(), "a length<=512 row is unfolded"
t = t[t.status.notna()].copy()
t = t.merge(nov, left_on="sequence_sha256", right_on="id", how="left", validate="1:1")
assert (t.Lf == t.length).all()
notok = t[t.status != "ok"]
t = t[t.status == "ok"].copy()
t["L"] = t.length.astype(float)
t["tuned"] = (t.arm_tag == "tuned").astype(float)
t["capped"] = t.stop_status == "budget_censored"
t["stratum_id"] = t.identity.map(hm.assign_stratum)
def comp(s):
    return dict(zlib9=len(zlib.compress(s.encode(), 9)) / len(s), d3=len({s[i:i + 3] for i in range(len(s) - 2)}) / max(1, len(s) - 2))
c = pd.DataFrame([comp(seq[d]) for d in t.sequence_sha256], index=t.index); t = pd.concat([t, c], axis=1)
ZLIB_THRESHOLD = 0.4361   # fixed by the original cohort's capped-degeneracy split before any score was read
t["degenerate"] = t.zlib9 < ZLIB_THRESHOLD
ORDER = list(hm.STRATUM_NAMES)
STR = {"16-128": (16, 128), "129-256": (129, 256), "257-512": (257, 512)}
ci = lambda x: np.percentile(np.asarray(x)[np.isfinite(x)], [2.5, 97.5])
out = {}

def basis(L, kn, flex):
    cols = [L / 100.]
    if flex: cols += [(L / 100.) ** 2, (L / 100.) ** 3] + [np.maximum(L / 100. - k, 0) ** 3 for k in kn]
    return np.column_stack(cols)
def reg(d, ep, kn, flex, extra=(), strata=False):
    X = [np.ones(len(d)), d.tuned.values, basis(d.L.values, kn, flex)]
    if strata:
        for sname in ORDER[1:]: X.append((d.stratum_id == sname).values.astype(float)[:, None])
    for e in extra: X.append(((d[e] - d[e].mean()) / d[e].std()).values[:, None])
    X = np.column_stack(X); return np.linalg.lstsq(X, d[ep].values, rcond=None)[0][1]
def bin_match(d, ep, w=16):
    b, tu = d[d.tuned == 0], d[d.tuned == 1]; num = den = 0.
    for k in np.unique((tu.L // w).values):
        mb = (b.L // w) == k; mt = (tu.L // w) == k
        if mb.sum() == 0: continue
        num += mt.sum() * (tu[mt][ep].mean() - b[mb][ep].mean()); den += mt.sum()
    return num / den if den else np.nan
def boot(d, fn):
    b, tu = d[d.tuned == 0], d[d.tuned == 1]; r = []
    for _ in range(B):
        r.append(fn(pd.concat([b.iloc[rng.integers(0, len(b), len(b))], tu.iloc[rng.integers(0, len(tu), len(tu))]])))
    return r
def estimates(d, ep, kn):
    return dict(raw=d[d.tuned == 1][ep].mean() - d[d.tuned == 0][ep].mean(), linear=reg(d, ep, kn, False), spline=reg(d, ep, kn, True), bin16=bin_match(d, ep))

nat = t[~t.capped]
# 1. realised counts
cnt = t.assign(stratum=pd.cut(t.L, [15, 128, 256, 512, 768], labels=["16-128", "129-256", "257-512", "513-768"])).groupby(["population", "stratum"], observed=False).size().unstack(fill_value=0)
cnt["all"] = cnt.sum(axis=1); cnt.to_csv(H + "analysis/counts_by_population_stratum.tsv", sep="\t"); print(cnt, "\nnot ok:", len(notok))

# 2. primary: natural, shared strata
rows = []
prim = nat[(nat.L >= 16) & (nat.L <= 512)]
for ep in ("plddt", "ptm"):
    for name, (lo, hi) in list(STR.items()) + [("16-512 pooled", (16, 512))]:
        d = prim[(prim.L >= lo) & (prim.L <= hi)]
        kn = np.percentile(d.L / 100., [25, 50, 75]); pt = estimates(d, ep, kn)
        bs = {k: [] for k in pt}
        for e in boot(d, lambda x: estimates(x, ep, kn)):
            for k in pt: bs[k].append(e[k])
        r = dict(endpoint=ep, stratum=name, n_base=int((d.tuned == 0).sum()), n_tuned=int((d.tuned == 1).sum()),
                 mean_L_base=d[d.tuned == 0].L.mean(), mean_L_tuned=d[d.tuned == 1].L.mean(),
                 base_mean=d[d.tuned == 0][ep].mean(), tuned_mean=d[d.tuned == 1][ep].mean())
        for k in pt: r[k] = pt[k]; r[k + "_lo"], r[k + "_hi"] = ci(bs[k])
        rows.append(r)
pd.DataFrame(rows).to_csv(H + "analysis/primary_natural_16_512.tsv", sep="\t", index=False, float_format="%.4f")
print(pd.DataFrame(rows).round(3).to_string())

# 3. identity adjustment on the primary sample
rows = []
for ep in ("plddt", "ptm"):
    kn = np.percentile(prim.L / 100., [25, 50, 75])
    f = lambda x, ex, st: reg(x, ep, kn, True, ex, st)
    for lab_, ex, st in (("length spline only", (), False), ("length spline + identity (linear)", ("identity",), False),
                         ("length spline + identity stratum dummies (4 declared strata)", (), True),
                         ("length spline + identity stratum dummies + composition (zlib9, d3)", ("zlib9", "d3"), True)):
        pt = f(prim, ex, st); bs = boot(prim, lambda x: f(x, ex, st)); lo, hi = ci(bs)
        rows.append(dict(endpoint=ep, model=lab_, arm_coef_tuned_minus_base=pt, lo=lo, hi=hi))
pd.DataFrame(rows).to_csv(H + "analysis/primary_identity_adjusted.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(rows).round(3).to_string())
rows = []
for s in ORDER:
    for ep in ("plddt", "ptm"):
        g = prim[prim.stratum_id == s]; xb, xt = g[g.tuned == 0][ep].values, g[g.tuned == 1][ep].values
        r = dict(identity_stratum=s, endpoint=ep, n_base=len(xb), n_tuned=len(xt))
        if len(xb) >= 2 and len(xt) >= 2:
            bs = [xt[rng.integers(0, len(xt), len(xt))].mean() - xb[rng.integers(0, len(xb), len(xb))].mean() for _ in range(B)]
            r.update(base_mean=xb.mean(), tuned_mean=xt.mean(), diff=xt.mean() - xb.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                     margin=max(2.0, 0.25 * max(0.0, 70 - xb.mean())) if ep == "plddt" else np.nan,
                     L_base=g[g.tuned == 0].L.mean(), L_tuned=g[g.tuned == 1].L.mean())
        rows.append(r)
pd.DataFrame(rows).to_csv(H + "analysis/primary_identity_stratified.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(rows).round(3).to_string())
rows = []
for sname in ORDER:
    for ep in ("plddt", "ptm"):
        g = prim[prim.stratum_id == sname]; nb_, nt_ = int((g.tuned == 0).sum()), int((g.tuned == 1).sum())
        r = dict(identity_stratum=sname, endpoint=ep, n_base=nb_, n_tuned=nt_)
        if nb_ >= 10 and nt_ >= 10:
            r["bin16"] = bin_match(g, ep); bs = boot(g, lambda x: bin_match(x, ep)); r["lo"], r["hi"] = ci(bs)
        rows.append(r)
pd.DataFrame(rows).to_csv(H + "analysis/primary_identity_stratified_length_matched.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(rows).round(3).to_string())
rows = []
for grp, g in (("natural 16-512", prim), ("all natural", nat), ("capped", t[t.capped])):
    xb, xt = g[g.tuned == 0].identity.values, g[g.tuned == 1].identity.values
    if len(xb) < 2 or len(xt) < 2: continue
    bs = [xt[rng.integers(0, len(xt), len(xt))].mean() - xb[rng.integers(0, len(xb), len(xb))].mean() for _ in range(B)]
    rows.append(dict(group=grp, n_base=len(xb), n_tuned=len(xt), mean_identity_base=xb.mean(), mean_identity_tuned=xt.mean(), shift=xt.mean() - xb.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                     frac_lt30_base=(xb < 30).mean(), frac_lt30_tuned=(xt < 30).mean()))
pd.DataFrame(rows).to_csv(H + "analysis/identity_shift.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(rows).round(3).to_string())

# 4. populations: levels, degeneracy, identity association
rows = []
for pop, g in t.groupby("population"):
    r = dict(population=pop, n=len(g), plddt_mean=g.plddt.mean(), plddt_sd=g.plddt.std(), ptm_mean=g.ptm.mean(), identity_mean=g.identity.mean(), frac_degenerate=g.degenerate.mean(), zlib9_median=g.zlib9.median(), d3_median=g.d3.median(), L_mean=g.L.mean())
    for ep in ("plddt", "ptm"):
        rho = stats.spearmanr(g.identity, g[ep]); r["spearman_identity_" + ep] = rho[0]
    rows.append(r)
pd.DataFrame(rows).to_csv(H + "analysis/population_levels.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(rows).round(3).to_string())
rows = []
for pop in ("base-capped", "tuned-capped", "base-natural", "tuned-natural"):
    g = t[t.population == pop]
    for deg in (True, False):
        h = g[g.degenerate == deg]
        if len(h): rows.append(dict(population=pop, degenerate=deg, n=len(h), plddt_mean=h.plddt.mean(), ptm_mean=h.ptm.mean(), identity_mean=h.identity.mean(), zlib9_median=h.zlib9.median(), L_mean=h.L.mean(),
                                    plddt_lo=ci([h.plddt.values[rng.integers(0, len(h), len(h))].mean() for _ in range(B)])[0], plddt_hi=ci([h.plddt.values[rng.integers(0, len(h), len(h))].mean() for _ in range(B)])[1]))
pd.DataFrame(rows).to_csv(H + "analysis/degeneracy_split.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(rows).round(3).to_string())
# capped contrasts: base capped vs tuned capped, with and without identity/composition, and degenerate-adjusted
rows = []
cap = t[t.capped]
for ep in ("plddt", "ptm"):
    xb, xt = cap[cap.tuned == 0][ep].values, cap[cap.tuned == 1][ep].values
    bs = [xt[rng.integers(0, len(xt), len(xt))].mean() - xb[rng.integers(0, len(xb), len(xb))].mean() for _ in range(B)]
    rows.append(dict(contrast="tuned capped - base capped", endpoint=ep, n_base=len(xb), n_tuned=len(xt), diff=xt.mean() - xb.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    nd = cap[~cap.degenerate]; xb, xt = nd[nd.tuned == 0][ep].values, nd[nd.tuned == 1][ep].values
    bs = [xt[rng.integers(0, len(xt), len(xt))].mean() - xb[rng.integers(0, len(xb), len(xb))].mean() for _ in range(B)]
    rows.append(dict(contrast="tuned capped - base capped, non-degenerate rows only", endpoint=ep, n_base=len(xb), n_tuned=len(xt), diff=xt.mean() - xb.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    for lab_, g in (("base capped - base natural", (t[t.population == "base-capped"], t[t.population == "base-natural"])), ("tuned capped - tuned natural", (t[t.population == "tuned-capped"], t[t.population == "tuned-natural"]))):
        x, y = g[0][ep].values, g[1][ep].values
        bs = [x[rng.integers(0, len(x), len(x))].mean() - y[rng.integers(0, len(y), len(y))].mean() for _ in range(B)]
        rows.append(dict(contrast=lab_, endpoint=ep, n_base=len(y), n_tuned=len(x), diff=x.mean() - y.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
pd.DataFrame(rows).to_csv(H + "analysis/capped_contrasts.tsv", sep="\t", index=False, float_format="%.4f"); print(pd.DataFrame(rows).round(3).to_string())
t.drop(columns=["id"]).to_csv(H + "analysis/per_structure_joined.tsv", sep="\t", index=False, float_format="%.5g")
