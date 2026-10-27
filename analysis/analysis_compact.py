#!/usr/bin/env python3
"""Statistics from the compact outcome file written by export_compact.py.

Computes bootstrap intervals, exact McNemar tests, held-out grid and random search over 1,000
compound splits, checkpoint selection on held-out compounds, mixed-source scores and the LINCS
summary. Usage: analysis_compact.py compact.json report.json
"""
import json, sys, itertools
import numpy as np
from scipy import stats

d = json.load(open(sys.argv[1])); R = d["runs"]; rep = {}
rng = np.random.default_rng(0)


def vec(s):
    return np.array([np.nan if c == "-" else float(c) for c in s])


def acc(v, mask=None):
    v = v if mask is None else v[mask]
    return float(np.nanmean(v))


def boot(v, n=10000):
    v = v[~np.isnan(v)]; idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def mcnemar(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b); a, b = a[ok], b[ok]
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def final_ep(run):
    eps = [int(e) for e in R[run] if "plain" in R[run][e]]
    return str(max(eps))


def outc(run, ep, norm, metric="NSC"):
    return vec(R[run][ep][norm][metric])


# ------------------------------------------------ point estimates with bootstrap intervals
B = {}
for run in R:
    for ep, rec in R[run].items():
        if "plain" not in rec:
            continue
        for norm in ("plain", "l2"):
            for m in ("NSC", "NSCB"):
                v = vec(rec[norm][m]); B[f"{run}|{ep}|{norm}|{m}"] = {"acc": acc(v), "ci": boot(v, 2000)}
rep["scores"] = B

# ------------------------------------------------ McNemar tests
pairs = {
    "c100_s0 l2 vs untrained l2": (("c100_s0", "100", "l2"), ("vit_imagenet_cp", "0", "l2")),
    "c100_s0 plain vs untrained plain": (("c100_s0", "100", "plain"), ("vit_imagenet_cp", "0", "plain")),
    "c400_s0 l2 vs untrained l2": (("c400_s0", "400", "l2"), ("vit_imagenet_cp", "0", "l2")),
    "c100_s0 e1 l2 vs untrained l2": (("c100_s0", "1", "l2"), ("vit_imagenet_cp", "0", "l2")),
    "m20 l2 vs untrained l2": (("m20_s0", "20", "l2"), ("vit_imagenet_cp", "0", "l2")),
    "m20 plain vs untrained plain": (("m20_s0", "20", "plain"), ("vit_imagenet_cp", "0", "plain")),
    "c20 l2 vs untrained l2": (("c20_s0", "20", "l2"), ("vit_imagenet_cp", "0", "l2")),
    "L c100_s2 plain vs L untrained plain": (("L_c100_s2", "100", "plain"), ("L_vit_imagenet_cp", "0", "plain")),
    "L vit vs L resnet plain": (("L_vit_imagenet_cp", "0", "plain"), ("L_resnet50_cp", "0", "plain")),
}
rep["mcnemar"] = {}
for k, (x, y) in pairs.items():
    for m in ("NSC", "NSCB"):
        rep["mcnemar"][k + " " + m] = mcnemar(outc(*x, m), outc(*y, m))

# ------------------------------------------------ held-out grid / random search
mb = d["meta_bbbc"]; comp = np.array(mb["compound"]); moa = np.array(mb["moa"])
UNTR = {"vit": "vit_imagenet_cp", "rn": "resnet50_cp"}
cfg = {}   # name -> (cost, outcome vector NSC, NSCB)
for b, run in UNTR.items():
    for norm in ("plain", "l2"):
        cfg[f"{b} 3ch {norm}"] = (0.0, outc(run, "0", norm), outc(run, "0", norm, "NSCB"))
    # single DNA channel, NSC only (no NSCB stored for single channels)
    cfg[f"{b} DNA"] = (0.0, vec(R[run]["0"]["single"]["DAPI"]), None)
TR = {"c20_s0": ("compound", 20), "c100_s0": ("compound", 100), "c400_s0": ("compound", 400),
      "t20_s0": ("treatment", 20), "t100_s0": ("treatment", 100), "m20_s0": ("moa", 20)}
for run, (lab, ep) in TR.items():
    e = final_ep(run); assert int(e) == ep, (run, e)
    for norm in ("plain", "l2"):
        cfg[f"{lab}{ep} {norm}"] = (3 * ep * 0.0125, outc(run, e, norm), outc(run, e, norm, "NSCB"))
names = list(cfg)
comps = sorted(set(comp)); moas = sorted(set(moa))


def split():
    sel = []
    for m in moas:
        cs = sorted(set(comp[moa == m])); rng.shuffle(cs); sel += cs[:(len(cs) + 1) // 2]
    return np.isin(comp, sel)


def best(names_, S):
    sc = [(acc(cfg[n][1], S), -cfg[n][0]) for n in names_]
    return names_[max(range(len(names_)), key=lambda i: sc[i])]


def strategies(excl_moa):
    pool = [n for n in names if not (excl_moa and n.startswith("moa"))]
    res = {k: [] for k in ["F1 vit plain", "F2 vit l2", "agent config (compound100 l2)", "grid", "random k=3", "random k=5", "oracle on test"]}
    picks = {}
    for _ in range(1000):
        S = split(); T = ~S
        res["F1 vit plain"].append(acc(cfg["vit 3ch plain"][1], T))
        res["F2 vit l2"].append(acc(cfg["vit 3ch l2"][1], T))
        res["agent config (compound100 l2)"].append(acc(cfg["compound100 l2"][1], T))
        g = best(pool, S); picks[g] = picks.get(g, 0) + 1
        res["grid"].append(acc(cfg[g][1], T))
        for k in (3, 5):
            sub = list(rng.choice(pool, k, replace=False))
            res[f"random k={k}"].append(acc(cfg[best(sub, S)][1], T))
        res["oracle on test"].append(max(acc(cfg[n][1], T) for n in pool))
    out = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
               "q05": float(np.percentile(v, 5)), "q95": float(np.percentile(v, 95))} for k, v in res.items()}
    out["grid_picks"] = dict(sorted(picks.items(), key=lambda x: -x[1]))
    return out


rep["config_full"] = {n: {"cost": cfg[n][0], "NSC": acc(cfg[n][1]),
                          "NSCB": (acc(cfg[n][2]) if cfg[n][2] is not None else None)} for n in names}
rep["heldout_all"] = strategies(False)
rep["heldout_no_moa"] = strategies(True)

# fixed split seed 0 (the one given to agents)
r0 = np.random.default_rng(0); sel = []
for m in moas:
    cs = sorted(set(comp[moa == m])); r0.shuffle(cs); sel += cs[:(len(cs) + 1) // 2]
S0 = np.isin(comp, sel)
rep["split0"] = {"n_select_compounds": len(sel), "n_test_compounds": len(comps) - len(sel),
                 "configs": {n: {"select": acc(cfg[n][1], S0), "test": acc(cfg[n][1], ~S0)} for n in names}}

# ------------------------------------------------ checkpoint selection on held-out compounds
ck = {}
for run in ("c100_s0", "c400_s0", "c20_s0", "t100_s0", "t20_s0"):
    eps = sorted((int(e) for e in R[run] if "plain" in R[run][e]))
    for norm in ("plain", "l2"):
        sel_ep, test_sel, test_final = [], [], []
        for _ in range(1000):
            S = split(); T = ~S
            cand = [(acc(outc(run, str(e), norm), S), -e) for e in eps]
            eb = eps[max(range(len(eps)), key=lambda i: cand[i])]
            sel_ep.append(eb); test_sel.append(acc(outc(run, str(eb), norm), T))
            test_final.append(acc(outc(run, str(eps[-1]), norm), T))
        ck[f"{run} {norm}"] = {"epochs": eps, "median_selected_epoch": float(np.median(sel_ep)),
                               "frac_epoch1": float(np.mean(np.array(sel_ep) == 1)),
                               "test_selected": float(np.mean(test_sel)), "test_selected_q05": float(np.percentile(test_sel, 5)), "test_selected_q95": float(np.percentile(test_sel, 95)), "test_final": float(np.mean(test_final)),
                               "curve": {str(e): acc(outc(run, str(e), norm)) for e in eps}}
rep["checkpoint_selection"] = ck

# ------------------------------------------------ mixed sources
rep["mixed"] = {k: {"NSC": acc(vec(v["NSC"])), "NSCB": acc(vec(v["NSCB"])), "epoch": v["epoch"]}
                for k, v in d["mixed_sources"].items()}

# ------------------------------------------------ single channel and energy shares
rep["single"] = {f"{run}|{ep}": {c: acc(vec(s)) for c, s in rec["single"].items()}
                 for run in R for ep, rec in R[run].items()}
rep["energy"] = {f"{run}|{ep}": rec["energy_share"] for run in R for ep, rec in R[run].items() if "energy_share" in rec}
json.dump(rep, open(sys.argv[2], "w"), indent=1)
print("ok")
