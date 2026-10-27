#!/usr/bin/env python3
"""Statistical analysis of the revision study from per-channel treatment Gram matrices.

Reads <gram-dir>/<run>/<CH>_e<epoch>.npz (written by embed_ckpts.py) and writes a JSON report with
  * NSC/NSCB of every configuration with 95% bootstrap intervals over treatments,
  * seed summaries, Welch tests and TOST equivalence tests between epoch budgets,
  * exact McNemar tests between paired configurations,
  * held-out selection: compounds split in halves stratified by MoA; strategies select on one half
    and are scored on the other (fixed pipeline, agent configuration, grid search, random search,
    checkpoint selection),
  * the site factorial (PyTorch version x preprocessing recipe),
  * checkpoint curves and single-channel scores, and the plain-concatenation channel weights.

Usage: analysis.py --gram-dir DIR --out report.json [--prefix ''] [--splits 1000]
"""
import argparse, itertools, json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--gram-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--prefix", default=""); ap.add_argument("--splits", type=int, default=1000)
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
CH = ["DAPI", "Tubulin", "Actin"]
SIGMA = 0.0125

# ---------------------------------------------------------------- load
G = defaultdict(dict)          # run -> (ch, epoch) -> gram
meta = None
for d in sorted(Path(a.gram_dir).iterdir()):
    if not d.is_dir() or not d.name.startswith(a.prefix):
        continue
    for f in d.glob("*_e*.npz"):
        ch, ep = re.match(r"(\w+?)_e(\d+)\.npz", f.name).groups()
        z = np.load(f, allow_pickle=True)
        G[d.name[len(a.prefix):]][(ch, int(ep))] = z["G"]
        m = (tuple(z["treatment"]), tuple(z["compound"]), tuple(z["moa"]), tuple(z["batch"]))
        if meta is None:
            meta = m
        assert m == meta, f"treatment order differs in {f}"
treat, comp, moa, batch = (np.array(x) for x in meta)
N = len(treat)
same_c = comp[:, None] == comp[None, :]
same_b = batch[:, None] == batch[None, :]
MASK = {"NSC": ~same_c, "NSCB": ~same_c & ~same_b}


def sim(grams, norm):
    if norm == "plain":
        S = sum(grams); d = np.sqrt(np.diag(S)); return S / np.outer(d, d)
    return sum(g / np.outer(np.sqrt(np.diag(g)), np.sqrt(np.diag(g))) for g in grams) / len(grams)


def correct(S, metric):
    M = MASK[metric]; out = np.full(N, np.nan)
    for i in range(N):
        if M[i].any():
            j = np.where(M[i])[0][np.argmax(S[i, M[i]])]
            out[i] = float(moa[j] == moa[i])
    return out


def channels_at(run, ep):
    g = G[run]
    return [g[(c, ep)] for c in CH] if all((c, ep) in g for c in CH) else None


def epochs(run):
    return sorted({e for (c, e) in G[run] if all((cc, e) in G[run] for cc in CH)})


def final_epoch(run):
    return max(epochs(run))


# ------------------------------------------------- configurations (finals)
CFG = {}        # name -> dict(nsc vec, nscb vec, cost, family)


def add(name, grams, norm, cost, fam):
    S = sim(grams, norm)
    CFG[name] = {"NSC": correct(S, "NSC"), "NSCB": correct(S, "NSCB"), "cost": cost, "family": fam}


for bb in ("vit_imagenet", "resnet50"):
    for pr in ("cp", "um"):
        run = f"{bb}_{pr}"
        if run in G:
            gr = channels_at(run, 0)
            for norm in ("plain", "l2"):
                add(f"{run}|all|{norm}", gr, norm, 0.0, f"untrained_{pr}")
            add(f"{run}|dapi|plain", [G[run][("DAPI", 0)]], "plain", 0.0, f"untrained_{pr}")
for run in G:
    if run.startswith(("vit_", "resnet")):
        continue
    ep = final_epoch(run)
    for norm in ("plain", "l2"):
        add(f"{run}|e{ep}|{norm}", channels_at(run, ep), norm, SIGMA * ep * 3, run.split("_s")[0])


def acc(v):
    v = v[~np.isnan(v)]; return float(v.mean())


def boot_ci(v, B=a.boot):
    v = v[~np.isnan(v)]; idx = rng.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def mcnemar(x, y):
    ok = ~np.isnan(x) & ~np.isnan(y); x, y = x[ok], y[ok]
    b = int(((x == 1) & (y == 0)).sum()); c = int(((x == 0) & (y == 1)).sum())
    p = 1.0 if b + c == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))
    return {"x_only": b, "y_only": c, "p_exact": p}


R = {"n_treatments": int(N), "n_compounds": int(len(set(comp))), "n_moa": int(len(set(moa))),
     "n_layouts_or_batches": int(len(set(batch)))}
R["configs"] = {k: {"NSC": acc(v["NSC"]), "NSC_ci": boot_ci(v["NSC"]), "NSCB": acc(v["NSCB"]),
                    "NSCB_ci": boot_ci(v["NSCB"]), "cost_SU": v["cost"]} for k, v in CFG.items()}

# ------------------------------------------------------------ seed summaries
groups = defaultdict(list)
for k in CFG:
    run, ep, norm = k.split("|")
    if "_s" in run and not run.startswith(("vit_", "resnet")):
        groups[(run.rsplit("_s", 1)[0], norm)].append(k)
R["seed_summary"] = {}
for (fam, norm), ks in sorted(groups.items()):
    for met in ("NSC", "NSCB"):
        v = np.array([acc(CFG[k][met]) for k in ks])
        R["seed_summary"][f"{fam}|{norm}|{met}"] = {"n": len(v), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()), "max": float(v.max()),
            "values": [float(x) for x in v]}


def tost(x, y, delta):
    x, y = np.asarray(x), np.asarray(y)
    d = y.mean() - x.mean(); vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy); df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    p1 = 1 - stats.t.cdf((d + delta) / se, df); p2 = stats.t.cdf((d - delta) / se, df)
    lo, hi = d - stats.t.ppf(0.95, df) * se, d + stats.t.ppf(0.95, df) * se
    return {"diff": float(d), "ci90": [float(lo), float(hi)], "p_tost": float(max(p1, p2)), "delta": delta,
            "welch_p": float(stats.ttest_ind(x, y, equal_var=False).pvalue),
            "mwu_p": float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)}


R["budget_tests"] = {}
for norm in ("l2", "plain"):
    for met in ("NSC", "NSCB"):
        k1, k4 = f"c100|{norm}|{met}", f"c400|{norm}|{met}"
        if k1 in R["seed_summary"] and k4 in R["seed_summary"]:
            x, y = R["seed_summary"][k1]["values"], R["seed_summary"][k4]["values"]
            R["budget_tests"][f"{norm}|{met}"] = {f"delta_{d}": tost(x, y, d) for d in (0.01, 0.02, 0.03)}

# trained (agent configuration) versus untrained, paired per seed
R["paired"] = {}
for pr in ("cp",):
    for norm, ref in (("l2", f"vit_imagenet_{pr}|all|l2"), ("plain", f"vit_imagenet_{pr}|all|plain")):
        if ref not in CFG:
            continue
        out = []
        for k in sorted(k for k in CFG if k.startswith("c100_s") and k.endswith("|" + norm)):
            out.append({"config": k, **mcnemar(CFG[k]["NSC"], CFG[ref]["NSC"])})
        R["paired"][f"c100_vs_untrained|{norm}"] = out

# ------------------------------------------------------------ held-out selection
moas = sorted(set(moa)); cmp_moa = {c: m for c, m in zip(comp, moa)}
by_moa = defaultdict(list)
for c in sorted(set(comp)):
    by_moa[cmp_moa[c]].append(c)


def split():
    S = set()
    for m, cs in by_moa.items():
        cs = list(cs); rng.shuffle(cs)
        k = len(cs) // 2 + (rng.random() < 0.5 if len(cs) % 2 else 0)
        S |= set(cs[:k])
    s = np.array([c in S for c in comp]); return s, ~s


GRID = [k for k in CFG if (CFG[k]["family"] == "untrained_cp") or
        re.match(r"^[ctm](20|100|400)_s0\|", k)]
AGENT = "c100_s0|e100|l2"
FIXED = "vit_imagenet_cp|all|plain"
R["grid"] = {"configs": GRID, "total_training_cost_SU": float(sum(CFG[k]["cost"] for k in GRID) / 2)}
ckpt_runs = [r for r in G if re.match(r"^c100_s\d$", r)]
sel = defaultdict(list)
cost = np.array([CFG[k]["cost"] for k in GRID])
for _ in range(a.splits):
    s, t = split()
    for met in ("NSC", "NSCB"):
        Sv = np.array([np.nanmean(CFG[k][met][s]) for k in GRID]); Tv = np.array([np.nanmean(CFG[k][met][t]) for k in GRID])
        best = np.lexsort((cost, -Sv))[0]
        sel[f"{met}|grid_select"].append(Tv[best]); sel[f"{met}|grid_optimism"].append(Sv[best] - Tv[best])
        sel[f"{met}|grid_oracle"].append(Tv.max())
        sel[f"{met}|fixed_F1"].append(np.nanmean(CFG[FIXED][met][t]))
        if AGENT in CFG:
            sel[f"{met}|agent_config"].append(np.nanmean(CFG[AGENT][met][t]))
        for kk in (1, 3, 5, 10):
            vals = []
            for _r in range(50):
                pick = rng.choice(len(GRID), size=min(kk, len(GRID)), replace=False)
                b = pick[np.lexsort((cost[pick], -Sv[pick]))[0]]; vals.append(Tv[b])
            sel[f"{met}|random_k{kk}"].append(float(np.mean(vals)))
    # checkpoint selection within each 100-epoch seed (per-channel L2), selected on S, scored on T
    for run in ckpt_runs:
        eps = epochs(run)
        sc = {e: correct(sim(channels_at(run, e), "l2"), "NSC") for e in eps}
        e_best = max(eps, key=lambda e: (np.nanmean(sc[e][s]), -e))
        sel["NSC|ckpt_select_l2"].append(np.nanmean(sc[e_best][t])); sel["NSC|ckpt_final_l2"].append(np.nanmean(sc[max(eps)][t]))
        sel["NSC|ckpt_selected_epoch"].append(e_best)
R["heldout"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "p2.5": float(np.percentile(v, 2.5)),
                    "p97.5": float(np.percentile(v, 97.5))} for k, v in sel.items()}
if "NSC|ckpt_selected_epoch" in sel:
    R["heldout"]["NSC|ckpt_selected_epoch"]["counts"] = {str(int(e)): int(c) for e, c in
        zip(*np.unique(sel["NSC|ckpt_selected_epoch"], return_counts=True))}

# ------------------------------------------------------------ factorial
fac = {}
for name, pat in (("torch27_cp", r"^c100_s[0-2]$"), ("torch251_cp", r"^f_251_cp_s\d$"),
                  ("torch27_um", r"^f_sys_um_s\d$"), ("torch251_um", r"^f_251_um_s\d$")):
    runs = sorted(r for r in G if re.match(pat, r))
    for norm in ("l2", "plain"):
        v = [acc(CFG[f"{r}|e{final_epoch(r)}|{norm}"]["NSC"]) for r in runs]
        if v:
            fac[f"{name}|{norm}"] = {"n": len(v), "mean": float(np.mean(v)), "values": v}
R["factorial"] = fac

# ------------------------------------------------------------ checkpoint curves
curves = {}
for fam in ("c100", "c400", "c20", "t100", "m100"):
    runs = sorted(r for r in G if re.match(rf"^{fam}_s\d$", r))
    if not runs:
        continue
    eps = sorted(set.intersection(*[set(epochs(r)) for r in runs]))
    for norm in ("l2", "plain"):
        curves[f"{fam}|{norm}"] = {str(e): [acc(correct(sim(channels_at(r, e), norm), "NSC")) for r in runs] for e in eps}
R["curves"] = curves

# ------------------------------------------------------------ single channel and channel weights
single = {}
for run in [r for r in G if re.match(r"^(c100_s\d|vit_imagenet_cp)$", r)]:
    ep = final_epoch(run) if not run.startswith("vit") else 0
    single[run] = {c: acc(correct(sim([G[run][(c, ep)]], "plain"), "NSC")) for c in CH}
R["single_channel"] = single
wts = {}
for run in [r for r in G if re.match(r"^(c100_s\d|vit_imagenet_cp)$", r)]:
    ep = final_epoch(run) if not run.startswith("vit") else 0
    nrm = np.array([np.sqrt(np.diag(G[run][(c, ep)])) for c in CH])     # 3 x N
    wts[run] = dict(zip(CH, (nrm ** 2 / (nrm ** 2).sum(0)).mean(1).round(4).tolist()))
R["plain_channel_weight"] = wts

# ------------------------------------------------------------ mixed channel sources
# Every combination of trained (final) and untrained ImageNet encoders across the three channels,
# for each 100-epoch compound seed. This is what an order-dependent checkpoint resolver can produce
# when the ImageNet initialization file sits next to the trained checkpoint.
mix = {}
if "vit_imagenet_cp" in G:
    for run in sorted(r for r in G if re.match(r"^c100_s\d$", r)):
        ep = final_epoch(run)
        for combo in itertools.product((0, 1), repeat=3):
            grams = [G[run][(c, ep)] if b else G["vit_imagenet_cp"][(c, 0)] for c, b in zip(CH, combo)]
            key = "".join("T" if b else "I" for b in combo)
            for norm in ("l2", "plain"):
                S = sim(grams, norm)
                mix.setdefault(f"{key}|{norm}", []).append(
                    {"run": run, "NSC": acc(correct(S, "NSC")), "NSCB": acc(correct(S, "NSCB"))})
R["mixed_sources"] = mix

Path(a.out).write_text(json.dumps(R, indent=1))
print("wrote", a.out)
