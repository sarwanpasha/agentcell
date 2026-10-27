#!/usr/bin/env python3
"""Export per-treatment retrieval outcomes from Gram matrices into one compact JSON file.

For every run and every epoch with all three channels, the file stores, for plain and per-channel
L2 normalization, whether each treatment's nearest admissible neighbor (NSC and NSCB rules) shares
its MoA, plus single-channel outcomes, the mean energy share of each channel under plain
concatenation, and outcomes for every mix of trained and untrained ImageNet encoders. Outcomes are
written as strings of '0', '1' and '-' (no admissible neighbor). Usage: export_compact.py GRAMDIR OUT.json
"""
import json, re, sys, itertools
from pathlib import Path
import numpy as np

gd = Path(sys.argv[1]); CH = ["DAPI", "Tubulin", "Actin"]
out = {"runs": {}}; meta = None


def outcomes(S, comp, moa, batch, use_batch):
    s = []
    for i in range(len(comp)):
        m = comp != comp[i]
        if use_batch:
            m &= batch != batch[i]
        if not m.any():
            s.append("-"); continue
        j = np.where(m)[0][np.argmax(S[i, m])]
        s.append("1" if moa[j] == moa[i] else "0")
    return "".join(s)


def sims(Gs):
    P = sum(Gs); d = np.sqrt(np.diag(P)); plain = P / np.outer(d, d)
    l2 = sum(g / np.outer(np.sqrt(np.diag(g)), np.sqrt(np.diag(g))) for g in Gs)
    return {"plain": plain, "l2": l2}


grams = {}
for rd in sorted(p for p in gd.iterdir() if p.is_dir()):
    fs = {}
    for f in rd.glob("*_e*.npz"):
        ch, ep = re.match(r"(\w+?)_e(\d+)\.npz", f.name).groups(); fs[(ch, int(ep))] = f
    grams[rd.name] = fs
    for (ch, ep), f in sorted(fs.items()):
        z = np.load(f, allow_pickle=True)
        key = (rd.name[:2] == "L_")
        m = {"compound": [str(x) for x in z["compound"]], "moa": [str(x) for x in z["moa"]],
             "batch": [str(x) for x in z["batch"]], "treatment": [int(x) for x in z["treatment"]]}
        out.setdefault("meta_lincs" if key else "meta_bbbc", m)
        break


def load(fs, ch, ep):
    return np.load(fs[(ch, ep)], allow_pickle=True)


for run, fs in grams.items():
    if not fs:
        continue
    mk = "meta_lincs" if run.startswith("L_") else "meta_bbbc"
    comp = np.array(out[mk]["compound"]); moa = np.array(out[mk]["moa"]); batch = np.array(out[mk]["batch"])
    R = {}
    for ep in sorted({e for (c, e) in fs}):
        chans = [c for c in CH if (c, ep) in fs]
        Gs = {c: load(fs, c, ep)["G"] for c in chans}
        rec = {"single": {c: outcomes(Gs[c] / np.outer(np.sqrt(np.diag(Gs[c])), np.sqrt(np.diag(Gs[c]))), comp, moa, batch, False)
                          for c in chans}}
        if len(chans) == 3:
            for norm, S in sims([Gs[c] for c in CH]).items():
                rec[norm] = {"NSC": outcomes(S, comp, moa, batch, False), "NSCB": outcomes(S, comp, moa, batch, True)}
            nrm = np.array([np.diag(Gs[c]) for c in CH])
            rec["energy_share"] = dict(zip(CH, (nrm / nrm.sum(0)).mean(1).round(4).tolist()))
        R[str(ep)] = rec
    out["runs"][run] = R

# mixtures of trained (final epoch) and untrained encoders
mix = {}
for run, fs in grams.items():
    if not re.match(r"^(c100_s\d|c400_s\d|gh_sys_cp_s\d)$", run):
        continue
    base = "gh_vit_imagenet_cp" if run.startswith("gh_") else "vit_imagenet_cp"
    if base not in grams or not grams[base]:
        continue
    eps = [e for e in sorted({e for (c, e) in fs}) if all((c, e) in fs for c in CH)]
    if not eps:
        continue
    ep = max(eps); comp = np.array(out["meta_bbbc"]["compound"]); moa = np.array(out["meta_bbbc"]["moa"])
    batch = np.array(out["meta_bbbc"]["batch"])
    for combo in itertools.product((0, 1), repeat=3):
        Gs = [load(fs, c, ep)["G"] if b else load(grams[base], c, 0)["G"] for c, b in zip(CH, combo)]
        key = "".join("T" if b else "I" for b in combo)
        for norm, S in sims(Gs).items():
            mix[f"{run}|{key}|{norm}"] = {"NSC": outcomes(S, comp, moa, batch, False),
                                          "NSCB": outcomes(S, comp, moa, batch, True), "epoch": ep}
out["mixed_sources"] = mix
Path(sys.argv[2]).write_text(json.dumps(out, separators=(",", ":")))
print("wrote", sys.argv[2], len(out["runs"]), "runs", len(mix), "mixes")
