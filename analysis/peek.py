#!/usr/bin/env python3
"""Print NSC/NSCB per available epoch for runs in a gram directory. Usage: peek.py GRAMDIR [RUN ...]"""
import sys, re, numpy as np
from pathlib import Path
gd = Path(sys.argv[1]); runs = sys.argv[2:] or sorted(p.name for p in gd.iterdir() if p.is_dir())
CH = ["DAPI", "Tubulin", "Actin"]
for run in runs:
    fs = {}
    for f in (gd / run).glob("*_e*.npz"):
        ch, ep = re.match(r"(\w+?)_e(\d+)\.npz", f.name).groups(); fs[(ch, int(ep))] = f
    eps = sorted({e for (c, e) in fs if all((cc, e) in fs for cc in CH)})
    for ep in eps:
        zs = [np.load(fs[(c, ep)], allow_pickle=True) for c in CH]
        comp, moa, batch = zs[0]["compound"], zs[0]["moa"], zs[0]["batch"]
        Gs = [z["G"] for z in zs]
        out = []
        for norm in ("plain", "l2"):
            if norm == "plain":
                S = sum(Gs); d = np.sqrt(np.diag(S)); S = S / np.outer(d, d)
            else:
                S = sum(g / np.outer(np.sqrt(np.diag(g)), np.sqrt(np.diag(g))) for g in Gs)
            for useb in (False, True):
                ok = []
                for i in range(len(comp)):
                    m = comp != comp[i]
                    if useb: m &= batch != batch[i]
                    if m.any():
                        j = np.where(m)[0][np.argmax(S[i, m])]; ok.append(moa[j] == moa[i])
                out.append(np.mean(ok))
        single = []
        for g in Gs:
            d = np.sqrt(np.diag(g)); S = g / np.outer(d, d); ok = []
            for i in range(len(comp)):
                m = comp != comp[i]; j = np.where(m)[0][np.argmax(S[i, m])]; ok.append(moa[j] == moa[i])
            single.append(np.mean(ok))
        print(f"{run:22s} e{ep:<4d} plain {out[0]:.4f}/{out[1]:.4f}  l2 {out[2]:.4f}/{out[3]:.4f}  single " +
              " ".join(f"{x:.3f}" for x in single))
