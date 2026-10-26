#!/usr/bin/env python3
"""Tabulate study results: per-run NSC/NSCB and mean +- sd per epoch budget."""
import json, glob, os, statistics as st, re
rows = []
for d in sorted(glob.glob(os.path.expanduser("~/agentcell_work/runs/study/ep*_seed*"))):
    m = re.search(r"ep(\d+)_seed(\d+)", d); ep, s = int(m.group(1)), int(m.group(2))
    r = {"epochs": ep, "seed": s}
    for n in ["plain", "per_channel_l2"]:
        f = f"{d}/score_{n}.json"
        if os.path.exists(f):
            mm = json.load(open(f))["metrics"]; r[f"NSC_{n}"] = mm.get("NSC"); r[f"NSCB_{n}"] = mm.get("NSCB")
    rows.append(r)
print("epochs,seed,NSC_l2,NSCB_l2,NSC_plain,NSCB_plain")
for r in rows:
    print(",".join(str(r.get(k, "")) for k in ["epochs", "seed", "NSC_per_channel_l2", "NSCB_per_channel_l2", "NSC_plain", "NSCB_plain"]))
for ep in sorted({r["epochs"] for r in rows}):
    for k in ["NSC_per_channel_l2", "NSCB_per_channel_l2", "NSC_plain", "NSCB_plain"]:
        v = [r[k] for r in rows if r["epochs"] == ep and r.get(k) is not None]
        if v:
            print(f"ep{ep} {k}: n={len(v)} mean={st.mean(v):.4f} sd={(st.stdev(v) if len(v)>1 else 0):.4f} min={min(v):.4f} max={max(v):.4f}")
