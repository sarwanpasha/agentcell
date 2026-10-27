#!/usr/bin/env python3
"""Fixed selection/test split of BBBC021 compounds, stratified by MoA (seed 0). Compound codes are
written as they appear in the aggregated feature tables."""
import json, sys, numpy as np, glob
z = np.load(sorted(glob.glob(sys.argv[1] + "/vit_imagenet_cp/DAPI_e0.npz"))[0], allow_pickle=True)
comp, moa = z["compound"], z["moa"]
rng = np.random.default_rng(0); sel, test = [], []
for m in sorted(set(moa)):
    cs = sorted(set(comp[moa == m])); rng.shuffle(cs); k = (len(cs) + 1) // 2
    sel += cs[:k]; test += cs[k:]
json.dump({"select": [str(c) for c in sel], "test": [str(c) for c in test]}, open(sys.argv[2], "w"), indent=1)
print(len(sel), len(test))
