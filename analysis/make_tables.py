#!/usr/bin/env python3
"""Write CSV tables and a short text summary from the report of analysis_compact.py.

Usage: make_tables.py report.json OUTDIR
"""
import csv, json, sys
from pathlib import Path

r = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
S = r["scores"]


def sc(run, ep, norm, m):
    k = f"{run}|{ep}|{norm}|{m}"
    return round(S[k]["acc"], 4) if k in S else ""


def write(name, header, rows):
    with open(out / name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# grid configurations scored on all 103 treatments
write("grid_configurations.csv", ["configuration", "cost_SU", "NSC", "NSCB"],
      [[n, v["cost"], round(v["NSC"], 4), "" if v["NSCB"] is None else round(v["NSCB"], 4)]
       for n, v in r["config_full"].items()])

# held-out selection strategies
rows = []
for pool, key in (("all labels", "heldout_all"), ("without MoA label", "heldout_no_moa")):
    for strat, v in r[key].items():
        if strat == "grid_picks":
            continue
        rows.append([pool, strat, round(v["mean"], 4), round(v["q05"], 4), round(v["q95"], 4)])
write("heldout_strategies.csv", ["pool", "strategy", "test_NSC_mean", "test_NSC_q05", "test_NSC_q95"], rows)

# checkpoint selection on held-out compounds
write("checkpoint_selection.csv",
      ["run", "normalization", "fraction_epoch1_selected", "test_NSC_selected", "test_NSC_final"],
      [[k.split()[0], k.split()[1], round(v["frac_epoch1"], 3), round(v["test_selected"], 4), round(v["test_final"], 4)]
       for k, v in r["checkpoint_selection"].items()])

# checkpoint curves (all runs, all epochs with three channels)
rows = []
for k in sorted(S):
    run, ep, norm, m = k.split("|")
    if norm == "plain" and m == "NSC":
        rows.append([run, int(ep), sc(run, ep, "plain", "NSC"), sc(run, ep, "plain", "NSCB"),
                     sc(run, ep, "l2", "NSC"), sc(run, ep, "l2", "NSCB")])
rows.sort(key=lambda x: (x[0], x[1]))
write("checkpoint_scores.csv", ["run", "epoch", "NSC_plain", "NSCB_plain", "NSC_l2", "NSCB_l2"], rows)

# mixed trained / untrained encoders
write("mixed_sources.csv", ["run", "encoders_DNA_tubulin_actin", "normalization", "epoch", "NSC", "NSCB"],
      [k.split("|") [:2] + [k.split("|")[2], v["epoch"], round(v["NSC"], 4), round(v["NSCB"], 4)]
       for k, v in r["mixed"].items()])

# McNemar tests
write("mcnemar_tests.csv", ["comparison", "only_first_correct", "only_second_correct", "P"],
      [[k, v["a_only"], v["b_only"], round(v["p"], 5)] for k, v in r["mcnemar"].items()])

# single-channel scores
write("single_channel.csv", ["run", "epoch", "DNA", "tubulin_or_ER", "actin_or_AGP"],
      [[k.split("|")[0], k.split("|")[1]] + [round(v[c], 4) if c in v else "" for c in ("DAPI", "Tubulin", "Actin")]
       for k, v in sorted(r["single"].items())])

h, hn, ck = r["heldout_all"], r["heldout_no_moa"], r["checkpoint_selection"]["c100_s0 l2"]
print(f"untrained ViT, plain:        NSC {sc('vit_imagenet_cp', '0', 'plain', 'NSC'):.4f}  NSCB {sc('vit_imagenet_cp', '0', 'plain', 'NSCB'):.4f}")
print(f"untrained ViT, per-ch. L2:   NSC {sc('vit_imagenet_cp', '0', 'l2', 'NSC'):.4f}  NSCB {sc('vit_imagenet_cp', '0', 'l2', 'NSCB'):.4f}")
print(f"compound 100 ep, L2, final:  NSC {sc('c100_s0', '100', 'l2', 'NSC'):.4f}  epoch 1: {sc('c100_s0', '1', 'l2', 'NSC'):.4f}")
print(f"held-out test NSC  F2 {h['F2 vit l2']['mean']:.3f}  agent config {h['agent config (compound100 l2)']['mean']:.3f}  "
      f"grid {h['grid']['mean']:.3f} (without MoA label {hn['grid']['mean']:.3f})")
print(f"checkpoint selection (compound 100, L2): test NSC {ck['test_selected']:.3f} vs final {ck['test_final']:.3f}")
print(f"LINCS untrained ViT plain NSC {sc('L_vit_imagenet_cp', '0', 'plain', 'NSC'):.4f}, "
      f"trained seed 2 epoch 100 {sc('L_c100_s2', '100', 'plain', 'NSC'):.4f}")
print("tables written to", out)
