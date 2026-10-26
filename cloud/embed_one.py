#!/usr/bin/env python3
"""Embed ONE channel with ONE trained WS-DINO checkpoint (same path as tool_embed:
4 centre crops, TVN on DMSO, aggregate to 103 treatments) and write
aggregated_features_<CH>_epoch_<tag>.csv into --features-out. Lets the three
channels of a seed be trained on different machines and scored together."""
import argparse, os, sys, types
from pathlib import Path
import pandas as pd, torch
TOOLS = os.environ["AGENTCELL_TOOLS"]; REPO = os.environ["WSDINO_REPO"]
sys.path.insert(0, TOOLS)
import tool_embed as te
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--channel", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--features-out", required=True)
a = ap.parse_args()
data = Path(os.environ["AGENTCELL_DATA"])
te.setup_dist(); os.chdir(REPO); sys.path.insert(0, REPO)
import Eval_BBBC021 as ev
from torch.utils.data import DataLoader, SequentialSampler
ev.x0train = pd.read_csv(data / "BBBC021_annotated_wsdino.csv")
ev.x0DMSO = pd.read_csv(data / "BBBC021_DMSO_wsdino.csv")
ev.args = types.SimpleNamespace(batch_size_per_gpu=1, num_workers=4, use_cuda=True, arch="vit_small", patch_size=8,
    channel_headers=["Image_FileName_DAPI", "Image_FileName_Tubulin", "Image_FileName_Actin"],
    weak_label_header="Unique_Compounds")
model, _, _ = te.build_model("vit_trained", a.ckpt)
ci = te.CHANNELS_ALL.index(a.channel)
ds = ev.ReturnIndexDataset(ev.x0train, ci); ds2 = ev.ReturnIndexDataset_DMSO(ev.x0DMSO, ci)
dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=1, num_workers=4, pin_memory=True, drop_last=True)
dl2 = DataLoader(ds2, sampler=SequentialSampler(ds2), batch_size=1, num_workers=4, pin_memory=True, drop_last=True)
tf = ev.extract_features(model, dl, True).cuda(); df = ev.extract_features2(model, dl2, True).cuda()
ev.Aggregate_features_NSC(ev.correct_tvn(df, tf), a.channel, a.tag)
src = Path(REPO) / f"aggregated_features_{a.channel}_epoch_{a.tag}.csv"
dst = Path(a.features_out) / src.name; dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(src.read_text()); print("WROTE", dst)
