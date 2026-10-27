#!/usr/bin/env python3
"""Embed many checkpoints of one channel and store per-checkpoint treatment Gram matrices.

For each checkpoint (final model and saved snapshots) of a training directory, the
script embeds the annotated and DMSO images of one channel, applies TVN, aggregates
to the 103 treatments (same code path as tool_embed) and writes
    <gram-dir>/<run>/<CH>_e<epoch>.npz
holding the treatment-level Gram matrix G = X X^T and the treatment metadata.
The epoch is read from the checkpoint itself, and every backbone tensor must load.

Usage:
  embed_ckpts.py --train-dir DIR --channel CH --run NAME --gram-dir GDIR
  embed_ckpts.py --model vit_imagenet|resnet50 --channel CH --run NAME --gram-dir GDIR
"""
import argparse, os, re, sys, types
from pathlib import Path
import numpy as np, pandas as pd, torch

TOOLS = os.environ["AGENTCELL_TOOLS"]; REPO = os.environ["WSDINO_REPO"]
sys.path.insert(0, TOOLS)
import tool_embed as te
import tool_score as ts

ap = argparse.ArgumentParser()
ap.add_argument("--train-dir"); ap.add_argument("--model", default="vit_trained")
ap.add_argument("--channel", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--gram-dir", required=True); ap.add_argument("--only-final", action="store_true")
a = ap.parse_args()

data = Path(os.environ["AGENTCELL_DATA"])
te.setup_dist(); os.chdir(REPO); sys.path.insert(0, REPO)
import Eval_BBBC021 as ev
import vision_transformer as vits
from torch.utils.data import DataLoader, SequentialSampler
ev.x0train = pd.read_csv(data / "BBBC021_annotated_wsdino.csv")
ev.x0DMSO = pd.read_csv(data / "BBBC021_DMSO_wsdino.csv")
ev.args = types.SimpleNamespace(batch_size_per_gpu=1, num_workers=4, use_cuda=True, arch="vit_small", patch_size=8,
    channel_headers=["Image_FileName_DAPI", "Image_FileName_Tubulin", "Image_FileName_Actin"],
    weak_label_header="Unique_Compounds")
ci = te.CHANNELS_ALL.index(a.channel)
ds = ev.ReturnIndexDataset(ev.x0train, ci); ds2 = ev.ReturnIndexDataset_DMSO(ev.x0DMSO, ci)
dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=1, num_workers=4, pin_memory=True, drop_last=True)
dl2 = DataLoader(ds2, sampler=SequentialSampler(ds2), batch_size=1, num_workers=4, pin_memory=True, drop_last=True)


def load_trained(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = vits.vit_small(patch_size=8, num_classes=12)
    bb = {}
    for k, v in ck["teacher"].items():
        nk = k[len("module."):] if k.startswith("module.") else k
        if nk.startswith("backbone."):
            bb[nk[len("backbone."):]] = v
    missing, unexpected = m.load_state_dict(bb, strict=False)
    missing = [k for k in missing if not k.startswith("head")]
    assert not missing and not unexpected, (path, missing[:5], unexpected[:5])
    return m.eval().cuda(), int(ck["epoch"])


jobs = []
if a.model == "vit_trained":
    d = Path(a.train_dir)
    final = d / f"{a.channel}_weak_compound_DINO_checkpoint.pth"
    assert final.exists(), final
    jobs.append(final)
    if not a.only_final:
        jobs += sorted(d.glob(f"{a.channel}_weak_compound_DINO_checkpoint[0-9][0-9][0-9][0-9].pth"))
else:
    jobs.append(None)

out = Path(a.gram_dir) / a.run; out.mkdir(parents=True, exist_ok=True)
for path in jobs:
    if path is None:
        model, _, _ = te.build_model(a.model, None); ep = 0
    else:
        model, ep = load_trained(path)
        if path.name.endswith("checkpoint.pth"):
            pass  # final model; epoch already verified by the training job
    dst = out / f"{a.channel}_e{ep}.npz"
    if dst.exists():
        print("skip", dst); continue
    tag = f"{a.run}_{a.channel}_e{ep}".replace("/", "_")
    with torch.no_grad():
        tf = ev.extract_features(model, dl, True).cuda(); df = ev.extract_features2(model, dl2, True).cuda()
    ev.Aggregate_features_NSC(ev.correct_tvn(df, tf), a.channel, tag)
    merged, keep = ts.load_channel(tag, a.channel, REPO)
    merged = merged.sort_values("treatment").reset_index(drop=True)
    X = merged[keep].to_numpy(np.float64)
    np.savez_compressed(dst, G=X @ X.T, treatment=merged["treatment"].astype(int).to_numpy(),
                        compound=merged["compound"].astype(str).to_numpy(), moa=merged["moa"].astype(str).to_numpy(),
                        batch=merged["batch"].astype(str).to_numpy(), epoch=ep,
                        source=str(path) if path else a.model)
    os.remove(Path(REPO) / f"aggregated_features_{a.channel}_epoch_{tag}.csv")
    print("WROTE", dst, "epoch", ep, "n", len(X), flush=True)
    del model; torch.cuda.empty_cache()
