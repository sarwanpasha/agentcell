#!/usr/bin/env python3
"""
tool_embed.py -- AGENT PRIMITIVE. Embed images -> per-channel treatment features.

Thin wrapper over the validated WS-DINO eval embedding path (4 center crops,
median, TVN on DMSO, aggregate to 103 treatments). Writes the per-channel
aggregated_features_<CH>_epoch_<tag>.csv files that tool_score consumes.

Consequential decisions exposed to the agent:
  --model    {resnet50, vit_imagenet, vit_trained}
               resnet50     : ImageNet ResNet50 trunk (2048-dim) - the weak,
                              non-morphology-specific baseline.
               vit_imagenet : DINO ImageNet ViT-S/8 (384-dim) - strong no-train.
               vit_trained  : a WS-DINO checkpoint the agent TRAINED earlier in
                              this run (path must come from a prior
                              tool_train_ssl result; see --init + --allow-external).
  --channels {all_three, dapi_only}
               all_three : embed DAPI+Tubulin+Actin separately, write 3 CSVs.
               dapi_only : embed only DAPI (the known biological under-use).
  --init     path to a trained checkpoint dir (required for vit_trained). Per the
             earned-access rule, the agent may only pass a checkpoint produced by
             its own prior tool_train_ssl step. For cheap loop-debugging we relax
             this with --allow-external (NOT used in real autonomy runs).

Structured-result protocol (same as tool_score): JSON to --result-out and as the
last stdout line, with {tool,status,decision,metrics,message,artifacts}.

Usage:
  /usr/bin/python tool_embed.py \
      --data-root ./runs/bbbc021_wsdino \
      --model vit_imagenet --channels all_three \
      --tag run1_step2 \
      --features-out ./runs/agent_results/run1 \
      --result-out  ./runs/agent_results/run1/embed_step2.json
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CHANNELS_ALL = ["DAPI", "Tubulin", "Actin"]
REPO_DIR = os.environ.get("WSDINO_REPO", os.path.expanduser("~/WS-DINO"))


def emit(result, result_out):
    if result_out:
        Path(result_out).parent.mkdir(parents=True, exist_ok=True)
        Path(result_out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    sys.exit(0 if result.get("status") == "ok" else 1)


def setup_dist():
    import random
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Unique port per process to avoid EADDRINUSE when many tool calls run in
    # sequence (a prior run's process group may not have released the port).
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(random.randint(29600, 39600))
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://", world_size=1, rank=0)


def build_model(model_kind, ckpt_path):
    """Return an eval-ready feature extractor. For vit_trained, ckpt_path is the
    specific channel's trained checkpoint to load."""
    sys.path.insert(0, REPO_DIR)
    if model_kind == "resnet50":
        from torchvision.models import resnet50
        m = resnet50(pretrained=True)
        m = torch.nn.Sequential(*list(m.children())[:-1])  # 2048-dim trunk
        m.eval().cuda()
        return m, "resnet50", 2048
    import vision_transformer as vits
    import utils
    m = vits.vit_small(patch_size=8, num_classes=12)
    if model_kind == "vit_imagenet":
        weights = os.path.join(REPO_DIR, "dino_deitsmall8_pretrain.pth")
        utils.load_pretrained_weights(m, weights, "teacher", "vit_small", 8)
    elif model_kind == "vit_trained":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        teacher_sd = ckpt["teacher"]
        bb = {}
        for k, v in teacher_sd.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            if nk.startswith("backbone."):
                bb[nk[len("backbone."):]] = v
        m.load_state_dict(bb, strict=False)
    m.eval().cuda()
    return m, model_kind, 384


def resolve_channel_ckpt(init_dir, channel):
    """Find a channel's trained checkpoint inside an init-dir.

    The agent's training writes to <run>/train_<CH>_<weak>/<CH>_weak_compound_
    DINO_checkpoint.pth. We search init_dir recursively for a checkpoint whose
    path names this channel."""
    from pathlib import Path as _P
    cands = list(_P(init_dir).rglob("*checkpoint.pth"))
    # prefer one whose train dir or filename names this channel
    for c in cands:
        if f"train_{channel}_" in str(c) or c.name.startswith(f"{channel}_"):
            return str(c)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", choices=["resnet50", "vit_imagenet", "vit_trained"],
                    required=True)
    ap.add_argument("--channels", choices=["all_three", "dapi_only"],
                    default="all_three")
    ap.add_argument("--init", default=None,
                    help="single checkpoint path for vit_trained (applies one "
                         "model to all requested channels; usually wrong for "
                         "multi-channel — prefer --init-dir)")
    ap.add_argument("--init-dir", default=None,
                    help="directory containing per-channel trained checkpoints "
                         "(each channel embedded with its OWN model — the correct "
                         "WS-DINO setup). Used with --model vit_trained.")
    ap.add_argument("--allow-external", action="store_true",
                    help="DEV ONLY: permit an externally-supplied checkpoint "
                         "(used for cheap loop-debugging, not real autonomy runs)")
    ap.add_argument("--run-train-artifacts", default=None,
                    help="JSON list of checkpoint paths this agent run has earned "
                         "via tool_train_ssl; --init must be among them")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--features-out", required=True,
                    help="dir to write aggregated_features_<CH>_epoch_<tag>.csv")
    ap.add_argument("--result-out", default=None)
    args = ap.parse_args()

    result = {"tool": "embed", "status": "error",
              "decision": {"model": args.model, "channels": args.channels,
                           "init": args.init, "tag": args.tag},
              "metrics": {}, "message": "", "artifacts": []}

    # Enforce the earned-trained-model rule, supporting per-channel init-dir.
    if args.model == "vit_trained":
        if not args.init and not args.init_dir:
            result["message"] = ("vit_trained requires --init-dir (per-channel "
                                  "trained checkpoints) or --init (single).")
            emit(result, args.result_out)
        if not args.allow_external:
            earned = []
            if args.run_train_artifacts and os.path.exists(args.run_train_artifacts):
                earned = json.loads(Path(args.run_train_artifacts).read_text())
            # The checkpoints we will actually use must all be earned.
            to_check = []
            if args.init:
                to_check.append(args.init)
            if args.init_dir:
                chans = CHANNELS_ALL if args.channels == "all_three" else ["DAPI"]
                for ch in chans:
                    ck = resolve_channel_ckpt(args.init_dir, ch)
                    if ck:
                        to_check.append(ck)
            bad = [c for c in to_check if c not in earned]
            if bad:
                result["message"] = (
                    "earned-access violation: these checkpoints were not produced "
                    f"by a prior tool_train_ssl step in this run: {bad}. Train them "
                    "first, or pass --allow-external for dev.")
                emit(result, args.result_out)

    data_root = Path(args.data_root)
    annotated = data_root / "BBBC021_annotated_wsdino.csv"
    dmso = data_root / "BBBC021_DMSO_wsdino.csv"
    if not annotated.exists() or not dmso.exists():
        result["message"] = f"missing CSVs under {data_root}"
        emit(result, args.result_out)

    setup_dist()
    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)
    import Eval_BBBC021 as ev
    ev.x0train = pd.read_csv(annotated)
    ev.x0DMSO = pd.read_csv(dmso)
    ev.args = types.SimpleNamespace(
        batch_size_per_gpu=1, num_workers=4, use_cuda=True,
        arch="vit_small", patch_size=8,
        channel_headers=["Image_FileName_DAPI", "Image_FileName_Tubulin",
                         "Image_FileName_Actin"],
        weak_label_header="Unique_Compounds",
    )

    channels = CHANNELS_ALL if args.channels == "all_three" else ["DAPI"]
    out_dir = Path(args.features_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # For vit_trained with --init-dir, each channel gets its OWN model (the
    # correct WS-DINO setup). Otherwise build one shared model up front.
    per_channel_ckpt = {}
    shared_model = None
    if args.model == "vit_trained" and args.init_dir:
        for ch in channels:
            ck = resolve_channel_ckpt(args.init_dir, ch)
            if not ck:
                result["message"] = (f"no trained checkpoint for channel {ch} "
                                     f"found under {args.init_dir}")
                emit(result, args.result_out)
            per_channel_ckpt[ch] = ck
    else:
        try:
            shared_model, model_name_shared, dim_shared = build_model(
                args.model, args.init)
        except Exception as e:
            result["message"] = f"model build failed: {e}"
            emit(result, args.result_out)

    from torch.utils.data import DataLoader, SequentialSampler
    written = []
    model_name, dim = None, None
    try:
        for ch_idx, ch in enumerate(channels):
            # pick the model for this channel
            if per_channel_ckpt:
                model, model_name, dim = build_model("vit_trained",
                                                     per_channel_ckpt[ch])
            else:
                model, model_name, dim = shared_model, model_name_shared, dim_shared
            ds = ev.ReturnIndexDataset(ev.x0train, CHANNELS_ALL.index(ch))
            ds2 = ev.ReturnIndexDataset_DMSO(ev.x0DMSO, CHANNELS_ALL.index(ch))
            dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=1,
                            num_workers=4, pin_memory=True, drop_last=True)
            dl2 = DataLoader(ds2, sampler=SequentialSampler(ds2), batch_size=1,
                             num_workers=4, pin_memory=True, drop_last=True)
            train_feats = ev.extract_features(model, dl, True).cuda()
            dmso_feats = ev.extract_features2(model, dl2, True).cuda()
            corrected = ev.correct_tvn(dmso_feats, train_feats)
            ev.Aggregate_features_NSC(corrected, ch, args.tag)
            csv_path = out_dir / f"aggregated_features_{ch}_epoch_{args.tag}.csv"
            src = Path(REPO_DIR) / f"aggregated_features_{ch}_epoch_{args.tag}.csv"
            if src.exists() and src.resolve() != csv_path.resolve():
                csv_path.write_text(src.read_text())
            written.append(str(csv_path))
    except Exception as e:
        import traceback
        result["message"] = f"embedding failed on channel '{ch}': {e}"
        result["artifacts"] = written
        result["traceback"] = traceback.format_exc()[-600:]
        emit(result, args.result_out)

    result["status"] = "ok"
    result["artifacts"] = written
    result["metrics"] = {"model": model_name, "feature_dim_per_channel": dim,
                         "channels_embedded": len(channels),
                         "per_channel_models": bool(per_channel_ckpt)}
    result["message"] = (f"Embedded {len(channels)} channel(s) with {model_name}"
                         f"{' (per-channel models)' if per_channel_ckpt else ''}; "
                         f"wrote {len(written)} CSV(s) tagged '{args.tag}'. "
                         f"Score with tool_score --tag {args.tag}.")
    emit(result, args.result_out)


if __name__ == "__main__":
    main()
