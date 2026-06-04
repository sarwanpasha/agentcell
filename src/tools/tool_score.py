#!/usr/bin/env python3
"""
tool_score.py -- AGENT PRIMITIVE. Score per-channel aggregated features.

This is one of the primitive tools the autonomous agent composes. It is a thin
wrapper around the validated scoring logic (final_score_wsdino). It exposes ONE
consequential decision to the agent: the concatenation normalization, which
trades NSC against NSCB (we found per-channel-L2 -> NSC up / NSCB down; plain ->
the reverse). The agent must choose; there is no silent default that hides the
trade-off.

Structured-result protocol (ALL agent tools follow this):
  - Writes a JSON result file to --result-out with keys:
      {tool, status: "ok"|"error", decision: {...chosen args...},
       metrics: {...}, message, artifacts: [paths]}
  - Prints the same JSON to stdout (last line) so the agent can parse it
    directly from the observation.
  - Exit code 0 on success, nonzero on failure.

Usage (agent invokes):
  /usr/bin/python tool_score.py \
      --features-dir <dir with aggregated_features_<CH>_epoch_<tag>.csv> \
      --tag <tag> \
      --normalization {plain, per_channel_l2} \
      --result-out <path.json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CHANNELS = ["DAPI", "Tubulin", "Actin"]


def emit(result: dict, result_out: Path):
    """Write structured result and echo to stdout (last line = JSON)."""
    if result_out:
        result_out.parent.mkdir(parents=True, exist_ok=True)
        result_out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    sys.exit(0 if result.get("status") == "ok" else 1)


def load_channel(tag, channel, csv_dir):
    path = Path(csv_dir) / f"aggregated_features_{channel}_epoch_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path).rename(columns={"Unnamed: 0": "treatment"})
    df = df.rename(columns={df.columns[0]: "treatment"})
    labels = [c for c in ("compound", "moa", "batch") if c in df.columns]
    feats = [c for c in df.columns if c not in (["treatment"] + labels)]
    fnum = df[feats].apply(pd.to_numeric, errors="coerce")
    keep = list(fnum.columns[~fnum.isna().all()])
    g = df.groupby("treatment")
    fm = g[keep].median()
    meta = g[["compound", "moa", "batch"]].first()
    merged = fm.join(meta).reset_index()
    return merged, keep


def nn_accuracy(F, compound, moa, batch=None):
    """1-NN cosine MoA accuracy. If batch given, NSCB (exclude same compound
    OR same batch); else NSC (exclude same compound)."""
    tally = []
    for i in range(len(compound)):
        if batch is None:
            mask = compound != compound[i]
        else:
            mask = (compound != compound[i]) & (batch != batch[i])
        if not mask.any():
            continue
        sims = cosine_similarity(F[mask], F[i:i+1]).reshape(-1)
        nn = int(np.argmax(sims))
        tally.append(1 if moa[mask][nn] == moa[i] else 0)
    return (float(np.mean(tally)), len(tally)) if tally else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--normalization", choices=["plain", "per_channel_l2"],
                    required=True,
                    help="plain concat or per-channel L2 before concat "
                         "(trades NSCB vs NSC)")
    ap.add_argument("--result-out", type=Path, default=None)
    args = ap.parse_args()

    result = {"tool": "score", "status": "error",
              "decision": {"normalization": args.normalization, "tag": args.tag},
              "metrics": {}, "message": "", "artifacts": []}

    # Load and align whatever channels are present (supports single-channel
    # configs like DAPI-only, which the agent may legitimately want to test).
    try:
        blocks, base, present = [], None, []
        for ch in CHANNELS:
            path = Path(args.features_dir) / f"aggregated_features_{ch}_epoch_{args.tag}.csv"
            if not path.exists():
                continue  # channel not embedded for this tag; skip it
            merged, keep = load_channel(args.tag, ch, args.features_dir)
            present.append(ch)
            if base is None:
                base = merged[["treatment", "compound", "moa", "batch"]].copy()
            order = base["treatment"].values
            m = merged.set_index("treatment").loc[order, keep].to_numpy(float)
            blocks.append(m)
        if not blocks:
            result["message"] = (f"no channel feature files found for tag "
                                 f"'{args.tag}' in {args.features_dir}")
            emit(result, args.result_out)
    except FileNotFoundError as e:
        result["message"] = f"failed loading features: {e}"
        emit(result, args.result_out)

    compound = base["compound"].to_numpy()
    moa = base["moa"].to_numpy()
    batch = base["batch"].to_numpy()

    # Apply the chosen normalization.
    if args.normalization == "per_channel_l2":
        normed = []
        for b in blocks:
            n = np.linalg.norm(b, axis=1, keepdims=True)
            n[n == 0] = 1.0
            normed.append(b / n)
        concat = np.concatenate(normed, axis=1)
    else:  # plain
        concat = np.concatenate(blocks, axis=1)

    nsc, n_nsc = nn_accuracy(concat, compound, moa)
    nscb, n_nscb = nn_accuracy(concat, compound, moa, batch)

    result["status"] = "ok"
    result["metrics"] = {
        "NSC": round(nsc, 4), "NSC_n": n_nsc,
        "NSCB": round(nscb, 4), "NSCB_n": n_nscb,
        "feature_dim": int(concat.shape[1]),
        "n_treatments": int(concat.shape[0]),
        "channels_scored": present,
    }
    result["message"] = (f"Scored {concat.shape[0]} treatments at "
                         f"{concat.shape[1]}-dim ({'+'.join(present)}) with "
                         f"{args.normalization} normalization.")
    emit(result, args.result_out)


if __name__ == "__main__":
    main()
