#!/usr/bin/env python3
"""
tool_train_ssl.py -- AGENT PRIMITIVE. Train a WS-DINO model (one channel).

Thin wrapper over the validated train_wsdino_channel.sbatch machinery. Because
training takes ~5h, this tool is ASYNCHRONOUS: it submits an sbatch job and
returns immediately with a job id and an SU-cost estimate. A separate --status
mode lets the agent poll for completion and obtain the produced checkpoint path.

Consequential decisions exposed to the agent:
  --weak-label {compound, moa, treatment, none}
        The WS-DINO weak supervision signal. 'compound' is the paper's choice;
        the agent may explore alternatives and discover their effect.
  --channel {DAPI, Tubulin, Actin}
        Which channel's model to train (per-channel architecture).
  --epochs N   (hard-capped at --max-epochs, default 400)

Compute-awareness:
  * Reports estimated SU cost (epochs * ~0.0125 SU/epoch/channel, from measured
    ~45 s/epoch on GH200 and ~1 SU/node-hour) in the result.
  * Refuses epochs beyond the cap.
  * Writes/updates a run-level ledger (--ledger) of cumulative SU; refuses to
    submit if a per-run budget (--run-su-budget) would be exceeded.

Modes:
  submit (default): submit training, return {job_id, est_su, checkpoint_path}
  --status JOBID  : report job state and, if finished, the checkpoint path

Structured-result protocol as in the other tools.

Usage (submit):
  /usr/bin/python tool_train_ssl.py \
      --weak-label compound --channel DAPI --epochs 400 \
      --data-root ./runs/bbbc021_wsdino \
      --run-dir ./runs/agent_results/run1 \
      --ledger ./runs/agent_results/run1/su_ledger.json \
      --run-su-budget 100 \
      --result-out ./runs/agent_results/run1/train_dapi.json

Usage (status):
  /usr/bin/python tool_train_ssl.py --status 730812 \
      --run-dir ./runs/agent_results/run1 \
      --channel DAPI --result-out .../train_dapi_status.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = os.environ.get("AGENTCELL_TOOLS", os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.environ.get("WSDINO_REPO", os.path.expanduser("~/WS-DINO"))
SBATCH = f"{TOOLS_DIR}/train_wsdino_channel.sbatch"
CH_INDEX = {"DAPI": 0, "Tubulin": 1, "Actin": 2}
SU_PER_EPOCH = 0.0125  # ~45s/epoch on GH200, ~1 SU/node-hour


def emit(result, result_out):
    if result_out:
        Path(result_out).parent.mkdir(parents=True, exist_ok=True)
        Path(result_out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    sys.exit(0 if result.get("status") == "ok" else 1)


def read_ledger(path):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return {"cumulative_su": 0.0, "jobs": []}


def write_ledger(path, ledger):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(ledger, indent=2))


def squeue_state(job_id):
    """Return SLURM state string for a job, or 'COMPLETED'/'UNKNOWN'."""
    r = subprocess.run(["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                       capture_output=True, text=True)
    st = r.stdout.strip()
    if st:
        return st  # PENDING / RUNNING / etc.
    # not in queue -> finished (or never existed); check sacct if available
    r2 = subprocess.run(["sacct", "-j", str(job_id), "-n", "-o", "State", "-X"],
                        capture_output=True, text=True)
    s2 = r2.stdout.strip().split("\n")[0].strip() if r2.stdout.strip() else ""
    return s2 or "COMPLETED_OR_UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default=None, help="poll this job id instead of submitting")
    ap.add_argument("--weak-label", choices=["compound", "moa", "treatment", "none"],
                    default="compound")
    ap.add_argument("--channel", choices=["DAPI", "Tubulin", "Actin"], required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--data-root", default=os.environ.get("AGENTCELL_DATA", "./data/bbbc021_wsdino"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--run-su-budget", type=float, default=200.0)
    ap.add_argument("--inline", action="store_true",
                    help="train synchronously in this process (for agent-as-sbatch "
                         "Option 2) instead of submitting a separate sbatch job")
    ap.add_argument("--result-out", default=None)
    args = ap.parse_args()

    result = {"tool": "train_ssl", "status": "error",
              "decision": {"weak_label": args.weak_label, "channel": args.channel,
                           "epochs": args.epochs},
              "metrics": {}, "message": "", "artifacts": []}

    run_dir = Path(args.run_dir)
    ch_outdir = run_dir / f"train_{args.channel}_{args.weak_label}"
    ckpt = ch_outdir / f"{args.channel}_weak_compound_DINO_checkpoint.pth"

    # ---- STATUS mode ----
    if args.status:
        state = squeue_state(args.status)
        finished = state.startswith("COMPLETED") or state == "COMPLETED_OR_UNKNOWN"
        result["status"] = "ok"
        result["metrics"] = {"job_id": args.status, "job_state": state,
                             "finished": finished,
                             "checkpoint_exists": ckpt.exists()}
        if finished and ckpt.exists():
            result["artifacts"] = [str(ckpt)]
            result["message"] = f"Job {args.status} {state}; checkpoint ready."
        elif finished and not ckpt.exists():
            result["status"] = "error"
            result["message"] = (f"Job {args.status} finished ({state}) but no "
                                 f"checkpoint at {ckpt} — training may have failed.")
        else:
            result["message"] = f"Job {args.status} state={state}; not finished."
        emit(result, args.result_out)

    # ---- SUBMIT mode ----
    if args.epochs > args.max_epochs:
        result["message"] = (f"requested epochs {args.epochs} exceeds cap "
                             f"{args.max_epochs}; reduce epochs.")
        emit(result, args.result_out)

    est_su = round(args.epochs * SU_PER_EPOCH, 2)
    ledger = read_ledger(args.ledger)
    if ledger["cumulative_su"] + est_su > args.run_su_budget:
        result["message"] = (f"SU budget exceeded: cumulative {ledger['cumulative_su']:.1f} "
                             f"+ this {est_su} > budget {args.run_su_budget}. "
                             f"Choose fewer epochs or stop training.")
        result["metrics"] = {"est_su": est_su,
                             "cumulative_su": ledger["cumulative_su"],
                             "run_su_budget": args.run_su_budget}
        emit(result, args.result_out)

    ch_outdir.mkdir(parents=True, exist_ok=True)

    # ---- INLINE mode: train synchronously within the caller's allocation ----
    # Used when the agent harness itself runs as a long sbatch job (Option 2).
    # Avoids nested scheduling; the checkpoint is ready when this returns.
    if args.inline:
        env = dict(os.environ)
        if os.environ.get("PYTHONUSERBASE"):
            env["PYTHONUSERBASE"] = os.environ["PYTHONUSERBASE"]
        env["RANK"] = "0"; env["WORLD_SIZE"] = "1"
        env["MASTER_ADDR"] = "127.0.0.1"
        import random as _r
        env["MASTER_PORT"] = str(29520 + CH_INDEX[args.channel] + _r.randint(0, 2000))
        env["LOCAL_RANK"] = "0"
        weak_header = {"compound": "Unique_Compounds", "moa": "Unique_MoA",
                       "treatment": "Unique_Treatments",
                       "none": "Unique_Compounds"}[args.weak_label]
        # Warmups must not exceed total epochs (DINOLoss builds an array of
        # size epochs - warmup_teacher_temp_epochs). Scale them down for small
        # epoch counts; at the full 400 these are the paper's values (10).
        warmup_tt = min(10, max(0, args.epochs - 1))
        warmup_lr = min(10, max(0, args.epochs - 1))
        # init checkpoint must be present in the output dir
        init_src = Path(REPO_DIR) / "pretrain_full_checkpoint.pth"
        (ch_outdir / "pretrain_full_checkpoint.pth").write_bytes(init_src.read_bytes())
        cmd = ["/usr/bin/python", f"{REPO_DIR}/WS-DINO_BBBC021.py",
               "--data_path", args.data_root + "/BBBC021_annotated_wsdino.csv",
               "--output_dir", str(ch_outdir),
               "--channel_index", str(CH_INDEX[args.channel]),
               "--weak_label_header", weak_header,
               "--arch", "vit_small", "--patch_size", "8",
               "--batch_size_per_gpu", "16",
               "--epochs", str(args.epochs),
               "--warmup_epochs", str(warmup_lr),
               "--warmup_teacher_temp_epochs", str(warmup_tt),
               "--lr", "0.000004", "--min_lr", "0.000003",
               "--momentum_teacher", "0.99",
               "--weight_decay", "0.0", "--weight_decay_end", "0.0",
               "--clip_grad", "0",
               "--global_crops_scale", "0.1", "0.2",
               "--local_crops_scale", "0.04", "0.08",
               "--local_crops_number", "8",
               "--saveckp_freq", "50", "--num_workers", "8"]
        est_su = round(args.epochs * SU_PER_EPOCH, 2)
        ledger = read_ledger(args.ledger)
        if ledger["cumulative_su"] + est_su > args.run_su_budget:
            result["message"] = (f"SU budget exceeded inline: "
                                 f"{ledger['cumulative_su']:.1f}+{est_su} > "
                                 f"{args.run_su_budget}")
            emit(result, args.result_out)
        proc = subprocess.run(cmd, cwd=REPO_DIR, env=env)
        if proc.returncode != 0 or not ckpt.exists():
            result["message"] = (f"inline training failed (rc={proc.returncode}); "
                                 f"checkpoint {'present' if ckpt.exists() else 'missing'}")
            emit(result, args.result_out)
        ledger["cumulative_su"] = round(ledger["cumulative_su"] + est_su, 2)
        ledger["jobs"].append({"mode": "inline", "channel": args.channel,
                              "weak_label": args.weak_label,
                              "epochs": args.epochs, "est_su": est_su})
        write_ledger(args.ledger, ledger)
        result["status"] = "ok"
        result["artifacts"] = [str(ckpt)]
        result["metrics"] = {"mode": "inline", "est_su": est_su,
                             "cumulative_su": ledger["cumulative_su"],
                             "checkpoint": str(ckpt)}
        result["message"] = (f"Trained {args.channel} inline "
                             f"(weak_label={args.weak_label}, {args.epochs} epochs, "
                             f"{est_su} SU). Checkpoint at {ckpt}.")
        emit(result, args.result_out)

    # The sbatch reads CHANNEL_INDEX/CHTAG from env and copies the init ckpt.
    env = dict(os.environ)
    env["CHANNEL_INDEX"] = str(CH_INDEX[args.channel])
    env["CHTAG"] = args.channel
    # Override the sbatch's output dir + epochs + weak label by exporting vars
    # the sbatch is parameterized to read. We pass them via --export.
    export = (f"ALL,CHANNEL_INDEX={CH_INDEX[args.channel]},CHTAG={args.channel},"
              f"WSDINO_OUTDIR={ch_outdir},WSDINO_EPOCHS={args.epochs},"
              f"WSDINO_WEAKLABEL={args.weak_label},WSDINO_DATA={args.data_root}")
    cmd = ["sbatch", f"--job-name=agent_train_{args.channel}",
           f"--export={export}", SBATCH]
    p = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Submitted batch job (\d+)", p.stdout)
    if not m:
        result["message"] = f"sbatch submit failed: {p.stdout} {p.stderr}"
        emit(result, args.result_out)
    job_id = m.group(1)

    ledger["cumulative_su"] = round(ledger["cumulative_su"] + est_su, 2)
    ledger["jobs"].append({"job_id": job_id, "channel": args.channel,
                          "weak_label": args.weak_label, "epochs": args.epochs,
                          "est_su": est_su})
    write_ledger(args.ledger, ledger)

    result["status"] = "ok"
    result["metrics"] = {"job_id": job_id, "est_su": est_su,
                         "cumulative_su": ledger["cumulative_su"],
                         "expected_checkpoint": str(ckpt)}
    result["message"] = (f"Submitted training job {job_id} for {args.channel} "
                         f"(weak_label={args.weak_label}, {args.epochs} epochs, "
                         f"est {est_su} SU). Poll with --status {job_id}. "
                         f"Checkpoint will appear at {ckpt}.")
    result["artifacts"] = []
    emit(result, args.result_out)


if __name__ == "__main__":
    main()
