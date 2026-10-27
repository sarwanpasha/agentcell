#!/usr/bin/env python3
"""Run agent jobs on reserved GPUs, N concurrent runs per GPU. Usage: agent_runner.py SPEC --gpus 6,7 --per-gpu 3
spec lines: NAME|FLAGS. Runs whose final.json exists are skipped."""
import argparse, os, subprocess, threading, time, queue
from pathlib import Path
W = Path.home() / "agentcell_work"; S = W / "setup_files"
ap = argparse.ArgumentParser(); ap.add_argument("spec"); ap.add_argument("--gpus", required=True)
ap.add_argument("--per-gpu", type=int, default=3); a = ap.parse_args()
q = queue.Queue()
for line in open(a.spec):
    if "|" in line:
        n, f = line.rstrip("\n").split("|", 1); q.put((n, f))
log = open(W / "logs/agent_runner.log", "a")
def worker(g, k):
    time.sleep(k * 45)
    while True:
        try: n, f = q.get_nowait()
        except queue.Empty: return
        if (W / "runs/agent" / n / "final.json").exists(): continue
        e = dict(os.environ, NAME=n, GPU=str(g), FLAGS=f)
        log.write(f"{time.strftime('%H:%M:%S')} start {n} gpu{g}\n"); log.flush()
        with open(W / f"logs/agent_{n}.out", "a") as fo:
            rc = subprocess.call(["bash", str(S / "sb_agent.sbatch")], env=e, stdout=fo, stderr=subprocess.STDOUT)
        log.write(f"{time.strftime('%H:%M:%S')} end {n} rc={rc}\n"); log.flush()
ts = [threading.Thread(target=worker, args=(int(g), k)) for k in range(a.per_gpu) for g in a.gpus.split(",")]
[t.start() for t in ts]; [t.join() for t in ts]
