#!/usr/bin/env python3
"""Run training and embedding tasks on the local GPUs without SLURM scheduling.

Each GPU runs up to --per-gpu trainings at a time; tasks are taken from one shared queue in
spec order, so faster GPUs take more work. After a channel finishes training, its checkpoint
embedding starts in the background on the same GPU. Finished channels (DONE marker) and existing
Gram files are skipped, so the runner can be restarted at any time.

Usage: gpu_runner.py SPECFILE [SPECFILE ...] --per-gpu N [--wait-free-mib M]
spec line: NAME WEAK EPOCHS SEED SAVEFREQ VENV(sys|251) PREP(cp|um|lincs)
"""
import argparse, os, queue, subprocess, threading, time, random
from pathlib import Path

W = Path.home() / "agentcell_work"; S = W / "setup_files"
DATA = {"cp": W / "data/bbbc021_wsdino", "um": W / "data/bbbc021_wsdino_unitmean", "lincs": W / "data/lincs_wsdino"}
ap = argparse.ArgumentParser()
ap.add_argument("specs", nargs="+"); ap.add_argument("--per-gpu", type=int, default=1)
ap.add_argument("--wait-free-mib", type=int, default=22000)
ap.add_argument("--gpus", default=None, help="comma-separated GPU indices (default: all)")
a = ap.parse_args()
ngpu = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
q = queue.Queue()
for spec in a.specs:
    for line in open(spec):
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        name, weak, ep, seed, sf, ve, pr = f
        for ch in ("DAPI", "Tubulin", "Actin"):
            q.put((name, weak, int(ep), seed, sf, ve, pr, ch))
log = open(W / "logs/gpu_runner.log", "a")


def say(*x):
    log.write(time.strftime("%H:%M:%S ") + " ".join(map(str, x)) + "\n"); log.flush()


def free_mib(g):
    out = subprocess.check_output(["nvidia-smi", "-i", str(g), "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"])
    return int(out.decode().strip())


def env_for(g, name, weak, ep, seed, sf, ve, pr, ch):
    e = dict(os.environ)
    e.update(SEED=seed, CH=ch, EPOCHS=str(ep), OUTBASE=str(W / "runs/rev" / name), GPU=str(g), WEAK=weak,
             SAVEFREQ=sf, VENV=str(W / ("venv251" if ve == "251" else "venv")), DATA_OVERRIDE=str(DATA[pr]),
             SLURM_JOB_ID=str(random.randint(1, 10 ** 6)), TRAIN_DIR=str(W / "runs/rev" / name / f"train_{ch}_{weak}"),
             RUN=name)
    if ep == 0:
        e["MODEL"] = weak
    return e


LOCKS = {}


def worker(g, slot):
    time.sleep(slot * 20 + g)
    while True:
        try:
            t = q.get_nowait()
        except queue.Empty:
            return
        name, weak, ep, seed, sf, ve, pr, ch = t
        e = env_for(g, *t)
        tag = f"{name}_{ch}"
        tdir = Path(e["TRAIN_DIR"]); stf = tdir / "STARTED"
        if ep and stf.exists() and not (tdir / "DONE").exists():
            try:
                os.kill(int(stf.read_text().split()[0]), 0); say("skip running", tag); continue
            except (OSError, ValueError):
                pass
        if ep:
            with LOCKS[g]:
                while free_mib(g) < a.wait_free_mib:
                    time.sleep(60)
                say("train start", tag, "gpu", g)
                fo = open(W / f"logs/run_tr_{tag}.out", "a")
                p = subprocess.Popen(["bash", str(S / "sb_train.sbatch")], env=e, stdout=fo, stderr=subprocess.STDOUT)
                tdir.mkdir(parents=True, exist_ok=True); stf.write_text(f"{p.pid} gpu{g}\n")
                time.sleep(120)   # let memory use settle before the next start on this GPU
            rc = p.wait()
            say("train end", tag, "rc", rc)
            if rc != 0:
                continue
        fo = open(W / f"logs/run_em_{tag}.out", "a")
        subprocess.Popen(["bash", str(S / "sb_embed.sbatch")], env=e, stdout=fo, stderr=subprocess.STDOUT)
        say("embed launched", tag, "gpu", g)
        if not ep:
            time.sleep(30)


LOCKS.update({g: threading.Lock() for g in range(ngpu)})
GPUS = [int(x) for x in a.gpus.split(",")] if a.gpus else list(range(ngpu))
threads = [threading.Thread(target=worker, args=(g, s)) for s in range(a.per_gpu) for g in GPUS]
for t in threads:
    t.start()
for t in threads:
    t.join()
say("all trainings finished")
