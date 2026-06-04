#!/usr/bin/env python3
"""
agent_full.py -- autonomous research agent over the FULL toolbox.

Runs as a long-lived sbatch job (Option 2) on a GPU node. The DeepSeek planner
is given a natural-language goal and three tools -- embed, train_ssl (inline),
score -- and must autonomously construct and optimize a BBBC021 MoA pipeline:
choose a model, decide whether/what to train (weak label, channel, epochs under
an SU budget), re-embed the trained model, score, compare, and converge.

This is the experiment for the systems contribution. Everything the agent does
is recorded to a JSONL trace for analysis (what it chose, why, what it recovered
from, where it landed relative to the 0.91 no-train baseline and the 0.97
hand-built expert ceiling).

Key properties:
  * Earned-trained-model rule: the agent may only embed with vit_trained using a
    checkpoint it produced via a prior train_ssl step in THIS run (tracked in
    earned_ckpts). No relaxation here -- this is a real autonomy run.
  * SU budget: a ledger caps cumulative training cost; the agent is told its
    remaining budget each step and must reason about it.
  * Inline training: train_ssl --inline blocks within this job (hours), so the
    agent's allocation must be long (submit this with --time 24:00:00).
  * --no-self-correct ablation flag: when set, the agent is NOT given tool error
    text (only "error"), to test the value of self-correction.

Submit as sbatch (see run_agent.sbatch). Requires DEEPSEEK_API_KEY in env.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import requests

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL_DEFAULT = "deepseek-chat"
TOOLS_DIR = os.environ.get("AGENTCELL_TOOLS", os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.environ.get("WSDINO_REPO", os.path.expanduser("~/WS-DINO"))
DATA_ROOT = os.environ.get("AGENTCELL_DATA", "./data/bbbc021_wsdino")
PY = "/usr/bin/python"


SYSTEM_PROMPT = """You are an autonomous research agent. Your goal is to build a
cell-image mechanism-of-action (MoA) pipeline on the BBBC021 benchmark that
maximizes NSC accuracy (also report NSCB). You operate real tools on a GPU
cluster, one tool call per step, observing each structured JSON result and
deciding the next action.

TOOLS:

1) embed -- compute treatment-level features from images.
   args: model    : "resnet50" | "vit_imagenet" | "vit_trained"
         channels : "all_three" | "dapi_only"
   For model="vit_trained", each channel is automatically embedded with its OWN
   trained checkpoint (the correct WS-DINO setup) -- you do NOT pass checkpoint
   paths. To embed all_three with trained models, you must FIRST train all three
   channels (DAPI, Tubulin, Actin) via train_ssl; embed then picks them up
   automatically. If you have only trained some channels, embed only those.
   Produces features under a tag derived from your choices.

2) train_ssl -- train a WS-DINO self-supervised model for ONE channel (inline,
   takes hours, costs SU from your budget).
   args: weak_label : "compound" | "moa" | "treatment"
         channel    : "DAPI" | "Tubulin" | "Actin"
         epochs     : integer (<=400)
   Returns the trained checkpoint path. To embed with a trained model you must
   train each channel you intend to use, then call embed model=vit_trained.

3) score -- score embedded features.
   args: tag           : the tag from a prior embed step
         normalization : "plain" | "per_channel_l2"
   Returns NSC and NSCB.

CONTEXT YOU SHOULD USE:
- An ImageNet ViT (no training) typically reaches ~0.91 NSC concatenated over
  all three channels. ResNet50 is a weaker, non-morphology-specific baseline.
- Training (WS-DINO) with the right weak label can push NSC toward ~0.97.
- All three channels (DAPI, Tubulin, Actin) carry complementary MoA signal;
  a single channel underuses it.
- The concatenation normalization trades NSC against NSCB.
- You have a limited SU training budget; spend it deliberately. Each step you
  will be told your remaining budget.

Reason scientifically. Establish baselines, form hypotheses, test them, compare
results, and converge on the best pipeline. Do not repeat an identical config.

Respond with ONLY one JSON object, no markdown:
{
  "thought": "one or two sentences of scientific reasoning",
  "tool": "embed" | "train_ssl" | "score" | "done",
  "args": { ... }
}
Declare "done" when you have a final best pipeline; in the thought, state the
configuration and its NSC/NSCB."""


@dataclass
class Step:
    idx: int
    thought: str
    tool: str
    args: dict
    observation: str = ""
    metrics: dict = field(default_factory=dict)
    status: str = ""
    duration_s: float = 0.0


@dataclass
class Trace:
    goal: str
    model: str
    self_correct: bool
    steps: list = field(default_factory=list)
    earned_ckpts: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def dump(self, path):
        Path(path).write_text(json.dumps(
            {"goal": self.goal, "model": self.model,
             "self_correct": self.self_correct, "timestamp": self.timestamp,
             "earned_ckpts": self.earned_ckpts,
             "steps": [asdict(s) for s in self.steps]}, indent=2))


def llm_plan(goal, trace, remaining_su, api_key, llm_model, temperature=0.0,
             endpoint=DEEPSEEK_URL, timeout=300):
    idx = len(trace.steps)
    history = []
    for s in trace.steps[-10:]:
        obs = ""
        if s.observation and trace.self_correct:
            obs = f"\n  observation: {s.observation}"
        elif s.observation and not trace.self_correct and s.status != "ok":
            obs = "\n  observation: error"  # ablation: hide error detail
        history.append(
            f"Step {s.idx}: {s.tool} {json.dumps(s.args)} -> "
            f"status={s.status} metrics={json.dumps(s.metrics)}{obs}")
    hist = "\n".join(history) if history else "(no steps yet)"
    earned = (f"Checkpoints you have trained this run (usable for vit_trained): "
              f"{trace.earned_ckpts}" if trace.earned_ckpts else
              "You have not trained any model yet.")
    user = (f"GOAL: {goal}\n\nREMAINING SU BUDGET: {remaining_su}\n{earned}\n\n"
            f"HISTORY:\n{hist}\n\nDecide the next action. JSON object only.")
    # Model-aware payload. GPT-5-series (and o-series) reasoning models reject
    # 'max_tokens' (need 'max_completion_tokens') and only accept default
    # temperature; they also spend hidden reasoning tokens, so give headroom.
    is_reasoning = any(llm_model.startswith(p) for p in
                       ("gpt-5", "o1", "o3", "o4"))
    payload = {"model": llm_model,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user}]}
    if is_reasoning:
        payload["max_completion_tokens"] = 3000
        # omit temperature (only default supported)
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = 600
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return Step(idx, f"LLM call failed: {e}", "done", {})
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return Step(idx, f"unparseable: {content[:150]}", "done", {})
    try:
        p = json.loads(m.group())
    except json.JSONDecodeError as e:
        return Step(idx, f"json error: {e}", "done", {})
    return Step(idx, p.get("thought", ""), p.get("tool", "done"),
                p.get("args", {}) or {})


def run_tool(step, trace, workdir, ledger_path, su_budget):
    t0 = time.time()
    env = dict(os.environ)
    if os.environ.get("PYTHONUSERBASE"):
        env["PYTHONUSERBASE"] = os.environ["PYTHONUSERBASE"]
    feat_dir = workdir / "features"

    if step.tool == "embed":
        model = str(step.args.get("model", "vit_imagenet"))
        channels = str(step.args.get("channels", "all_three"))
        tag = f"s{step.idx}_{model}_{channels}"
        step.args["_tag"] = tag
        cmd = [PY, f"{TOOLS_DIR}/tool_embed.py",
               "--data-root", DATA_ROOT, "--model", model,
               "--channels", channels, "--tag", tag,
               "--features-out", str(feat_dir),
               "--result-out", str(workdir / f"embed_{step.idx}.json")]
        if model == "vit_trained":
            # per-channel checkpoints live under the run workdir in
            # train_<CH>_<weak>/ subdirs; the tool auto-resolves each channel.
            cmd += ["--init-dir", str(workdir),
                    "--run-train-artifacts", str(workdir / "earned.json")]
    elif step.tool == "train_ssl":
        cmd = [PY, f"{TOOLS_DIR}/tool_train_ssl.py", "--inline",
               "--weak-label", str(step.args.get("weak_label", "compound")),
               "--channel", str(step.args.get("channel", "DAPI")),
               "--epochs", str(step.args.get("epochs", 400)),
               "--data-root", DATA_ROOT,
               "--run-dir", str(workdir),
               "--ledger", str(ledger_path),
               "--run-su-budget", str(su_budget),
               "--result-out", str(workdir / f"train_{step.idx}.json")]
    elif step.tool == "score":
        cmd = [PY, f"{TOOLS_DIR}/tool_score.py",
               "--features-dir", str(feat_dir),
               "--tag", str(step.args.get("tag", "")),
               "--normalization", str(step.args.get("normalization", "per_channel_l2")),
               "--result-out", str(workdir / f"score_{step.idx}.json")]
    else:
        step.status = "done"
        return step

    timeout = 22 * 3600 if step.tool == "train_ssl" else 3600
    try:
        p = subprocess.run(cmd, cwd=REPO_DIR, env=env, capture_output=True,
                           text=True, timeout=timeout)
        last = ""
        for line in p.stdout.strip().splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                last = line
                break
        if last:
            res = json.loads(last)
            step.status = res.get("status", "?")
            step.metrics = res.get("metrics", {})
            msg = res.get("message", "")
            if res.get("status") != "ok" and res.get("traceback"):
                msg += f" | trace: {res['traceback'][-200:]}"
            step.observation = msg[:500]
            # record earned checkpoint
            if step.tool == "train_ssl" and step.status == "ok":
                ck = res.get("artifacts", [])
                trace.earned_ckpts.extend(ck)
                (workdir / "earned.json").write_text(json.dumps(trace.earned_ckpts))
        else:
            step.status = "error"
            step.observation = ("no parseable result; stderr tail: " + p.stderr[-300:])
    except subprocess.TimeoutExpired:
        step.status = "timeout"
        step.observation = f"timed out after {timeout}s"
    step.duration_s = round(time.time() - t0, 1)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True)
    ap.add_argument("--budget", type=int, default=12, help="max agent steps")
    ap.add_argument("--su-budget", type=float, default=60.0,
                    help="cumulative training SU cap for this run")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--llm-model", default=MODEL_DEFAULT)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="planner sampling temperature (use >0 for robustness runs)")
    ap.add_argument("--endpoint", default=DEEPSEEK_URL,
                    help="OpenAI-compatible chat completions URL (for ablation 3, "
                         "point at another provider)")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY",
                    help="env var holding the API key for --endpoint")
    ap.add_argument("--no-self-correct", action="store_true",
                    help="ablation: hide tool error detail from the agent")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} not set")
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "features").mkdir(exist_ok=True)
    ledger_path = args.workdir / "su_ledger.json"
    trace_path = args.workdir / f"agent_trace_{datetime.now():%Y%m%d_%H%M%S}.json"
    trace = Trace(goal=args.goal, model=args.llm_model,
                  self_correct=not args.no_self_correct)

    print(f"FULL agent run. goal={args.goal}")
    print(f"step_budget={args.budget} su_budget={args.su_budget} "
          f"self_correct={trace.self_correct} llm={args.llm_model} "
          f"temp={args.temperature} endpoint={args.endpoint}")
    print(f"trace -> {trace_path}\n", flush=True)

    best = {"nsc": -1, "info": None}
    for _ in range(args.budget):
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"cumulative_su": 0}
        remaining = args.su_budget - ledger.get("cumulative_su", 0)
        step = llm_plan(args.goal, trace, remaining, api_key, args.llm_model,
                        temperature=args.temperature, endpoint=args.endpoint)
        if step.tool == "done":
            trace.steps.append(step)
            print(f"[{step.idx}] DONE: {step.thought}", flush=True)
            break
        step = run_tool(step, trace, args.workdir, ledger_path, args.su_budget)
        trace.steps.append(step)
        trace.dump(trace_path)
        nsc = step.metrics.get("NSC")
        if isinstance(nsc, (int, float)) and nsc > best["nsc"]:
            best = {"nsc": nsc, "info": dict(step.args),
                    "nscb": step.metrics.get("NSCB")}
        print(f"[{step.idx}] {step.tool} {json.dumps({k:v for k,v in step.args.items() if not k.startswith('_')})} "
              f"-> {step.status} NSC={step.metrics.get('NSC')} "
              f"NSCB={step.metrics.get('NSCB')} ({step.duration_s}s)", flush=True)
        print(f"      thought: {step.thought}", flush=True)

    trace.dump(trace_path)
    print(f"\n{'='*60}\nBest NSC: {best['nsc']}  (NSCB={best.get('nscb')})  "
          f"config={best['info']}\nTrace: {trace_path}", flush=True)


if __name__ == "__main__":
    main()
