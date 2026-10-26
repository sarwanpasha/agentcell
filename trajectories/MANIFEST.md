# Released decision trajectories

Each subdirectory contains the complete, unedited record of one run reported in
these experiments. These are the agent's own logs, copied verbatim from the run output
directories.

## Files in each run directory
- `agent_trace_<timestamp>.json` — the full trajectory: an ordered list of steps,
  each with the planner's `thought`, the chosen `tool` and `args`, the observed
  `status`/`metrics`/`observation`, and timing. This is the primary artifact.
- `su_ledger.json` — the service-unit budget accounting (cumulative cost and a
  record of each training job the agent launched).
- `embed_<step>.json`, `train_<step>.json`, `score_<step>.json` — the individual
  tool result envelopes observed at each step (the same JSON the agent saw).
- `<run>.out` / `<run>.err` — the SLURM stdout/stderr for the job.

## Run inventory
| Directory | Planner | Flags | Headline |
|---|---|---|---|
| `main_run/`              | DeepSeek-V3 | temp 0                  | NSC 0.971 |
| `robustness_temp04/`     | DeepSeek-V3 | temp 0.4                | NSC 0.971 |
| `ablation_self_correct/` | DeepSeek-V3 | --no-self-correct       | NSC 0.971 (null ablation) |
| `planner_gpt55/`         | GPT-5.5     | 400 epochs/channel      | NSC 0.942 (within seed noise) |
| `seed_runs/`             | n/a (fixed config) | seeds 1–3        | seed-variance study |

## How to read a trace
A trace is a JSON object with a `steps` array. To print the agent's decision
path:

```bash
python - << 'PY'
import json, glob
f = sorted(glob.glob("main_run/agent_trace_*.json"))[0]
t = json.load(open(f))
for s in t["steps"]:
    m = s.get("metrics", {})
    print(f"[{s['idx']}] {s['tool']:9} NSC={m.get('NSC')}  {s['thought'][:80]}")
PY
```

## Note for this release
The trajectory JSON files are copied from the run output directories on the
compute cluster. Place the real files from each run into the matching directory
above before publishing the repository; this directory currently contains only
this manifest and per-run README stubs describing the expected contents.
