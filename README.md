# AgentCell

**An autonomous LLM agent that constructs image-based morphological-profiling pipelines from a natural-language goal, with a reproducible harness for studying agent behavior under a real compute budget.**

This repository accompanies the paper *"From Goal to State of the Art: Characterizing Autonomous LLM Agents that Construct Scientific Image-Analysis Pipelines."* It releases the agent harness, the primitive tool wrappers the agent composes, the SLURM scripts used to run everything on a GPU cluster, the figure-generation code, and the complete JSON decision trajectories from every run reported in the paper.

The task is mechanism-of-action (MoA) prediction on the BBBC021 Cell Painting benchmark, scored by not-same-compound (NSC) and not-same-compound-and-batch (NSCB) nearest-neighbor retrieval accuracy. Given only the goal *"maximize NSC on BBBC021,"* the agent autonomously chooses a backbone, decides whether and what to train, trains a weakly supervised self-supervised model per channel, re-embeds, selects a normalization, and stops — reaching NSC up to 0.971 (0.959 ± 0.009 across training seeds), matching a hand-tuned expert configuration.

---

## What this repository contains

```
agentcell/
├── README.md                  # this file
├── LICENSE                    # MIT
├── CITATION.cff               # how to cite
├── requirements.txt           # pinned Python dependencies
├── src/
│   ├── agent/
│   │   └── agent_full.py       # the autonomous agent (planner loop, budget ledger, earned-access)
│   └── tools/
│       ├── tool_embed.py       # primitive: compute treatment-level features
│       ├── tool_train_ssl.py   # primitive: train one WS-DINO channel (inline or sbatch)
│       └── tool_score.py       # primitive: score features (NSC / NSCB)
├── slurm/
│   ├── run_agent.sbatch        # run the agent as a long-lived GPU job
│   ├── train_wsdino_channel.sbatch  # parameterized single-channel training
│   └── seed_retrain.sbatch     # multi-seed retrain of a fixed configuration
├── scripts/
│   ├── preprocess_bbbc021.py   # raw TIFF -> illumination-corrected, resized, per-channel
│   ├── ablation_fixed_pipelines.py  # fixed (non-agent) baselines F1–F5
│   └── make_figures.py         # regenerate all paper figures from result numbers
├── trajectories/              # complete JSON decision traces from every run (see below)
├── results/                   # consolidated metric tables (CSV/JSON)
├── figures/                   # generated figures (PDF + PNG)
└── docs/
    ├── REPRODUCING.md          # step-by-step reproduction guide
    └── TOOLS.md                # the tool JSON contract and argument reference
```

> **Note on authoritative sources.** The code here is released as run for the paper. The tool wrappers are thin orchestration layers over the WS-DINO training/evaluation code of Cross-Zamirski et al. (2022); see *Dependencies* below for how to obtain that upstream code, which we do not redistribute.

---

## The agent in one paragraph

The agent runs as a long-lived job on a single GPU. A planner LLM receives the goal, the running history of decisions and observations, the remaining service-unit (SU) compute budget, and the list of checkpoints it has trained so far; it returns a single JSON action per step. Three primitive tools form the action space: `embed` (compute features with a chosen backbone over one or three channels), `train_ssl` (train one WS-DINO channel with a chosen weak label and epoch budget), and `score` (evaluate features under a chosen normalization). An **earned-access** rule forbids embedding with any trained checkpoint the agent did not itself produce earlier in the same run, forcing genuine end-to-end construction. Every thought, action, observation, and metric is logged to JSON; those logs are in `trajectories/`.

---

## Quick start

### 1. Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ is required. The pinned versions in `requirements.txt` match the cluster environment used for the paper (notably `albumentations==1.3.1`, which the upstream WS-DINO data pipeline depends on; later 2.x releases change the augmentation API).

### 2. Obtain the upstream WS-DINO code and BBBC021 data

This repository does not redistribute the WS-DINO training/evaluation code or the BBBC021 images. See `docs/REPRODUCING.md` for exact steps to:
- clone the WS-DINO repository and place it where the tools expect it;
- download the DINO ImageNet-pretrained ViT-S/8 initialization;
- stage the BBBC021 dataset and run `scripts/preprocess_bbbc021.py`.

### 3. Set API keys for the planner

```bash
export DEEPSEEK_API_KEY=...     # for the DeepSeek-V3 planner (deepseek-chat)
export OPENAI_API_KEY=...        # only needed for the GPT-5.5 planner comparison
```

### 4. Run the agent

On a cluster with SLURM:

```bash
sbatch --export=ALL,\
AGENT_GOAL="Build a BBBC021 mechanism-of-action pipeline that maximizes NSC accuracy (also report NSCB).",\
AGENT_WORKDIR="$SCRATCH/agentcell/runs/my_run",\
AGENT_STEPS=14,AGENT_SU=60 \
  slurm/run_agent.sbatch
```

Or directly on a GPU node:

```bash
python src/agent/agent_full.py \
  --goal "Build a BBBC021 mechanism-of-action pipeline that maximizes NSC accuracy (also report NSCB)." \
  --budget 14 --su-budget 60 \
  --workdir ./runs/my_run
```

The agent writes its trace to `<workdir>/agent_trace_<timestamp>.json` and prints each step as it goes.

---

## Reproducing the paper's results

Full instructions are in `docs/REPRODUCING.md`. In summary:

| Result | Command |
|---|---|
| Main run (NSC 0.971) | `run_agent.sbatch` with the goal above, DeepSeek planner, temp 0 |
| Robustness (temp 0.4) | same, with `AGENT_FLAGS="--temperature 0.4"` |
| Self-correction ablation | same, with `AGENT_FLAGS="--no-self-correct"` |
| Planner comparison (GPT-5.5) | same, with `AGENT_FLAGS="--endpoint https://api.openai.com/v1/chat/completions --llm-model gpt-5.5 --api-key-env OPENAI_API_KEY"` |
| Fixed-pipeline baselines (F1–F5) | `python scripts/ablation_fixed_pipelines.py --workdir ./runs/fixed` |
| Seed-variance study | `for S in 1 2 3; do sbatch --export=ALL,SEED=$S slurm/seed_retrain.sbatch; done` |
| Regenerate figures | `python scripts/make_figures.py` |

---

## Released trajectories

`trajectories/` contains the complete, unedited JSON decision traces for every run in the paper. Each subdirectory holds one run's `agent_trace_*.json` (the full step-by-step record of thoughts, actions, observations, and metrics), its `su_ledger.json` (the compute-budget accounting), and the per-step tool result JSONs.

| Directory | Run |
|---|---|
| `trajectories/main_run/` | Main DeepSeek run, temperature 0 (NSC 0.971) |
| `trajectories/robustness_temp04/` | DeepSeek, temperature 0.4 |
| `trajectories/ablation_self_correct/` | DeepSeek, `--no-self-correct` |
| `trajectories/planner_gpt55/` | GPT-5.5 planner |
| `trajectories/seed_runs/` | Four-seed retrain of the winning configuration |

See `trajectories/MANIFEST.md` for the exact file inventory and how to read a trace.

---

## Key results

| Configuration | NSC | NSCB |
|---|---|---|
| Best fixed pipeline (ViT, 3-channel, plain) | 0.913 | 0.699 |
| Agent (autonomous), best seed | **0.971** | 0.767 |
| Agent (autonomous), across 4 seeds | 0.959 ± 0.009 | 0.772 ± 0.006 |
| DeepSeek-V3 planner (100 epochs/channel) | 0.971 | 0.767 |
| GPT-5.5 planner (400 epochs/channel) | 0.942 | 0.777 |

The agent exceeds the best fixed pipeline by 5.8 NSC points. The difference between the two planners (0.971 vs 0.942) lies within the across-seed training variance (0.952–0.971), so it is attributable to self-supervised training stochasticity rather than planner capability.

---

## Dependencies and provenance

- **WS-DINO** (Cross-Zamirski et al., 2022, arXiv:2209.07819) — the weakly supervised self-supervised training and evaluation code the tools wrap. Obtain from the authors' repository; not redistributed here.
- **DINO** (Caron et al., ICCV 2021) — the ImageNet-pretrained ViT-S/8 initialization.
- **BBBC021** (Ljosa et al., Nat. Methods 2012; Caie et al., Mol. Cancer Ther. 2010) — the benchmark dataset, from the Broad Bioimage Benchmark Collection.

These are credited in full in the paper and in `CITATION.cff`.

---

## License

Code in this repository is released under the MIT License (see `LICENSE`). The released JSON trajectories and result tables are released under CC BY 4.0. Upstream WS-DINO code and the BBBC021 dataset retain their own respective licenses and are not redistributed here.

## Citing

If you use this code, the harness, or the released trajectories, please cite the paper as described in `CITATION.cff`.
